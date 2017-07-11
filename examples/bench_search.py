import multiprocessing
from time import time

import pandas as pd
import traces
from distributed import Client, LocalCluster
from distributed.diagnostics.eventstream import EventStream
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline

import dask_searchcv as dcv
from dask_searchcv.async_model_selection import AsyncGridSearchCV
from dask_searchcv.async_model_selection import CachingPlugin


# utilities
def bf_to_ss(bf):
    ss = bf
    ss = ss.startstops.apply(
        lambda l: pd.Series({v[0]: [v[1], v[2]] for v in l})).set_index(ss.key)
    ss = ss.reset_index().melt(value_vars=['compute', 'transfer'],
                               id_vars=['key']).dropna().set_index(
        ['key', 'variable']).value.apply(
        lambda l: pd.Series({'start': l[0], 'stop': l[1]}))
    return ss.sort_values('start')


def aggregate_ss(ss):
    L = []
    x = ss.iloc[0]
    for i, xi in ss.iloc[1:].iterrows():
        if x is None:
            x = xi
            continue
        if xi.start < x.stop:
            x = pd.Series([x.start, max(xi.stop, x.stop)], x.index, name=(x.name, xi.name))
        else:
            L.append(x.copy())
            x = None
    agg_ss = pd.DataFrame(L)
    return agg_ss


def set_index(s, index):
    s.index = index
    return s


def get_stats_from_ss(ss):
    df = pd.concat([
        pd.DataFrame(dict(t=ss.start, dx=1)), pd.DataFrame(dict(t=ss.stop, dx=-1))])
    w = df.sort_values('t')['dx'].cumsum().pipe(
        lambda s: set_index(s, df.t-df.t.iloc[0])).sort_index()
    x = traces.TimeSeries(data=w)
    s = pd.Series(*reversed(list(zip(*x.sample(0.1)))))
    return x.mean(), (s == 0).mean()


def get_stats_for_eventstream(bf):
    return pd.DataFrame(
        {worker: get_stats_from_ss(bf_to_ss(bf[bf.worker == worker])) for worker in
         bf.worker.unique()}, index=['mean_occupancy', 'mean_vacancy'])


def get_duration_for_eventstream(bf):
    ss = bf_to_ss(bf)
    return ss.stop.max() - ss.start.min()


def bench_search(version, lparam, rstate, refit, occupancy):
    categories = [
        'alt.atheism',
        'talk.religion.misc',
    ]

    # Uncomment the following to do the analysis on all the categories
    # categories = None
    data = fetch_20newsgroups(
        subset='train', categories=categories, data_home='/data/scikit_learn_data')
    test_data = fetch_20newsgroups(
        subset='test', categories=categories, data_home='/data/scikit_learn_data')

    parameters = [
        ('vect__max_df', (0.5, 0.75, 1.0)),
        ('vect__ngram_range', ((1, 1), (1, 2))),
        ('tfidf__use_idf', (True, False)),
        ('tfidf__norm', ('l1', 'l2')),
        ('clf__alpha', (1e-2, 1e-3, 1e-4, 1e-5)),
        ('clf__n_iter', (10, 50, 80)),
        ('clf__penalty', ('l2', 'elasticnet')),
    ]

    parameter_selection = [0, 3, 1, 4, 2, 5, 6]

    param_grid = dict([parameters[i] for i in parameter_selection[:lparam]])

    refit = {True: 1, False: 0}[refit]

    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(random_state=rstate)),
    ])

    if version > 0:
        # , diagnostics_port=None)
        # n_workers=4, threads_per_worker=2)
        cluster = LocalCluster(diagnostics_port=None)
        client = Client(address=cluster.scheduler.address)
        event_stream = EventStream(scheduler=cluster.scheduler)

        if version == 3:
            client.run_on_scheduler(
                lambda dask_scheduler: dask_scheduler.add_plugin(
                    CachingPlugin(dask_scheduler)))

    if version == 0:
        search = GridSearchCV(
            pipeline, param_grid, n_jobs=-1, verbose=1, refit=refit)
    elif version == 1:
        search = dcv.GridSearchCV(
            pipeline, param_grid, scheduler=client, refit=refit)
    else:  # version == 2 or version == 3:
        search = AsyncGridSearchCV(pipeline, param_grid, threshold=1.1,
                                   client=client, refit=refit,
                                   occupancy_factor=occupancy)

    print("Fitting with {} parameters".format(len(ParameterGrid(search.param_grid))))

    start_t = time()
    if version <= 1:
        search.fit(data.data, data.target)
    else:
        search.fit_async(data.data, data.target)

    if refit:
        print("Fit results: {}".format(
            (search.score(data.data, data.target),
             search.score(test_data.data, test_data.target))))

    fit_duration = (time() - start_t)
    print("Fit took: {}".format(fit_duration))

    results = {
        'start_time': start_t,
        'fit_duration': fit_duration,
        'ncpu': multiprocessing.cpu_count()
    }

    if version > 0:
        # output information about the event-stream
        bf = pd.DataFrame(event_stream.buffer)

        occupancy_series = []
        for i in range(len(bf.worker.unique())):
            ss_async = bf_to_ss(
                bf[bf.worker == bf.worker.unique()[i]])
            agg_ss = aggregate_ss(ss_async)
            occupancy_series.append((agg_ss.stop - agg_ss.start).sum() / (
                agg_ss.iloc[-1].stop - agg_ss.iloc[0].start))

        print("occupancy:")
        print(pd.Series(occupancy_series).describe())

        # print("transfer-ratio: {}".format())

        # and finally
        client.shutdown()

        results['events'] = bf

    return results
