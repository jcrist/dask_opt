import logging
import numbers
import dask_searchcv.online_model_selection as oms
import numpy as np
import toolz as tz
from dask_searchcv.online_model_selection import RandomizedSearchCV
from distributed import Client, as_completed
from sklearn.datasets import load_digits
from sklearn.metrics.scorer import check_scoring
from sklearn.svm import SVC
from scipy import stats

logger = logging.getLogger(__name__)


def fit_async(
    search, params_iter, criterion, X, y=None, groups=None, client=None, **fit_params
):
    if client is None:
        client = Client()

    estimator = search.estimator
    search.scorer_ = check_scoring(estimator, scoring=search.scoring)
    error_score = search.error_score
    if not (isinstance(error_score, numbers.Number) or
                    error_score == 'raise'):
        raise ValueError("error_score must be the string 'raise' or a"
                         " numeric value.")

    dsk, X_name, y_name, cv_name, n_splits = oms.build_graph(
        estimator, X, y, search.cv, groups, search.cache_cv)

    search.dask_graph_ = dsk
    search.n_splits_ = n_splits

    ncores = len(client.ncores())

    futures = []
    job_map = {}

    best_score, best_params = -np.inf, {}

    for _ in range(ncores * 2):
        params = next(params_iter)
        cv_score_names = oms.update_graph(dsk, estimator, X_name, y_name, params, fit_params,
                                          cv_name, n_splits,
                                          search.scorer_, search.return_train_score,
                                          error_score=error_score)

        fs = client._graph_to_futures(dsk, cv_score_names)
        score_name = 'score_{}'.format(oms.tokenize(params))
        dsk[score_name] = (list, cv_score_names)
        _, f = client._graph_to_futures(dsk, [score_name]).popitem()

        futures.append(f)
        job_map[f] = params

    results = {}
    af = as_completed(futures)
    for future in af:
        scores, params = future.result(), job_map[future]
        test_score = np.mean(list(tz.pluck(0, scores))) if search.return_train_score else np.mean(
            scores)
        results[future] = test_score

        if test_score > best_score:
            best_score = test_score
            best_params = params
            logger.info('Best score now: {}, for parameters {}'.format(best_score, best_params))
            if best_score > threshold:
                break
            # if criterion(results):
            #     break
        params = next(params_iter)
        cv_score_names = oms.update_graph(dsk, estimator, X_name, y_name, params, fit_params,
                                          cv_name, n_splits,
                                          search.scorer_, search.return_train_score,
                                          error_score=error_score)

        fs = client._graph_to_futures(dsk, cv_score_names)
        score_name = 'score_{}'.format(oms.tokenize(params))
        dsk[score_name] = (list, cv_score_names)
        _, f = client._graph_to_futures(dsk, [score_name]).popitem()
        af.add(f)
        job_map[f] = params

    client.shutdown()

    return best_score, best_params


@tz.curry
def simple_criterion(threshold, results):
    """Example stopping condition could be passed to fit_async"""
    return max(results.values()) > threshold


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    digits = load_digits()

    model = SVC(kernel='rbf')

    n_splits = 3
    param_space = {'C': np.linspace(0.00001, 1),
                   'gamma': stats.expon(0.00001, 1),
                   'class_weight': [None, 'balanced']}
    n_splits = 3
    n_iter = 100000
    random_state = 1

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_space,
        cv=n_splits,
        n_iter=n_iter,
        random_state=random_state
    )
    X = digits.data
    y = digits.target
    groups = None
    client = None
    fit_params = {}
    threshold = 0.9
    params_iter = iter(search._get_param_iterator())
    best_score = -np.inf

    client = Client()
    best_score, best_params = fit_async(
        search, params_iter, None, digits.data, digits.target, client=client
    )

    logger.info("Search finished")
    logger.info("best_score {}; best_params - {}".format(best_score, best_params))


