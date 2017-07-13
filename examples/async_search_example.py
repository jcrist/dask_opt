import time

from distributed import Client
from scipy import stats
from sklearn.datasets import load_digits
from sklearn.svm import SVC

from dask_searchcv.async_model_selection import AsyncRandomizedSearchCV, \
    CachingPlugin

if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y/%m/%d %I:%M:%S')

    digits = load_digits()

    model = SVC(kernel='rbf')

    # ended up tuning these parameters a bit :-P
    param_space = {'C': stats.expon(0.00001, 20),
                   'gamma': stats.expon(0.00001, 0.5),
                   'class_weight': [None, 'balanced']}

    n_splits = 3
    n_iter = 100000
    random_state = 1
    client = Client()

    client.run_on_scheduler(
        lambda dask_scheduler: dask_scheduler.add_plugin(
            CachingPlugin(dask_scheduler)))

    search = AsyncRandomizedSearchCV(
        estimator=model,
        param_distributions=param_space,
        cv=n_splits,
        n_iter=n_iter,
        random_state=random_state,
        client=client,
        threshold=0.9
    )

    X = digits.data
    y = digits.target
    fit_params = {}

    start_t = time.time()
    search.fit_async(digits.data, digits.target)

    print("Search finished")
    print("best_score {}; best_params - {}".format(search.best_score_,
                                                   search.best_params_))
    print("Async fit took {:.3f} seconds".format(time.time()-start_t))

    client.shutdown()
