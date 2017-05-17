import numbers

import numpy as np
from dask.base import tokenize
from distributed import Client, as_completed
from scipy import stats
from sklearn.datasets import load_digits
from sklearn.metrics.scorer import check_scoring
from sklearn.svm import SVC

from dask_searchcv.model_selection import RandomizedSearchCV, build_graph, \
    update_graph


def fit_async(search, X, y=None, groups=None, threshold=0.9, client=None, **fit_params):
    if client is None:
        client = Client()

    estimator = search.estimator
    search.scorer_ = check_scoring(estimator, scoring=search.scoring)
    error_score = search.error_score
    if not (isinstance(error_score, numbers.Number) or
                    error_score == 'raise'):
        raise ValueError("error_score must be the string 'raise' or a"
                         " numeric value.")

    params_iter = iter(search._get_param_iterator())

    (dsk, cv_name, X_name, y_name, n_splits, fit_params, weights,
     next_param_token, next_token) = build_graph(
        estimator, search.cv, X, y,
        groups, fit_params,
        iid=search.iid,
        error_score=error_score,
        return_train_score=search.return_train_score,
        cache_cv=search.cache_cv)

    ncores = len(client.ncores())

    candidate_params = [next(params_iter) for _ in range(ncores * 2)]

    def _test_score(*scores):
        """scores: ((test_score0, train_score0), ...) or (test_score0, test_score1, ...)"""
        if isinstance(scores[0], tuple):  # return_train_score
            return np.mean([_[0] for _ in scores])
        else:
            return np.mean(scores)

    fs, job_map = [], {}
    for p in candidate_params:
        cv_scores = update_graph(dsk, next_param_token, next_token, estimator,
                                 cv_name,
                                 X_name, y_name, [p], fit_params,
                                 n_splits,
                                 error_score, search.scorer_,
                                 search.return_train_score)
        test_score_name = 'test_score_{}'.format(tokenize(p))
        dsk[test_score_name] = (_test_score,) + tuple(cv_scores)
        f = client._graph_to_futures(dsk, [test_score_name])[test_score_name]
        fs.append(f)
        job_map[f] = p

    results = {}
    af = as_completed(fs)

    best_score, best_params = -np.inf, None

    for future in af:
        test_score, params = future.result(), job_map[future]

        results[future] = test_score

        if test_score > best_score:
            best_score = test_score
            best_params = params
            print('Best score now: {}, for parameters {}'.format(best_score,
                                                                 best_params))
            if best_score > threshold:
                break
                # if criterion(results):
                #     break

        p = next(params_iter)
        cv_scores = update_graph(dsk, next_param_token, next_token, estimator,
                                      cv_name,
                                      X_name, y_name, [p], fit_params,
                                      n_splits,
                                      error_score, search.scorer_,
                                      search.return_train_score)
        test_score_name = 'test_score_{}'.format(tokenize(p))
        dsk[test_score_name] = (_test_score,) + tuple(cv_scores)
        f = client._graph_to_futures(dsk, [test_score_name])[test_score_name]
        af.add(f)
        job_map[f] = params

    client.shutdown()

    return best_score, best_params


if __name__ == '__main__':
    digits = load_digits()

    model = SVC(kernel='rbf')

    # ended up tuning these parameters a bit :-P
    param_space = {'C': stats.expon(0.00001, 20),
                   'gamma': stats.expon(0.00001, 0.5),
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

    client = Client()
    best_score, best_params = fit_async(
        search, digits.data, digits.target, groups, threshold, client
    )

    print("Search finished")
    print("best_score {}; best_params - {}".format(best_score, best_params))


