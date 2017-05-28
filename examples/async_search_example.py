import numbers
import time
from collections import defaultdict

import numpy as np
import toolz as tz
from dask.delayed import Delayed, tokenize
from distributed import Client, as_completed
from scipy import stats
from sklearn import model_selection
from sklearn.datasets import load_digits
from sklearn.metrics.scorer import check_scoring
from sklearn.svm import SVC

from dask_searchcv.model_selection import build_graph, \
    update_graph, DaskBaseSearchCV, generate_results, _normalize_n_jobs


@tz.curry
def _simple_criterion(threshold, scores):
    return max(scores) > threshold


def _test_score(*scores):
    """scores: ((test_score0, train_score0), ...) or
    (test_score0, test_score1, ...)"""
    if isinstance(scores[0], tuple):  # return_train_score
        return np.mean([_[0] for _ in scores])
    else:
        return np.mean(scores)


class AsyncRandomizedSearchCV(DaskBaseSearchCV):
    def __init__(self, estimator, param_distributions, n_iter=10, random_state=None,
                 client=None, criterion=_simple_criterion(threshold=0.9),
                 scoring=None, iid=True, refit=True, cv=None, error_score='raise',
                 return_train_score=True, scheduler=None, n_jobs=-1, cache_cv=True):
        """
        criterion : a callable taking a list of scores and returning bool
        """
        super(AsyncRandomizedSearchCV, self).__init__(
            estimator=estimator,
            scoring=scoring, iid=iid,
            refit=refit, cv=cv,
            error_score=error_score,
            return_train_score=return_train_score,
            scheduler=scheduler,
            n_jobs=n_jobs,
            cache_cv=cache_cv,
        )
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state
        self._criterion = criterion
        if client is None:
            client = Client()
        self._client = client
        self._job_map = {}

    def _get_param_iterator(self):
        """Return ParameterSampler instance for the given distributions"""
        return model_selection.ParameterSampler(self.param_distributions,
                                                self.n_iter,
                                                random_state=self.random_state)

    def fit_async(self, X, y=None, groups=None, **fit_params):
        if not (isinstance(self.error_score, numbers.Number) or
                        self.error_score == 'raise'):
            raise ValueError("error_score must be the string 'raise' or a"
                             " numeric value.")
        estimator = self.estimator
        self.scorer_ = check_scoring(estimator, scoring=self.scoring)
        ncores = len(self._client.ncores())

        # build graph
        (dsk, cv_name, X_name, y_name, n_splits, fit_params, weights,
         next_param_token, next_token) = build_graph(
            estimator, self.cv, X, y, groups, fit_params, iid=self.iid,
            error_score=self.error_score, return_train_score=self.return_train_score,
            cache_cv=self.cache_cv
        )
        _update_graph = tz.curry(update_graph)(dsk, next_param_token, next_token,
                                               estimator, cv_name, X_name, y_name,
                                               fit_params, n_splits,
                                               self.error_score, self.scorer_,
                                               self.return_train_score)

        def _update_graph_with_test_scores(params):
            scores = _update_graph(params)
            scores = list(tz.groupby(1, scores).values())
            mean_test_scores = ['test-score-{}'.format(tokenize(*v)) for v in scores]
            for name, v in zip(mean_test_scores, scores):
                dsk[name] = (_test_score,) + tuple(v)
            return list(tz.concat(scores)), mean_test_scores

        # fill cluster with jobs
        params_iter = iter(self._get_param_iterator())
        candidate_params = [next(params_iter) for _ in range(int(ncores * 2))]

        next_token.counts = defaultdict(int)
        cv_scores, test_scores = _update_graph_with_test_scores(candidate_params)
        fs = [client.compute(Delayed(k, dsk)) for k in test_scores]
        self._job_map = {k: f for k, f in zip(fs, candidate_params)}

        af = as_completed(fs)
        all_params, all_scores = candidate_params, cv_scores
        test_scores = []

        # adding jobs as completed
        for future in af:
            params, test_score = self._job_map[future], future.result()
            test_scores.append(test_score)
            if self._criterion(scores=test_scores):
                # we cancel the remaining jobs
                nleft = len(af.futures)
                for f in af.futures:
                    del self._job_map[f]
                client.cancel(af.futures)
                all_params = all_params[:-nleft]
                all_scores = all_scores[:-n_splits * nleft]
                break

            p = next(params_iter)
            cv_scores, test_score_names = _update_graph_with_test_scores([p])
            f = client.compute(Delayed(test_score_names[0], dsk))
            af.add(f)
            self._job_map[f] = params
            all_params.append(p)
            all_scores.extend(cv_scores)

        # finalize results
        main_token = next_token.token
        keys = generate_results(dsk, estimator, all_scores, main_token, X_name,
                                y_name, all_params, n_splits, self.error_score,
                                weights, self.refit, fit_params)

        self.dask_graph_ = dsk
        self.n_splits_ = n_splits
        n_jobs = _normalize_n_jobs(self.n_jobs)
        scheduler = self._client.get
        out = scheduler(dsk, keys, num_workers=n_jobs)
        self.cv_results_ = results = out[0]
        self.best_index_ = np.flatnonzero(results["rank_test_score"] == 1)[0]
        if self.refit:
            self.best_estimator_ = out[1]
        return self
    

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
    client = Client()

    search = AsyncRandomizedSearchCV(
        estimator=model,
        param_distributions=param_space,
        cv=n_splits,
        n_iter=n_iter,
        random_state=random_state,
        client=client
    )

    X = digits.data
    y = digits.target
    fit_params = {}

    start_t = time.time()
    search.fit_async(digits.data, digits.target)

    client.shutdown()

    print("Search finished")
    print("best_score {}; best_params - {}".format(search.best_score_,
                                                   search.best_params_))
    print("Async fit took {:.3f} seconds".format(time.time()-start_t))

