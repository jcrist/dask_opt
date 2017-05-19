import numbers

import numpy as np
import toolz as tz
from dask.base import tokenize
from distributed import Client, as_completed, wait
from scipy import stats
from sklearn import model_selection
from sklearn.datasets import load_digits
from sklearn.metrics.scorer import check_scoring
from sklearn.svm import SVC

from dask_searchcv.model_selection import build_graph, \
    update_graph, DaskBaseSearchCV, generate_results, _normalize_n_jobs, _normalize_scheduler


# note: could probably include other information in this via a results dict/object containing
# timestamps etc
@tz.curry
def _simple_criterion(threshold, scores):
    return max(scores) > threshold


class AsyncRandomizedSearchCV(DaskBaseSearchCV):
    def __init__(self, estimator, param_distributions, n_iter=10, random_state=None, client=None,
                 criterion=_simple_criterion(threshold=0.9), scoring=None, iid=True, refit=True,
                 cv=None, error_score='raise', return_train_score=True,
                 scheduler=None, n_jobs=-1, cache_cv=True):
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

    def _get_param_iterator(self):
        """Return ParameterSampler instance for the given distributions"""
        return model_selection.ParameterSampler(self.param_distributions,
                                                self.n_iter,
                                                random_state=self.random_state)

    def fit_async(self, X, y=None, groups=None, **fit_params):
        estimator = self.estimator
        self.scorer_ = check_scoring(estimator, scoring=self.scoring)
        error_score = self.error_score
        if not (isinstance(error_score, numbers.Number) or
                        error_score == 'raise'):
            raise ValueError("error_score must be the string 'raise' or a"
                             " numeric value.")

        params_iter = iter(self._get_param_iterator())

        (dsk, cv_name, X_name, y_name, n_splits, fit_params, weights,
         next_param_token, next_token) = build_graph(
            estimator, self.cv, X, y,
            groups, fit_params,
            iid=self.iid,
            error_score=error_score,
            return_train_score=self.return_train_score,
            cache_cv=self.cache_cv)

        ncores = len(self._client.ncores())
        candidate_params = [next(params_iter) for _ in range(ncores * 2)]

        fs = []
        self._job_map = {}

        for p in candidate_params:
            cv_scores = update_graph(dsk, next_param_token, next_token, estimator,
                                     cv_name,
                                     X_name, y_name, [p], fit_params,
                                     n_splits,
                                     error_score, self.scorer_,
                                     self.return_train_score)
            test_score_name = 'test_score_{}'.format(tokenize(p))
            dsk[test_score_name] = (_test_score,) + tuple(cv_scores)
            f = self._client._graph_to_futures(dsk, [test_score_name])[test_score_name]
            fs.append(f)
            self._job_map[f] = p

        results = {}
        af = as_completed(fs)
        all_params = []
        all_scores = []
        best_score, best_params = -np.inf, None

        for future in af:
            test_score, params = future.result(), self._job_map[future]

            if test_score > best_score:
                best_score = test_score
                best_params = params
                print('Best score now: {}, for parameters {}'.format(best_score,
                                                                     best_params))

            results[future] = test_score
            if self._criterion(scores=list(results.values())):
                # todo: should decide whether to discard unfinished jobs if this
                # score is good enough
                wait(af.futures)
                break

            p = next(params_iter)
            all_params.append(p)
            cv_scores = update_graph(dsk, next_param_token, next_token, estimator,
                                     cv_name,
                                     X_name, y_name, [p], fit_params,
                                     n_splits,
                                     error_score, self.scorer_,
                                     self.return_train_score)
            all_scores.extend(cv_scores)
            test_score_name = 'test_score_{}'.format(tokenize(p))
            dsk[test_score_name] = (_test_score,) + tuple(cv_scores)
            f = self._client._graph_to_futures(dsk, [test_score_name])[test_score_name]
            af.add(f)
            self._job_map[f] = params

        main_token = next_token.token
        # we only take the completed jobs ... not sure if this is the best way
        # scores = list(results.values())
        # all_params = [self._job_map[f] for f in results.keys()]

        keys = generate_results(dsk, estimator, all_scores, main_token, X_name, y_name,
                                all_params, n_splits, error_score, weights, self.refit,
                                fit_params)

        self.dask_graph_ = dsk
        self.n_splits_ = n_splits

        n_jobs = _normalize_n_jobs(self.n_jobs)
        scheduler = client.get

        out = scheduler(dsk, keys, num_workers=n_jobs)

        self.cv_results_ = results = out[0]
        self.best_index_ = np.flatnonzero(results["rank_test_score"] == 1)[0]

        if self.refit:
            self.best_estimator_ = out[1]
        return self


def _test_score(*scores):
    """scores: ((test_score0, train_score0), ...) or (test_score0, test_score1, 
    ...)"""
    if isinstance(scores[0], tuple):  # return_train_score
        return np.mean([_[0] for _ in scores])
    else:
        return np.mean(scores)


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

    search.fit_async(digits.data, digits.target)

    print("Search finished")
    print("best_score {}; best_params - {}".format(search.best_score_, search.best_params_))
