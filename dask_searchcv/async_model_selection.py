import numbers
from collections import defaultdict

import numpy as np
import toolz as tz
from dask.delayed import tokenize, Delayed
from distributed import Client, as_completed
from sklearn import model_selection
from sklearn.metrics.scorer import check_scoring

from dask_searchcv.model_selection import DaskBaseSearchCV, build_graph, \
    update_graph, generate_results, _normalize_n_jobs

import logging

logger = logging.getLogger(__name__)


def _objective(*cv_scores):
    if isinstance(cv_scores[0], tuple):  # return_train_score
        return np.mean([_[0] for _ in cv_scores])
    else:
        return np.mean(cv_scores)


class AsyncSearchCV(DaskBaseSearchCV):
    def __init__(self, estimator,
                 scoring=None, iid=True, refit=True, cv=None,
                 error_score='raise', threshold=0.9,
                 return_train_score=True, scheduler=None, n_jobs=-1,
                 cache_cv=True):
        super(AsyncSearchCV, self).__init__(
            estimator=estimator,
            scoring=scoring, iid=iid,
            refit=refit, cv=cv,
            error_score=error_score,
            return_train_score=return_train_score,
            scheduler=scheduler,
            n_jobs=n_jobs,
            cache_cv=cache_cv,
        )
        self._threshold = threshold

    def _criterion(self, scores):
        return max(scores) > self._threshold

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

        def _update_graph_with_objective_scores(params):
            scores = _update_graph(params)
            scores = list(tz.groupby(1, scores).values())
            objective_scores = [
                'objective-score-{}'.format(tokenize(*v)) for v in scores]
            for name, v in zip(objective_scores, scores):
                dsk[name] = (_objective,) + tuple(v)
            return list(tz.concat(scores)), objective_scores

        # fill cluster with jobs
        params_iter = iter(self._get_param_iterator())
        candidate_params = [next(params_iter) for _ in range(int(ncores * 2))]

        next_token.counts = defaultdict(int)
        cv_scores, obj_scores = _update_graph_with_objective_scores(candidate_params)
        fs = [self._client.compute(Delayed(k, dsk)) for k in obj_scores]
        self._job_map = {k: f for k, f in zip(fs, candidate_params)}

        af = as_completed(fs)
        all_params, all_scores = candidate_params, cv_scores
        obj_scores = []
        best_score = -np.inf

        # adding jobs as completed
        for future in af:
            params, obj_score = self._job_map[future], future.result()
            logger.debug(
                "Current score {} for parameters: {}".format(obj_score, params))
            if obj_score > best_score:
                logger.info(
                    "Best score {} for parameters: {}".format(obj_score, params))
                best_score = obj_score

            obj_scores.append(obj_score)

            # select new parameters if criterion not met, else break
            if self._criterion(scores=obj_scores):
                # we cancel the remaining jobs
                logger.info("Criterion met, cleaning up jobs")
                nleft = len(af.futures)
                for f in af.futures:
                    del self._job_map[f]
                self._client.cancel(af.futures)
                all_params = all_params[:-nleft]
                all_scores = all_scores[:-n_splits * nleft]
                break
            p = next(params_iter)

            cv_scores, obj_score_names = _update_graph_with_objective_scores([p])
            f = self._client.compute(Delayed(obj_score_names[0], dsk))
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


class AsyncRandomizedSearchCV(AsyncSearchCV):
    def __init__(self, estimator, param_distributions, n_iter=10, random_state=None,
                 client=None, threshold=0.9,
                 scoring=None, iid=True, refit=True, cv=None, error_score='raise',
                 return_train_score=True, scheduler=None, n_jobs=-1, cache_cv=True):
        """
        criterion : a callable taking a list of scores and returning bool
        """
        super(AsyncRandomizedSearchCV, self).__init__(
            threshold=threshold,
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
        if client is None:
            client = Client()
        self._client = client
        self._job_map = {}

    def _get_param_iterator(self):
        """Return ParameterSampler instance for the given distributions"""
        return model_selection.ParameterSampler(self.param_distributions,
                                                self.n_iter,
                                                random_state=self.random_state)
