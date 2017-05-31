import abc
import numbers
from collections import defaultdict

import numpy as np
import time
import toolz as tz
import operator as op
from dask.delayed import tokenize, Delayed
from distributed import Client, as_completed, wait
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


class CriterionReached(BaseException):
    """Stopping criterion for search has been reached"""


# todo: find a way to use the threaded/... verions of scheduler with compute
class AsyncSearchCV(DaskBaseSearchCV):
    def __init__(self, estimator, scoring=None,
                 iid=True, refit=True, cv=None,
                 error_score='raise',
                 return_train_score=True, scheduler=None, n_jobs=-1,
                 cache_cv=True, client=None):
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
        if client is None:
            client = Client()
        self._client = client
        self._job_map = {}

    @abc.abstractmethod
    def _parameter_sampler(self, params, scores, timestamps):
        """Sample new parameters according to previous results with timestamps

        Args:
            params: list of parameter dicts
            scores: list of float scores
            timestamps: list of float timestamps

        Returns:
            new parameter dict for self._estimator

        """
        pass

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
            scores = [v for i, v in
                      sorted(tz.groupby(1, scores).items(), key=op.itemgetter(0))]
            objective_scores = [
                'objective-score-{}'.format(tokenize(*v)) for v in scores]
            for name, v in zip(objective_scores, scores):
                dsk[name] = (_objective,) + tuple(v)
            return scores, objective_scores

        # fill cluster with jobs
        candidate_params = [self._parameter_sampler(None, None, None) for _ in
                            range(int(ncores * 2))]

        next_token.counts = defaultdict(int)
        cv_scores, obj_scores = _update_graph_with_objective_scores(candidate_params)

        fs = [self._client.compute(Delayed(k, dsk)) for k in obj_scores]
        self._job_map = {k: f for k, f in zip(fs, candidate_params)}

        af = as_completed(fs)

        score_map = {k: s for k, s in zip(fs, cv_scores)}
        obj_scores, timestamps = {}, {}
        completed = []  # to keep ordering

        # adding jobs as completed
        for future in af:
            params, obj_score = self._job_map[future], future.result()
            completed.append(future)
            timestamps[future] = time.time()
            obj_scores[future] = obj_score

            try:
                p = self._parameter_sampler(
                    [self._job_map[f] for f in completed],
                    [obj_scores[f] for f in completed],
                    [timestamps[f] for f in completed]
                )
            except CriterionReached:
                for f in af.futures:
                    del self._job_map[f]
                    del score_map[f]
                self._client.cancel(af.futures)
                break

            cv_scores, obj_score_names = _update_graph_with_objective_scores([p])
            f = self._client.compute(Delayed(obj_score_names[0], dsk))
            score_map[f] = cv_scores[0]
            self._job_map[f] = params
            af.add(f)

        # finalize results
        main_token = next_token.token
        keys = generate_results(dsk, estimator, list(tz.concat([score_map[f] for f in completed])),
                                main_token, X_name, y_name,
                                [self._job_map[f] for f in completed], n_splits,
                                self.error_score, weights, self.refit, fit_params)

        self.dask_graph_ = dsk
        self.n_splits_ = n_splits
        n_jobs = _normalize_n_jobs(self.n_jobs)
        scheduler = self._client.get
        # n_jobs is a bit excessive if we've already gotten the results
        out = scheduler(dsk, keys, num_workers=n_jobs)
        self.cv_results_ = results = out[0]
        self.best_index_ = np.flatnonzero(results["rank_test_score"] == 1)[0]
        if self.refit:
            self.best_estimator_ = out[1]
        return self


class AsyncRandomizedSearchCV(AsyncSearchCV):
    def __init__(self, estimator, param_distributions, n_iter=10, random_state=None,
                 threshold=0.9, scoring=None, iid=True, refit=True, cv=None,
                 error_score='raise', return_train_score=True, scheduler=None,
                 n_jobs=-1, cache_cv=True, client=None
                 ):
        super(AsyncRandomizedSearchCV, self).__init__(
            estimator=estimator,
            scoring=scoring, iid=iid,
            refit=refit, cv=cv,
            error_score=error_score,
            return_train_score=return_train_score,
            scheduler=scheduler,
            n_jobs=n_jobs,
            cache_cv=cache_cv,
            client=client
        )
        self.param_distributions = param_distributions
        self.n_iter = n_iter  # maximum number of iterations
        self.random_state = random_state
        self._threshold = threshold
        self._param_iter = iter(
            model_selection.ParameterSampler(self.param_distributions,
                                                self.n_iter,
                                                random_state=self.random_state)
        )

    # todo: decide whether to use a generator method here instead of exception
    def _parameter_sampler(self, params, scores, timestamps):
        if params == scores == timestamps == None:  # fixme
            return next(self._param_iter)

        best_score, score = max(scores), scores[-1]

        logger.debug(
            "Current score {} for parameters: {}".format(score, params[-1]))

        if score > best_score:
            logger.info(
                "Best score {} for parameters: {}".format(score, params[-1]))

        if (best_score >= self._threshold) or (len(scores) > self.n_iter):
            raise CriterionReached()
        else:
            return next(self._param_iter)
