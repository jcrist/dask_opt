import abc
import logging
import numbers
import operator as op
import time
from collections import defaultdict

import numpy as np
import toolz as tz
from dask.delayed import tokenize, Delayed
from distributed import Client, as_completed
from distributed.client import Future
from distributed.diagnostics.plugin import SchedulerPlugin
from heapdict import heapdict
from sklearn import model_selection
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection._search import _check_param_grid

from dask_searchcv.model_selection import DaskBaseSearchCV, build_graph, \
    update_graph, generate_results, _normalize_n_jobs

log = logging.getLogger(__name__)


def _objective(*cv_scores):
    if isinstance(cv_scores[0], tuple):  # return_train_score
        return np.mean([_[0] for _ in cv_scores])
    else:
        return np.mean(cv_scores)

# todo: sort out docstrings in sub-classes of DaskBaseSearchCV


class CriterionReached(BaseException):
    """Stopping criterion for search has been reached"""


class AsyncSearchCV(DaskBaseSearchCV):
    def __init__(self, estimator, scoring=None, iid=True, refit=True, cv=None,
                 error_score='raise', return_train_score=True,
                 n_jobs=-1, cache_cv=True, client=None, occupancy_factor=2):
        """Asynchronous version of DaskBaseSearchCV using distributed.Client"""
        if client is None:
            client = Client()
        scheduler = client.get
        super(AsyncSearchCV, self).__init__(estimator=estimator, scoring=scoring,
                                            iid=iid, refit=refit, cv=cv,
                                            error_score=error_score,
                                            return_train_score=return_train_score,
                                            scheduler=scheduler, n_jobs=n_jobs,
                                            cache_cv=cache_cv,
                                            )
        self._client = client
        self._job_map = {}
        self._occupancy_factor = occupancy_factor

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

    def _build_graph(self, X, y, groups, fit_params):
        # build graph
        (dsk, cv_name, X_name, y_name, n_splits, fit_params, weights,
         next_param_token, next_token) = build_graph(
            self.estimator, self.cv, X, y, groups, fit_params, iid=self.iid,
            error_score=self.error_score, return_train_score=self.return_train_score,
            cache_cv=self.cache_cv
        )
        self.dask_graph_ = dsk
        self._cv_name = cv_name
        self._X_name = X_name
        self._y_name = y_name
        self.n_splits_ = n_splits
        self._fit_params = fit_params
        self._next_token = next_token
        self._next_param_token = next_param_token
        self._weights = weights

    def _update_graph(self, params):
        # we reset the default iterator between updates to avoid duplicate keys
        self._next_token.counts = defaultdict(int)
        cv_scores = update_graph(self.dask_graph_, self._next_param_token,
                                 self._next_token, self.estimator, self._cv_name,
                                 self._X_name, self._y_name, self._fit_params,
                                 self.n_splits_, self.error_score, self.scorer_,
                                 self.return_train_score, params)

        cv_scores = [v for i, v in
                     sorted(tz.groupby(1, cv_scores).items(), key=op.itemgetter(0))]
        # maybe don't tokenize here:
        objective_scores = [
            'objective-score-{}'.format(tokenize(*v)) for v in cv_scores]

        for name, v in zip(objective_scores, cv_scores):
            self.dask_graph_[name] = (_objective,) + tuple(v)

        return cv_scores, objective_scores

    def fit_async(self, X, y=None, groups=None, **fit_params):
        if not (isinstance(self.error_score, numbers.Number) or
                        self.error_score == 'raise'):
            raise ValueError("error_score must be the string 'raise' or a"
                             " numeric value.")
        estimator = self.estimator
        self.scorer_ = check_scoring(estimator, scoring=self.scoring)
        ncores = len(self._client.ncores())

        # fill cluster with jobs
        candidate_params = []
        for _ in range(int(ncores * self._occupancy_factor)):
            try:
                candidate_params.append(self._parameter_sampler(None, None, None))
            except StopIteration:
                break

        self._build_graph(X, y, groups, fit_params)
        cv_scores, obj_scores = self._update_graph(candidate_params)

        # we do this just to keep the cv_scores around:
        # cv_scores_list = [
        #     self._client.persist(Delayed(k, self.dask_graph_)) for k in cv_scores
        # ]
        obj_scores_list = [
            self._client.persist(Delayed(k, self.dask_graph_)) for k in obj_scores
        ]

        fs = [Future(x.key, self._client) for x in obj_scores_list]
        self._job_map = {f: p for f, p in zip(fs, candidate_params)}

        af = as_completed(fs)

        score_map = {f: s for f, s in zip(fs, cv_scores)}
        completed, obj_scores, timestamps = [], {}, {}

        # adding jobs as completed
        while True:
            try:
                futures = af.next_batch()
            except StopIteration:
                break

            parameters_to_update = []
            for future in futures:
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
                    parameters_to_update.append(p)
                except CriterionReached:
                    for f in af.futures:
                        del self._job_map[f]
                        del score_map[f]
                    self._client.cancel(af.futures)
                    break
                except StopIteration:
                    continue

            if parameters_to_update:
                cv_score_names, obj_score_names = self._update_graph(parameters_to_update)

                # cv_sc, obj_sc = cv_score_names[0], obj_score_names[0]
                # cv_score_futures.append(
                #     self._client.compute(Delayed(cv_scores[0], self.dask_graph_)))
                # f = self._client.compute(
                # Delayed(obj_score_names[0], self.dask_graph_))

                # cv_scores_list.append(
                #     self._client.persist(Delayed(cv_score_names, self.dask_graph_))
                # )
                obj_scores_list.append(
                    self._client.persist(Delayed(obj_score_names, self.dask_graph_))
                )
                for obj_sc, cv_sc, p in zip(obj_score_names, cv_score_names, parameters_to_update):
                    f = Future(obj_sc, self._client)
                    score_map[f] = cv_sc
                    self._job_map[f] = p
                    af.add(f)

        # finalize results
        main_token = self._next_token.token
        keys = generate_results(self.dask_graph_, estimator,
                                list(tz.concat([score_map[f] for f in completed])),
                                main_token, self._X_name, self._y_name,
                                [self._job_map[f] for f in completed],
                                self.n_splits_, self.error_score, self._weights,
                                self.refit, self._fit_params)

        n_jobs = _normalize_n_jobs(self.n_jobs)
        scheduler = self._client.get

        # fixme: n_jobs is a bit excessive if we've already gotten the results?
        out = scheduler(self.dask_graph_, keys, num_workers=n_jobs)

        self.cv_results_ = results = out[0]
        self.best_index_ = np.flatnonzero(results["rank_test_score"] == 1)[0]
        if self.refit:
            self.best_estimator_ = out[1]

        return self


class AsyncRandomizedSearchCV(AsyncSearchCV):
    def __init__(self, estimator, param_distributions, n_iter=10, random_state=None,
                 threshold=0.9, scoring=None, iid=True, refit=True, cv=None,
                 error_score='raise', return_train_score=True,
                 n_jobs=-1, cache_cv=True, client=None, occupancy_factor=2
                 ):
        super(AsyncRandomizedSearchCV, self).__init__(
            estimator=estimator,
            scoring=scoring, iid=iid,
            refit=refit, cv=cv,
            error_score=error_score,
            return_train_score=return_train_score,
            n_jobs=n_jobs,
            cache_cv=cache_cv,
            client=client,
            occupancy_factor=occupancy_factor
        )
        self.param_distributions = param_distributions
        self.n_iter = n_iter  # maximum number of iterations
        self.random_state = random_state
        self._threshold = threshold
        self._param_iter = iter(
            model_selection.ParameterSampler(self.param_distributions, self.n_iter,
                                             random_state=self.random_state)
        )

    def _parameter_sampler(self, params, scores, timestamps):
        if params == scores == timestamps is None:
            return next(self._param_iter)

        best_score, score = max(scores), scores[-1]
        log.debug(
            "Current score {} for parameters: {}".format(score, params[-1]))
        if score > best_score:
            log.info(
                "Best score {} for parameters: {}".format(score, params[-1]))

        if (best_score >= self._threshold) or (len(scores) > self.n_iter):
            raise CriterionReached()
        else:
            return next(self._param_iter)


class AsyncGridSearchCV(AsyncSearchCV):
    def __init__(self, estimator, param_grid,
                 threshold=0.9, scoring=None, iid=True, refit=True, cv=None,
                 error_score='raise', return_train_score=True,
                 n_jobs=-1, cache_cv=True, client=None, occupancy_factor=2
                 ):
        super(AsyncGridSearchCV, self).__init__(estimator=estimator, scoring=scoring,
                                                iid=iid, refit=refit, cv=cv,
                                                error_score=error_score,
                                                return_train_score=return_train_score,
                                                n_jobs=n_jobs, cache_cv=cache_cv,
                                                client=client,
                                                occupancy_factor=occupancy_factor
                                                )
        self.param_grid = param_grid
        self._threshold = threshold
        _check_param_grid(param_grid)
        self._param_iter = iter(
            model_selection.ParameterGrid(self.param_grid)
        )

    def _parameter_sampler(self, params, scores, timestamps):
        if params == scores == timestamps is None:
            return next(self._param_iter)

        best_score, score = max(scores), scores[-1]
        log.debug(
            "Current score {} for parameters: {}".format(score, params[-1]))
        if score > best_score:
            log.info(
                "Best score {} for parameters: {}".format(score, params[-1]))

        if best_score >= self._threshold:
            raise CriterionReached()
        else:
            return next(self._param_iter)


class CachingPlugin(SchedulerPlugin):
    """Simple caching of results on the scheduler for asynchronous search speedup

    Run with:
        client.run_on_scheduler(
            lambda dask_scheduler: dask_scheduler.add_plugin(
                CachingPlugin(dask_scheduler)))

    """
    def __init__(self, scheduler, cache_size=1e9, limit=0.):
        self.scheduler = scheduler
        self.startstops = []
        self.limit = limit
        self.total_bytes = 0
        self.cache_size = cache_size
        self._key_heap = heapdict()
        self._key_sizes = dict()

    def cleanup(self):
        ks = []
        while self.total_bytes > self.cache_size:
            ks.append(self._shrink_one())
        return ks

    def _shrink_one(self):
        key, cost = self._key_heap.popitem()
        self.total_bytes -= self._key_sizes[key]
        del self._key_sizes[key]
        return key

    # def scoring(self, nbytes, compute_time):
    #     return compute_time / nbytes / 1e9

    def should_keep(self, nbytes, startstops, key):
        if (startstops is not None):
            compute_time = {k: v2 - v1 for k, v1, v2 in startstops}.get('compute')
            if compute_time is not None:
                cost = compute_time / nbytes / 1e9
                if cost > self.limit:
                    self.total_bytes += nbytes
                    self._key_heap[key] = cost
                    self._key_sizes[key] = nbytes
                    return True
        return False

    def transition(
        self, key, start, finish, nbytes=None, startstops=None, *args, **kwargs
    ):
        if start == 'processing' and finish == 'memory' and self.should_keep(
            nbytes, startstops, key
        ):
            log.debug('storing key: {}'.format(key))

            self.scheduler.client_desires_keys(
                keys=[key], client='fake-caching-client'
            )

        no_longer_desired_keys = self.cleanup()

        self.scheduler.client_releases_keys(
            keys=no_longer_desired_keys, client='fake-caching-client'
        )
        log.debug('keys {}'.format(self.scheduler._keys))
