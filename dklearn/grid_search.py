from __future__ import division, print_function, absolute_import

from dask.delayed import delayed
import numpy as np
from sklearn.base import clone, is_classifier
import sklearn.pipeline
from sklearn.metrics.scorer import check_scoring
from sklearn.grid_search import (_CVScoreTuple, _check_param_grid,
                                 ParameterGrid, ParameterSampler)
from sklearn.cross_validation import check_cv
from sklearn.utils import indexable, safe_indexing

from .core import DaskBaseEstimator, LazyDaskEstimator, is_dask_input
from .wrappers import DaskWrapper
from .pipeline import Pipeline


@delayed(pure=True)
def _extract_train_test(X, y, train, test):
    X_train = safe_indexing(X, train)
    y_train = safe_indexing(y, train)
    X_test = safe_indexing(X, test)
    y_test = safe_indexing(y, test)
    n_samples = len(X_test)
    return X_train, y_train, X_test, y_test, n_samples


@delayed(pure=True)
def get_grid_scores_and_best(score_len_params, n_folds, iid):
    n_fits = len(score_len_params)

    scores = list()
    grid_scores = list()
    for grid_start in range(0, n_fits, n_folds):
        n_test_samples = 0
        score = 0
        all_scores = []
        for this_score, this_n_test_samples, parameters in \
                score_len_params[grid_start:grid_start + n_folds]:
            all_scores.append(this_score)
            if iid:
                this_score *= this_n_test_samples
                n_test_samples += this_n_test_samples
            score += this_score
        if iid:
            score /= float(n_test_samples)
        else:
            score /= float(n_folds)
        scores.append((score, parameters))
        grid_scores.append(_CVScoreTuple(
            parameters,
            score,
            np.array(all_scores)))
    best = sorted(grid_scores, key=lambda x: x.mean_validation_score,
                  reverse=True)[0]
    return grid_scores, best


@delayed(pure=True)
def apply_scorer(scorer, estimator, X_test, y_test):
    if y_test is None:
        return scorer(estimator, X_test)
    return scorer(estimator, X_test, y_test)


def _fit_and_score(estimator, X, y, scorer, train, test,
                   parameters, fit_params):
    estimator = estimator.set_params(**parameters)

    # Lazily extract data
    data = _extract_train_test(X, y, train, test)
    X_train = data[0]
    y_train = data[1]
    X_test = data[2]
    y_test = data[3]
    n_samples = data[4]

    # Lazily train
    if y_train is None:
        estimator = estimator.fit(X_train, compute=False, **fit_params)
    else:
        estimator = estimator.fit(X_train, y_train, compute=False,
                                  **fit_params)

    score = apply_scorer(scorer, estimator.to_delayed(), X_test, y_test)

    return score, n_samples, parameters


class BaseSearchCV(DaskBaseEstimator):
    """Base class for hyper parameter search with cross-validation."""

    def __init__(self, estimator, scoring=None, fit_params=None, iid=True,
                 refit=True, cv=None, verbose=0, error_score='raise'):

        self.scoring = scoring
        self.estimator = estimator
        self.fit_params = fit_params if fit_params is not None else {}
        self.iid = iid
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.error_score = error_score

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    def _fit(self, X, y, parameter_iterable, compute=True):
        """Actual fitting,  performing the search over parameters."""

        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        if not (is_dask_input(X) or is_dask_input(y)):
            # Inputs are numpy arrays
            X, y = indexable(X, y)
            cv = check_cv(self.cv, X, y,
                          classifier=is_classifier(self.estimator))
            n_folds = len(cv)
            estimator = clone(self.estimator)
            if isinstance(estimator, sklearn.pipeline.Pipeline):
                estimator = Pipeline.from_sklearn(estimator)
            elif not isinstance(estimator, LazyDaskEstimator):
                estimator = DaskWrapper(estimator)

            tups = [_fit_and_score(estimator, X, y, self.scorer_, train, test,
                                   parameters, self.fit_params)
                    for parameters in parameter_iterable
                    for train, test in cv]
        else:
            raise NotImplementedError("CV with dask inputs")

        self._delayed = get_grid_scores_and_best(tups, n_folds, self.iid)
        self._refit_data = (X, y) if self.refit else None
        if compute:
            self.compute()
        return self

    def compute(self, **kwargs):
        if self._delayed is None:
            return self
        grid_scores, best = self._delayed.compute(**kwargs)
        self._delayed = None
        self.grid_scores_ = grid_scores
        self.best_params_ = best.parameters
        self.best_score_ = best.mean_validation_score
        if self._refit_data is not None:
            X, y = self._refit_data
            best_estimator = clone(self.estimator)
            best_estimator.set_params(**best.parameters)
            best_estimator.fit(X, y, **self.fit_params)
            self.best_estimator_ = best_estimator
        return self


class GridSearchCV(BaseSearchCV):
    def __init__(self, estimator, param_grid, scoring=None, fit_params=None,
                 iid=True, refit=True, cv=None, verbose=0,
                 error_score='raise'):

        super(GridSearchCV, self).__init__(
                estimator=estimator, scoring=scoring, fit_params=fit_params,
                iid=iid, refit=refit, cv=cv, verbose=verbose,
                error_score=error_score)
        self.param_grid = param_grid
        _check_param_grid(param_grid)

    def fit(self, X, y=None, compute=True):
        return self._fit(X, y, ParameterGrid(self.param_grid), compute=compute)


class RandomizedSearchCV(BaseSearchCV):
    def __init__(self, estimator, param_distributions, n_iter=10, scoring=None,
                 fit_params=None, iid=True, refit=True, cv=None, verbose=0,
                 random_state=None, error_score='raise'):

        super(RandomizedSearchCV, self).__init__(
            estimator=estimator, scoring=scoring, fit_params=fit_params,
            iid=iid, refit=refit, cv=cv, verbose=verbose,
            error_score=error_score)
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y=None, compute=True):
        sampled_params = ParameterSampler(self.param_distributions,
                                          self.n_iter,
                                          random_state=self.random_state)
        return self._fit(X, y, sampled_params, compute=compute)
