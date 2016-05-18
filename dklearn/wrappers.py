from __future__ import division, print_function, absolute_import
import numpy as np
from dask.delayed import delayed, Delayed
from sklearn.base import clone, is_classifier

from .core import (LazyDaskEstimator, check_X_y, as_lists_of_delayed,
                   is_dask_input)


__all__ = ('DaskWrapper', 'Averaged', 'Chained')


@delayed(pure=True)
def fit_chunk(est, X, y, **kwargs):
    return clone(est).fit(X, y, **kwargs)


@delayed(pure=True)
def partial_fit_chunk(est, x, y, **kwargs):
    return est.partial_fit(x, y, **kwargs)


@delayed(pure=True)
def transform_chunk(est, X):
    return est.transform(X)


@delayed(pure=True)
def fit_and_transform_chunk(est, X, y=None, **fit_params):
    """Call fit_transform, return fit and Xt"""
    est = clone(est)
    Xt = est.fit_transform(X, y, **fit_params)
    return est, Xt


@delayed(pure=True)
def score_chunk(est, X, y, **kwargs):
    return est.score(X, y, **kwargs)


@delayed(pure=True)
def methodcaller(self, method, *args, **kwargs):
    return getattr(self, method)(*args, **kwargs)


class DaskWrapper(LazyDaskEstimator):
    """Base class for classes that wrap a single estimator"""
    def __init__(self, estimator):
        self.estimator = estimator
        self._delayed = None

    def _from_delayed(self, delayed):
        o = clone(self)
        o._delayed = delayed
        return o

    def to_delayed(self):
        return delayed(type(self), pure=True)(self._delayed)

    @property
    def estimator_(self):
        """Most recently fit estimator"""
        if self._delayed is not None:
            return self._delayed
        elif hasattr(self.estimator, 'coef_'):
            return self.estimator
        else:
            raise ValueError("Not Fit")

    def fit(self, X, y, **kwargs):
        compute = kwargs.pop("compute", True)
        if not is_dask_input(X) and is_dask_input(y):
            res = fit_chunk(self.estimator, X, y, **kwargs)
        else:
            res = self._fit(X, y, **kwargs)
        if not compute:
            return self._from_delayed(res)
        self._delayed = res
        return self.compute()

    def compute(self, **kwargs):
        if self._delayed is None:
            return self
        self.estimator = self._delayed.compute(**kwargs)
        self._delayed = None
        return self

    def _elemwise(self, method, *args, **kwargs):
        estimator = self.estimator_
        if any(is_dask_input(x) for x in args):
            chunks = as_lists_of_delayed(*args)
            return [methodcaller(estimator, method, *c, **kwargs)
                    for c in zip(*chunks)]
        elif any(isinstance(x, Delayed) for x in args):
            return methodcaller(estimator, method, *c, **kwargs)
        else:
            return getattr(estimator, method)(*args, **kwargs)

    def predict(self, X):
        return self._elemwise('predict', X)

    def predict_proba(self, X):
        return self._elemwise('predict_proba', X)

    def predict_log_proba(self, X):
        return self._elemwise('predict_log_proba', X)

    def decision_function(self, X):
        return self._elemwise('decision_function', X)

    def score(self, X, y, **kwargs):
        if is_dask_input(X) or is_dask_input(y):
            raise ValueError("score not implemented for dask inputs")
        estimator = self.estimator_
        if isinstance(X, Delayed) or isinstance(y, Delayed):
            return methodcaller(estimator, 'score', X, y, **kwargs)
        return getattr(estimator, 'score')(X, y, **kwargs)


@delayed(pure=True)
def merge_classifiers(estimators):
    if len(estimators) == 1:
        return estimators[0]
    o = clone(estimators[0])
    classes = np.unique(np.concatenate([m.classes_ for m in estimators]))
    o.classes_ = classes
    n_classes = len(classes)
    if all(m.classes_.size == n_classes for m in estimators):
        o.coef_ = np.mean([m.coef_ for m in estimators], axis=0)
        o.intercept_ = np.mean([m.intercept_ for m in estimators], axis=0)
    else:
        # Not all estimators got all classes. Multiclass problems are fit using
        # ovr, which results in a row per class. Here we average the
        # coefficients for each class, using zero if that class wasn't fit.
        n_features = estimators[0].coef_.shape[1]
        coef = np.zeros((n_classes, n_features), dtype='f8')
        intercept = np.zeros(n_classes, dtype='f8')
        for m in estimators:
            ind = np.in1d(classes, m.classes_)
            coef[ind] += m.coef_
            intercept[ind] += m.intercept_
        o.coef_ = coef / len(estimators)
        o.intercept_ = intercept / len(estimators)
    return o


@delayed(pure=True)
def merge_regressors(estimators):
    if len(estimators) == 1:
        return estimators[0]
    o = clone(estimators[0])
    o.coef_ = np.mean([m.coef_ for m in estimators], axis=0)
    o.intercept_ = np.mean([m.intercept_ for m in estimators], axis=0)
    return o


class Averaged(DaskWrapper):
    """Wrap an estimator by averaging coefficients"""
    def _fit(self, X, y, **kwargs):
        if not hasattr(self.estimator, 'fit'):
            raise ValueError("estimator must support `fit`")
        X, y = check_X_y(X, y)
        x_parts, y_parts = as_lists_of_delayed(X, y)
        est = self.estimator
        chunks = [fit_chunk(est, xp, yp, **kwargs) for (xp, yp) in
                  zip(x_parts, y_parts)]
        if is_classifier(self.estimator):
            res = merge_classifiers(chunks)
        else:
            res = merge_regressors(chunks)
        return res


@delayed(pure=True)
def _unique_chunk(x):
    return np.unique(x)


@delayed(pure=True)
def _unique_merge(x):
    return np.unique(np.concatenate(x))


class Chained(DaskWrapper):
    """Wrap an estimator by chaining calls to `partial_fit`"""
    def _fit(self, X, y, **kwargs):
        if not hasattr(self.estimator, 'partial_fit'):
            raise ValueError("estimator must support `partial_fit`")
        X, y = check_X_y(X, y)
        est = clone(self.estimator)
        x_parts, y_parts = as_lists_of_delayed(X, y)
        if is_classifier(est):
            classes = kwargs.pop('classes', False)
            if not classes:
                classes = _unique_merge([_unique_chunk(i) for i in y_parts])
            kwargs_2 = {'classes': classes}
            kwargs_2.update(kwargs)
            est = partial_fit_chunk(est, x_parts[0], y_parts[0], **kwargs_2)
            x_parts = x_parts[1:]
            y_parts = y_parts[1:]
        for x, y in zip(x_parts, y_parts):
            est = partial_fit_chunk(est, x, y, **kwargs)
        return est
