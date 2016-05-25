from __future__ import division, print_function, absolute_import

from functools import partial

import numpy as np
from dask import threaded
from dask.base import Base
from dask.delayed import delayed
from sklearn.base import clone, is_classifier

from .core import LazyDaskEstimator, check_X_y, as_lists_of_delayed
from .utils import copy_to


__all__ = ('DelayedEstimator', 'Averaged', 'Chained')


def _finalize_estimator(res, base):
    base = clone(base)
    if hasattr(base, '_update_estimator'):
        return base._update_estimator(res[0])
    return copy_to(res[0], base)


class DelayedEstimator(Base):
    """A delayed estimator"""
    __slots__ = ('key', 'dask', '_base')
    _default_get = staticmethod(threaded.get)
    _optimize = staticmethod(lambda dsk, keys, **kwargs: dsk)

    @property
    def _finalize(self):
        return partial(_finalize_estimator, base=self._base)

    def __init__(self, name, dasks, base):
        object.__setattr__(self, 'key', name)
        object.__setattr__(self, 'dask', dasks)
        object.__setattr__(self, '_base', base)

    def _keys(self):
        return [self.key]

    def __repr__(self):
        return 'DelayedEstimator<{0}>'.format(self.key)

    def __hash__(self):
        return hash(self.key)

    def __dir__(self):
        o = set(dir(type(self)))
        o.update(self._base._get_param_names())
        return list(o)

    def __setattr__(self, attr, val):
        raise TypeError("DelayedEstimator objects are immutable")

    def __getattr__(self, attr):
        if not attr.startswith('_') and attr in self._base._get_param_names():
            return getattr(self._base, attr)
        raise AttributeError("Attribute '{0}' not found".format(attr))


class DaskWrapper(LazyDaskEstimator):
    """Base class for classes that wrap a single estimator"""
    def __init__(self, estimator):
        self.estimator = estimator

    def _to_delayed(self, x):
        return DelayedEstimator(x.key, x.dask, clone(self))

    def _update_estimator(self, x):
        copy_to(x, self.estimator)
        return self

    def fit(self, X, y, **kwargs):
        compute = kwargs.pop("compute", True)
        res = self._fit(X, y, **kwargs)
        if compute:
            return self._update_estimator(res.compute())
        return self._to_delayed(res)


@delayed(pure=True)
def fit_chunk(est, X, y, **kwargs):
    return clone(est).fit(X, y, **kwargs)


@delayed(pure=True)
def average_coefs(estimators):
    if len(estimators) == 1:
        return estimators[0]
    o = clone(estimators[0])
    is_clf = is_classifier(o)
    if is_clf:
        classes = np.unique(np.concatenate([m.classes_ for m in estimators]))
        o.classes_ = classes
        n_classes = len(classes)
    if not is_clf or all(m.classes_.size == n_classes for m in estimators):
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
        return average_coefs(chunks)


@delayed(pure=True)
def _unique_chunk(x):
    return np.unique(x)


@delayed(pure=True)
def _unique_merge(x):
    return np.unique(np.concatenate(x))


@delayed(pure=True)
def partial_fit_chunk(est, x, y, **kwargs):
    return est.partial_fit(x, y, **kwargs)


class Chained(DaskWrapper):
    """Wrap an estimator by chaining calls to `partial_fit`"""
    def _fit(self, X, y, **kwargs):
        if not hasattr(self.estimator, 'partial_fit'):
            raise ValueError("estimator must support `partial_fit`")
        X, y = check_X_y(X, y)
        est = clone(self.estimator)
        x_parts, y_parts = as_lists_of_delayed(X, y)
        if is_classifier(est):
            classes = kwargs.pop('classes', None)
            if classes is None:
                classes = _unique_merge([_unique_chunk(i) for i in y_parts])
            est = partial_fit_chunk(est, x_parts[0], y_parts[0],
                                    classes=classes, **kwargs)
            x_parts = x_parts[1:]
            y_parts = y_parts[1:]
        for x, y in zip(x_parts, y_parts):
            est = partial_fit_chunk(est, x, y, **kwargs)
        return est
