from __future__ import absolute_import, print_function, division

from copy import deepcopy
from operator import getitem

from sklearn.base import clone, BaseEstimator
from dask.base import tokenize, Base, normalize_token
from dask.delayed import Delayed
from dask.optimize import fuse
from dask.threaded import get as threaded_get
from toolz import partial

from .core import unpack_arguments, from_sklearn


def _fit(est, X, y, kwargs):
    return clone(est).fit(X, y, **kwargs)


def _predict(est, X):
    return est.predict(X)


def _score(est, X, y, kwargs):
    return est.score(X, y, **kwargs)


def _transform(est, X):
    return est.transform(X)


def _fit_transform(est, X, y, kwargs):
    est = clone(est)
    if hasattr(est, 'fit_transform'):
        fit = est
        tr = est.fit_transform(X, y, **kwargs)
    else:
        fit = est.fit(X, y, **kwargs)
        tr = est.transform(X)
    return fit, tr


class ClassProxy(object):
    def __init__(self, cls):
        self.cls = cls

    @property
    def __name__(self):
        return 'Dask' + self.cls.__name__

    def __call__(self, *args, **kwargs):
        return Estimator(self.cls(*args, **kwargs))


class Estimator(Base, BaseEstimator):
    _default_get = staticmethod(threaded_get)
    _finalize = staticmethod(lambda res: Estimator(res[0]))
    _optimize = staticmethod(lambda d, k, **kws: fuse(d, k)[0])

    def __init__(self, est, copy=True):
        if not isinstance(est, BaseEstimator):
            raise TypeError("Expected instance of `BaseEstimator`, "
                            "got {0}".format(type(est).__name__))
        if copy:
            est = deepcopy(est)
        self._base = est
        self._reset()

    def _reset(self, fastpath=False):
        """Reset the base graph for lazy computations."""
        self._name = name = 'from_sklearn-' + tokenize(self._base)
        self.dask = {name: self._base}

    def _keys(self):
        return [self._name]

    @property
    def __class__(self):
        return ClassProxy(type(self._base))

    @property
    def _estimator_type(self):
        return self._base._estimator_type

    def get_params(self, deep=True):
        return self._base.get_params(deep=True)

    def set_params(self, **params):
        self._base.set_params(**params)
        self._reset()
        return self

    def __getattr__(self, attr):
        if hasattr(self._base, attr):
            return getattr(self._base, attr)
        else:
            raise AttributeError("Attribute {0} either missing, or not "
                                 "computed yet. Try calling `.compute()`, "
                                 "and check again.".format(attr))

    def __setattr__(self, k, v):
        if k in ('_name', 'dask', '_base'):
            object.__setattr__(self, k, v)
        else:
            raise AttributeError("Attribute setting not permitted. "
                                 "Use `set_params` to change parameters")

    def __dir__(self):
        o = set(dir(type(self)))
        o.update(self.__dict__)
        o.update(self._base._get_param_names())
        o.update(i for i in dir(self._base) if i.endswith('_'))
        return list(o)

    @classmethod
    def from_sklearn(cls, est):
        return cls(est)

    def to_sklearn(self, compute=True):
        if compute:
            if len(self.dask) > 1:
                return Delayed(self._name, [self.dask]).compute()
            return self._base
        return Delayed(self._name, [self.dask])

    def fit(self, X, y, **kwargs):
        # Remove all fit (`foo_`) attributes, reset graph
        self._base = clone(self._base)
        self._reset()
        # Construct the fit step
        name = 'fit-' + tokenize(self, X, y, kwargs)
        X, y, dsk = unpack_arguments(X, y)
        dsk.update(self.dask)
        dsk[name] = (_fit, self._name, X, y, kwargs)
        self._name = name
        self.dask = dsk
        return self

    def _fit_transform(self, X, y, **kwargs):
        # Remove all fit (`foo_`) attributes, reset graph
        self._base = clone(self._base)
        self._reset()
        # Construct the fit and transform steps
        token = tokenize(self, X, y, kwargs)
        fit_tr_name = 'fit-transform-' + token
        fit_name = 'fit-' + token
        tr_name = 'tr-' + token
        X, y, dsk = unpack_arguments(X, y)
        dsk.update(self.dask)
        dsk[fit_tr_name] = (_fit_transform, self._name, X, y, kwargs)
        dsk2 = dsk.copy()
        dsk[fit_name] = (getitem, fit_tr_name, 0)
        dsk2[tr_name] = (getitem, fit_tr_name, 1)
        self._name = fit_name
        self.dask = dsk
        return self, Delayed(tr_name, [dsk2])

    def fit_transform(self, X, y, **kwargs):
        _, Xt = self._fit_transform(X, y, **kwargs)
        return Xt

    def predict(self, X):
        name = 'predict-' + tokenize(self, X)
        X, dsk = unpack_arguments(X)
        dsk[name] = (_predict, self._name, X)
        return Delayed(name, [dsk, self.dask])

    def score(self, X, y, **kwargs):
        name = 'score-' + tokenize(self, X, y, kwargs)
        X, y, dsk = unpack_arguments(X, y)
        dsk[name] = (_score, self._name, X, y, kwargs)
        return Delayed(name, [dsk, self.dask])

    def transform(self, X):
        name = 'transform-' + tokenize(self, X)
        X, dsk = unpack_arguments(X)
        dsk[name] = (_transform, self._name, X)
        return Delayed(name, [dsk, self.dask])


@partial(normalize_token.register, Estimator)
def normalize_dask_estimators(est):
    return type(est).__name__, est._name


from_sklearn.dispatch.register(BaseEstimator, Estimator.from_sklearn)
