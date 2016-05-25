from __future__ import division, print_function, absolute_import

import dask.array as da
import dask.bag as db
from dask.delayed import Delayed
from dask.base import normalize_token
from sklearn.base import BaseEstimator


class DaskBaseEstimator(BaseEstimator):
    """Base class for estimators provided by dklearn"""
    pass


class LazyDaskEstimator(DaskBaseEstimator):
    """Dask estimator that can compute lazily.

    Builds up single graph that can be computed lazily."""
    pass


class ImmediateDaskEstimator(DaskBaseEstimator):
    """Dask estimator that computes immediately."""
    pass


def normalize_sklearn_estimator(est):
    params = est.get_params(deep=False)
    for k, v in params.items():
        if isinstance(v, BaseEstimator):
            params[k] = normalize_sklearn_estimator(v)
    return type(est), sorted(est.get_params().items())


normalize_token.register(BaseEstimator, normalize_sklearn_estimator)


def check_X_y(X, y):
    """Check shape and partition alignment for dask types X and y."""
    x_is_array = isinstance(X, da.Array)
    y_is_array = isinstance(y, da.Array)

    if x_is_array and X.ndim != 2:
        raise ValueError("X must be 2 dimensional")
    if y_is_array and y.ndim not in (1, 2):
        raise ValueError("y must be 1 or 2 dimensional")
    if x_is_array and y_is_array:
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must share first dimension")
        elif X.chunks[0] != y.chunks[0]:
            raise ValueError("X and y chunks must be aligned")
    return X, y


def is_list_of(x, typ):
    """Is `x` a list of type `typ`"""
    return isinstance(x, (list, tuple)) and all(isinstance(i, typ) for i in x)


def is_dask_input(x):
    """Is `x` a dask input.

    Valid input types are ``da.Array``, ``db.Bag``, or a list of
    ``dask.delayed.Delayed``s"""
    return isinstance(x, (da.Array, db.Bag)) or is_list_of(x, Delayed)


def _as_list_of_delayed(x):
    if isinstance(x, da.Array):
        if x.ndim == 2 and len(x.chunks[1]) != 1:
            x = x.rechunk((x.chunks[0], x.shape[1]))
        return x.to_delayed().flatten().tolist()
    elif is_list_of(x, Delayed):
        return x
    elif isinstance(x, db.Bag):
        return x.to_delayed()
    else:
        raise TypeError("Invalid input type: {0}".format(type(x)))


def as_lists_of_delayed(*args):
    """Convert each argument to a list of delayed objects, as appropriate."""
    parts = tuple(_as_list_of_delayed(x) for x in args)
    if len(set(map(len, parts))) != 1:
        raise ValueError("inputs must all have the same number "
                         "of partitions along the first dimension")
    return parts
