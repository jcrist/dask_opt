from __future__ import division, print_function, absolute_import

import dask.array as da
import dask.bag as db
from dask.delayed import Delayed


def is_list_of(x, typ):
    """Is `x` a list of type `typ`"""
    return isinstance(x, (list, tuple)) and all(isinstance(i, typ) for i in x)


def xy_to_parts(X, y):
    """Check alignment of `X` and `y`, and return lists of parts for each"""
    # Extract parts from X
    if isinstance(X, da.Array):
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")
        if len(X.chunks[1]) != 1:
            X = X.rechunk((X.chunks[0], X.shape[1]))
        x_parts = X.to_delayed().flatten().tolist()
    elif is_list_of(X, Delayed):
        x_parts = X
    elif isinstance(X, db.Bag):
        x_parts = X.to_delayed()
    else:
        raise TypeError("invalid `X` type: {0}".format(type(X)))
    # Extract parts from y
    if isinstance(y, da.Array):
        if y.ndim not in (1, 2):
            raise ValueError("y must be 1 or 2 dimensional")
        if y.ndim == 2 and len(y.nchunks[1]) != 1:
            y = y.rechunk((y.chunks[0], y.shape[1]))
        y_parts = y.to_delayed().flatten().tolist()
    elif is_list_of(y, Delayed):
        y_parts = y
    elif isinstance(y, db.Bag):
        y_parts = y.to_delayed()
    else:
        raise TypeError("invalid `y` type: {0}".format(type(y)))
    # Alignment checks
    if isinstance(X, da.Array) and isinstance(y, da.Array):
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must share first dimension")
        elif X.chunks[0] != y.chunks[0]:
            raise ValueError("X and y chunks must be aligned")
    else:
        if not len(x_parts) == len(y_parts):
            raise ValueError("X and y must have the same number "
                             "of partitions along the first dimension")
    return x_parts, y_parts


def copy_doc(func):
    """Copy docstring from func"""
    def _(x):
        x.__doc__ = func.__doc__
        return x
    return _
