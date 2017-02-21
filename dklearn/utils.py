import copy

import numpy as np
import dask.array as da
from dask.base import Base, tokenize
from dask.delayed import delayed, Delayed

from sklearn.utils.validation import indexable, _num_samples


def _indexable(x):
    return indexable(x)[0]


def to_indexable(*args):
    """Ensure that all args are an indexable type.

    Conversion runs lazily for dask objects, immediately otherwise."""
    for x in args:
        if x is None or isinstance(x, da.Array):
            yield x
        elif isinstance(x, Base):
            yield delayed(_indexable, pure=True)(x)
        else:
            yield _indexable(x)


def to_keys(dsk, *args):
    for x in args:
        if x is None:
            yield None
        elif isinstance(x, da.Array):
            x = delayed(x)
            dsk.update(x.dask)
            yield x.key
        elif isinstance(x, Delayed):
            dsk.update(x.dask)
            yield x.key
        else:
            assert not isinstance(x, Base)
            key = 'array-' + tokenize(x)
            dsk[key] = x
            yield key


def unique(x):
    """The number of unique element in x"""
    if not isinstance(x, Base):
        return np.unique(x)

    if isinstance(x, da.Array):
        return da.unique(x).compute()

    return delayed(np.unique, pure=True)(x).compute()


def num_samples(x):
    """The number of samples in x"""
    if not isinstance(x, Base) or isinstance(x, da.Array):
        return _num_samples(x)

    return delayed(_num_samples, pure=True)(x).compute()


def copy_estimator(est):
    # Semantically, we'd like to use `sklearn.clone` here instead. However,
    # `sklearn.clone` isn't threadsafe, so we don't want to call it in
    # tasks.  Since `est` is guaranteed to not be a fit estimator, we can
    # use `copy.deepcopy` here without fear of copying large data.
    return copy.deepcopy(est)
