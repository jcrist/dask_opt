from __future__ import absolute_import, division, print_function, unicode_literals
import copy
from distutils.version import LooseVersion

import dask
import dask.array as da
from dask.base import tokenize
from dask.delayed import delayed, Delayed

from sklearn.utils.validation import indexable, _is_arraylike
from sklearn.pipeline import Pipeline as sk_Pipeline

def is_pipeline(estimator):
    if isinstance(estimator, sk_Pipeline):
        return True
    try:
        from elm.pipeline import Pipeline as elm_Pipeline
        return isinstance(estimator, elm_Pipeline)
    except:
        return False


def _get_est_type(est):
    if hasattr(est, '_cls_name'):
        est_type = est._cls_name.lower()
    else:
        est_type = type(est).__name__.lower()
    return est_type


if LooseVersion(dask.__version__) > '0.15.4':
    from dask.base import is_dask_collection
else:
    from dask.base import Base

    def is_dask_collection(x):
        return isinstance(x, Base)


def _indexable(x):
    return indexable(x)[0]


def _maybe_indexable(x):
    return indexable(x)[0] if _is_arraylike(x) else x


def to_indexable(*args, **kwargs):
    """Ensure that all args are an indexable type.

    Conversion runs lazily for dask objects, immediately otherwise.

    Parameters
    ----------
    args : array_like or scalar
    allow_scalars : bool, optional
        Whether to allow scalars in args. Default is False.
    """
    if kwargs.get('allow_scalars', False):
        indexable = _maybe_indexable
    else:
        indexable = _indexable
    for x in args:
        if x is None or isinstance(x, da.Array):
            yield x
        elif is_dask_collection(x):
            yield delayed(indexable, pure=True)(x)
        else:
            yield indexable(x)


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
            assert not is_dask_collection(x)
            key = 'array-' + tokenize(x)
            dsk[key] = x
            yield key


def copy_estimator(est):
    # Semantically, we'd like to use `sklearn.clone` here instead. However,
    # `sklearn.clone` isn't threadsafe, so we don't want to call it in
    # tasks.  Since `est` is guaranteed to not be a fit estimator, we can
    # use `copy.deepcopy` here without fear of copying large data.
    return copy.deepcopy(est)


def unzip(itbl, n):
    return zip(*itbl) if itbl else [()] * n


def _is_xy_tuple(result, typ=tuple):
    if typ and not isinstance(typ, tuple):
        typ = (typ,)
    typ = typ + (tuple,)
    return isinstance(result, typ) and len(result) == 2


def _split_Xy(X, y, typ=tuple):
    if _is_xy_tuple(X, typ=typ):
        X, y = X
    return X, y
