from __future__ import absolute_import, print_function, division

from functools import partial

from dask.delayed import Delayed, delayed
import dask.bag as db
from sklearn.feature_extraction import text

from ..base import DaskEstimator
from ..utils import is_list_of


def hashing_vectorizer_transform(x, model=None):
    return model.transform(x)


class HashingVectorizer(text.HashingVectorizer, DaskEstimator):
    _estimator = text.HashingVectorizer

    def transform(self, X, y=None):
        model = text.HashingVectorizer(**self.get_params())
        if isinstance(X, db.Bag):
            f = partial(hashing_vectorizer_transform, model=model)
            return X.map_partitions(f)
        elif is_list_of(X, Delayed):
            chunk = delayed(hashing_vectorizer_transform, pure=True)
            return [chunk(x, model) for x in X]
        else:
            raise TypeError("Expected either Bag or list of "
                            "Delayed, got {0}".format(type(X)))
