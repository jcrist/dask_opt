from __future__ import absolute_import, division, print_function

import numbers
from operator import getitem

import numpy as np
from dask.delayed import delayed
from dask.base import Base, tokenize
from dask.utils import Dispatch
from scipy.misc import comb
from sklearn.base import is_classifier
from sklearn.model_selection import check_cv as _sklearn_check_cv
from sklearn.model_selection._split import (_BaseKFold,
                                            BaseShuffleSplit,
                                            KFold,
                                            StratifiedKFold,
                                            LeaveOneOut,
                                            LeaveOneGroupOut,
                                            LeavePOut,
                                            LeavePGroupsOut,
                                            PredefinedSplit,
                                            _CVIterableWrapper)
from sklearn.utils import safe_indexing
from sklearn.utils.validation import check_consistent_length
from sklearn.utils.multiclass import type_of_target

from .utils import unique, num_samples, to_keys


def check_cv(cv=3, y=None, classifier=False):
    """Dask aware version of ``sklearn.model_selection.check_cv``

    Same as the scikit-learn version, but works if ``y`` is a dask object.
    """
    if cv is None:
        cv = 3

    # If ``cv`` is not an integer, the scikit-learn implementation doesn't
    # touch the ``y`` object, so passing on a dask object is fine
    if not (isinstance(y, Base) and isinstance(cv, numbers.Integral)):
        return _sklearn_check_cv(cv, y, classifier)

    if classifier:
        # ``y`` is a dask object. We need to compute the target type
        target_type = delayed(type_of_target, pure=True)(y).compute()
        if target_type in ('binary', 'multiclass'):
            return StratifiedKFold(cv)
    return KFold(cv)


# ------------------ #
# Functions in graph #
# ------------------ #

def cv_split(cv, X, y, groups):
    check_consistent_length(X, y, groups)
    return list(cv.split(X, y, groups))


def cv_extract(X, y, ind):
    return (safe_indexing(X, ind),
            None if y is None else safe_indexing(y, ind))


def cv_extract_pairwise(X, y, ind1, ind2):
    if not hasattr(X, "shape"):
        raise ValueError("Precomputed kernels or affinity matrices have "
                        "to be passed as arrays or sparse matrices.")
    if X.shape[0] != X.shape[1]:
        raise ValueError("X should be a square kernel matrix")
    return (X[np.ix_(ind1, ind2)],
            None if y is None else safe_indexing(y, ind1))


def initialize_dask_graph(estimator, cv, X, y, groups):
    """Initialize the dask graph for CV search.

    Parameters
    ----------
    estimator : BaseEstimator
    cv : cross validation object, integer, or None
    X, y, groups : array_like, dask_object, or None
    """
    cv = check_cv(cv, y, is_classifier(estimator))
    is_pairwise = getattr(estimator, '_pairwise', False)
    return splitter(cv).build_graph(X, y, groups, is_pairwise)


def _do_extract_splits(dsk, X_name, y_name, train_name, test_name,
                       n_splits, is_pairwise):
    token = tokenize(X_name, y_name, train_name, test_name, n_splits,
                     is_pairwise)
    # Extract the test-train subsets
    Xy_train = 'Xy-train-' + token
    X_train = 'X-train-' + token
    y_train = 'y-train-' + token
    Xy_test = 'Xy-test-' + token
    X_test = 'X-test-' + token
    y_test = 'y-test-' + token

    # Build a helper function to insert the extract tasks
    if is_pairwise:
        def extract_train(X, y, train, test):
            return (cv_extract_pairwise, X, y, train, train)

        def extract_test(X, y, train, test):
            return (cv_extract_pairwise, X, y, test, train)
    else:
        def extract_train(X, y, train, test):
            return (cv_extract, X, y, train)

        def extract_test(X, y, train, test):
            return (cv_extract, X, y, test)

    for n in range(n_splits):
        dsk[(Xy_train, n)] = extract_train(X_name, y_name, (train_name, n),
                                           (test_name, n))
        dsk[(X_train, n)] = (getitem, (Xy_train, n), 0)
        dsk[(y_train, n)] = (getitem, (Xy_train, n), 1)
        dsk[(Xy_test, n)] = extract_test(X_name, y_name, (train_name, n),
                                         (test_name, n))
        dsk[(X_test, n)] = (getitem, (Xy_test, n), 0)
        dsk[(y_test, n)] = (getitem, (Xy_test, n), 1)

    return X_train, y_train, X_test, y_test


splitter = Dispatch()


@splitter.register(object)
class DefaultSplitter(object):
    """Default wrapper for a scikit-learn CV object.

    Splitter classes wrap the scikit-learn CV objects, and are used to
    instantiate the dask graph for a CV run.
    """
    def __init__(self, cv):
        self.cv = cv

    def _compute_n_splits(self, X, y, groups):
        """Return the number of splits.

        Parameters
        ----------
        X, y, groups : array_like, dask object, or None

        Returns
        -------
        n_splits : int
        """
        if (isinstance(X, Base) or isinstance(y, Base) or
                isinstance(groups, Base)):
            return delayed(self.cv).get_n_splits(X, y, groups).compute()
        return self.cv.get_n_splits(X, y, groups)

    def _split_train_test(self, dsk, X_name, y_name, groups_name, n_splits):
        """Build the graph for generating the train/test indices.

        Parameters
        ----------
        dsk : dask graph
        X_name, y_name, groups_name : str or None
            The key names for X, y, and groups (if present).
        n_splits : int
            The number of splits

        Returns
        -------
        train_name, test_name : str
            The key prefixes for the train/test indices.
        """
        cv_token = tokenize(self.cv, X_name, y_name, groups_name)
        cv_name = 'cv-split-' + cv_token
        train_name = 'cv-split-train-' + cv_token
        test_name = 'cv-split-test-' + cv_token

        dsk[cv_name] = (cv_split, self.cv, X_name, y_name, groups_name)
        for n in range(n_splits):
            dsk[(cv_name, n)] = (getitem, cv_name, n)
            dsk[(train_name, n)] = (getitem, (cv_name, n), 0)
            dsk[(test_name, n)] = (getitem, (cv_name, n), 1)
        return train_name, test_name

    def build_graph(self, X, y=None, groups=None, is_pairwise=False):
        """Initialize the CV dask graph.

        Given input data X, y, and groups, build up a dask graph performing the
        initial CV splits.

        Parameters
        ----------
        X, y, groups : array_like, dask object, or None
        is_pairwise : bool
            Whether the estimator being evaluated has ``_pairwise`` as an
            attribute (which affects how the CV splitting is done).
        """
        n_splits = self._compute_n_splits(X, y, groups)
        # Build the graph
        dsk = {}
        X_name, y_name, groups_name = to_keys(dsk, X, y, groups)
        train_name, test_name = self._split_train_test(dsk, X_name, y_name,
                                                       groups_name, n_splits)
        (X_train, y_train,
         X_test, y_test) = _do_extract_splits(dsk, X_name, y_name, train_name,
                                              test_name, n_splits, is_pairwise)
        return dsk, X_train, y_train, X_test, y_test, n_splits


@splitter.register(_BaseKFold)
class KFoldSplitter(DefaultSplitter):
    def _compute_n_splits(self, X, y, groups):
        return self.cv.n_splits


@splitter.register(BaseShuffleSplit)
class ShuffleSplitSplitter(DefaultSplitter):
    def _compute_n_splits(self, X, y, groups):
        return self.cv.n_splits


@splitter.register(LeaveOneOut)
class LeaveOneOutSplitter(DefaultSplitter):
    def _compute_n_splits(self, X, y, groups):
        if isinstance(X, Base):
            return num_samples(X)
        return self.cv.get_n_splits(X, y, groups)


@splitter.register(LeavePOut)
class LeavePOutSplitter(DefaultSplitter):
    def _compute_n_splits(self, X, y, groups):
        if isinstance(X, Base):
            return int(comb(num_samples(X), self.cv.p, exact=True))
        return self.cv.get_n_splits(X, y, groups)


@splitter.register(LeaveOneGroupOut)
class LeaveOneGroupOutSplitter(DefaultSplitter):
    def _compute_n_splits(self, X, y, groups):
        if isinstance(X, Base):
            return len(unique(groups))
        return self.cv.get_n_splits(X, y, groups)


@splitter.register(LeavePGroupsOut)
class LeavePGroupsOutSplitter(DefaultSplitter):
    def _compute_n_splits(self, X, y, groups):
        if isinstance(X, Base):
            return int(comb(len(unique(groups)), self.cv.n_groups, exact=True))
        return self.cv.get_n_splits(X, y, groups)


@splitter.register(PredefinedSplit)
class PredefinedSplitSplitter(DefaultSplitter):
    def _compute_n_splits(self, X, y, groups):
        return len(self.cv.unique_folds)


@splitter.register(_CVIterableWrapper)
class IterableSplitter(DefaultSplitter):
    def _compute_n_splits(self, X, y, groups):
        return len(self.cv.cv)
