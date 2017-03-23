from __future__ import absolute_import, division, print_function

import warnings
from collections import defaultdict
from threading import Lock

import numpy as np
from toolz import pluck
from scipy import sparse
from dask.base import normalize_token

from sklearn.exceptions import FitFailedWarning
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.utils import safe_indexing
from sklearn.utils.fixes import rankdata
from sklearn.utils.validation import check_consistent_length, _is_arraylike

from .utils import copy_estimator

try:
    from sklearn.utils.fixes import MaskedArray
except:
    from numpy.ma import MaskedArray

# A singleton to indicate a missing parameter
MISSING = type('MissingParameter', (object,),
               {'__slots__': (),
                '__reduce__': lambda self: 'MISSING',
                '__doc__': "A singleton to indicate a missing parameter"})()
normalize_token.register(type(MISSING), lambda x: 'MISSING')


# A singleton to indicate a failed estimator fit
FIT_FAILURE = type('FitFailure', (object,),
                   {'__slots__': (),
                    '__reduce__': lambda self: 'FIT_FAILURE',
                    '__doc__': "A singleton to indicate fit failure"})()


def warn_fit_failure(error_score, e):
    warnings.warn("Classifier fit failed. The score on this train-test"
                  " partition for these parameters will be set to %f. "
                  "Details: \n%r" % (error_score, e), FitFailedWarning)


# ----------------------- #
# Functions in the graphs #
# ----------------------- #


def cv_split(cv, X, y, groups):
    check_consistent_length(X, y, groups)
    return list(cv.split(X, y, groups))


def cv_n_samples(cvs):
    return np.array([i.sum() if i.dtype == bool else len(i)
                     for i in pluck(1, cvs)])


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


def cv_extract_param(x, indices):
    return safe_indexing(x, indices) if _is_arraylike(x) else x


def pipeline(names, steps):
    """Reconstruct a Pipeline from names and steps"""
    if any(s is FIT_FAILURE for s in steps):
        return FIT_FAILURE
    return Pipeline(list(zip(names, steps)))


def feature_union(names, steps, weights):
    """Reconstruct a FeatureUnion from names, steps, and weights"""
    if any(s is FIT_FAILURE for s in steps):
        return FIT_FAILURE
    return FeatureUnion(list(zip(names, steps)),
                        transformer_weights=weights)


def feature_union_empty(X):
    return np.zeros((X.shape[0], 0))


def feature_union_concat(Xs, weights):
    """Apply weights and concatenate outputs from a FeatureUnion"""
    if any(x is FIT_FAILURE for x in Xs):
        return FIT_FAILURE
    Xs = [X if w is None else X * w for X, w in zip(Xs, weights)]
    if any(sparse.issparse(f) for f in Xs):
        return sparse.hstack(Xs).tocsr()
    return np.hstack(Xs)


# Current set_params isn't threadsafe
SET_PARAMS_LOCK = Lock()


def set_params(est, fields=None, params=None, copy=True):
    if copy:
        est = copy_estimator(est)
    if fields is None:
        return est
    params = {f: p for (f, p) in zip(fields, params) if p is not MISSING}
    # TODO: rewrite set_params to avoid lock for classes that use the standard
    # set_params/get_params methods
    with SET_PARAMS_LOCK:
        return est.set_params(**params)


def fit(est, X, y, error_score='raise', fields=None, params=None):
    if est is FIT_FAILURE or X is FIT_FAILURE:
        return FIT_FAILURE
    try:
        est = set_params(est, fields, params)
        est.fit(X, y)
    except Exception as e:
        if error_score == 'raise':
            raise
        warn_fit_failure(error_score, e)
        est = FIT_FAILURE
    return est


def fit_transform(est, X, y, error_score='raise', fields=None, params=None):
    if est is FIT_FAILURE or X is FIT_FAILURE:
        return FIT_FAILURE, FIT_FAILURE
    try:
        est = set_params(est, fields, params)
        if hasattr(est, 'fit_transform'):
            Xt = est.fit_transform(X, y)
        else:
            est.fit(X, y)
            Xt = est.transform(X)
    except Exception as e:
        if error_score == 'raise':
            raise
        warn_fit_failure(error_score, e)
        est = Xt = FIT_FAILURE
    return est, Xt


def score(est, X, y, scorer):
    if est is FIT_FAILURE:
        return FIT_FAILURE
    return scorer(est, X) if y is None else scorer(est, X, y)


def _store(results, key_name, array, n_splits, n_candidates,
           weights=None, splits=False, rank=False):
    """A small helper to store the scores/times to the cv_results_"""
    # When iterated first by parameters then by splits
    array = np.array(array, dtype=np.float64).reshape(n_candidates, n_splits)
    if splits:
        for split_i in range(n_splits):
            results["split%d_%s" % (split_i, key_name)] = array[:, split_i]

    array_means = np.average(array, axis=1, weights=weights)
    results['mean_%s' % key_name] = array_means
    # Weighted std is not directly available in numpy
    array_stds = np.sqrt(np.average((array - array_means[:, np.newaxis]) ** 2,
                                    axis=1, weights=weights))
    results['std_%s' % key_name] = array_stds

    if rank:
        results["rank_%s" % key_name] = np.asarray(
            rankdata(-array_means, method='min'), dtype=np.int32)


def create_cv_results(test_scores, train_scores, candidate_params, n_splits,
                      error_score, weights):
    test_scores = [error_score if s is FIT_FAILURE else s for s in test_scores]
    if train_scores is not None:
        train_scores = [error_score if s is FIT_FAILURE else s
                        for s in train_scores]

    # Construct the `cv_results_` dictionary
    results = {'params': candidate_params}
    n_candidates = len(candidate_params)

    if weights is not None:
        weights = np.broadcast_to(weights[None, :],
                                  (len(candidate_params), len(weights)))

    _store(results, 'test_score', test_scores, n_splits, n_candidates,
           splits=True, rank=True, weights=weights)
    if train_scores is not None:
        _store(results, 'train_score', train_scores,
               n_splits, n_candidates, splits=True)

    # Use one MaskedArray and mask all the places where the param is not
    # applicable for that candidate. Use defaultdict as each candidate may
    # not contain all the params
    param_results = defaultdict(lambda: MaskedArray(np.empty(n_candidates),
                                                    mask=True,
                                                    dtype=object))
    for cand_i, params in enumerate(candidate_params):
        for name, value in params.items():
            param_results["param_%s" % name][cand_i] = value

    results.update(param_results)
    return results


def get_best_params(candidate_params, cv_results):
    best_index = np.flatnonzero(cv_results["rank_test_score"] == 1)[0]
    return candidate_params[best_index]


def fit_best(estimator, params, X, y):
    estimator = copy_estimator(estimator).set_params(**params)
    estimator.fit(X, y)
    return estimator
