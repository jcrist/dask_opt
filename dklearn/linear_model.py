from __future__ import print_function, absolute_import, division

import numbers
import warnings

import dask.array as da
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import (check_array, check_consistent_length,
                           compute_class_weight)
from sklearn.utils.extmath import log_logistic
from sklearn.utils.fixes import expit
from sklearn.utils.multiclass import check_classification_targets

from .core import ImmediateDaskEstimator, check_X_y


# The following is a fairly literal translation of scikit-learn's lbfgs
# implementation (see:
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/linear_model/logistic.py)
# to work with dask.


def _intercept_dot(w, X, y):
    """Computes y * np.dot(X, w)."""
    c = 0.
    if w.size == X.shape[1] + 1:
        c = w[-1]
        w = w[:-1]
    z = X.dot(w)
    if c is not 0.:
        z = z + c
    yz = y * z
    return w, c, yz


def _logistic_loss_and_grad(w, X, y, alpha, sample_weight=None):
    n_samples, n_features = X.shape
    grad = np.empty_like(w)

    fit_c = grad.shape[0] > n_features

    w, c, yz = _intercept_dot(w, X, y)

    z = yz.map_blocks(expit, dtype='f8')
    z0 = (z - 1) * y
    log_log_yz = yz.map_blocks(log_logistic, dtype='f8')

    if sample_weight is not None:
        z0 = sample_weight * z0
        log_log_yz = sample_weight * log_log_yz

    # Dask outputs
    log_log_yz_sum = log_log_yz.sum()
    X_T_dot_z0 = X.T.dot(z0)
    results = [log_log_yz_sum, X_T_dot_z0]
    if fit_c:
        results.append(z0.sum())
    results = da.compute(*results)
    log_log_yz_sum, X_T_dot_z0 = results[:2]

    # Logistic loss is the negative of the log of the logistic function.
    out = -log_log_yz_sum + .5 * alpha * np.dot(w, w)
    grad[:n_features] = X_T_dot_z0 + alpha * w

    # Case where we fit the intercept.
    if fit_c:
        grad[-1] = results[2]
    return out, grad


def logistic_regression_path(X, y, pos_class=None, Cs=10, fit_intercept=True,
                             max_iter=100, tol=1e-4, verbose=0, coef=None,
                             class_weight=None, penalty='l2',
                             check_input=True, sample_weight=None):
    """Compute a Logistic Regression model for a list of regularization
    parameters.

    This is an implementation that uses the result of the previous model
    to speed up computations along the set of solutions, making it faster
    than sequentially calling LogisticRegression for the different parameters.

    Read more in the :ref:`User Guide <logistic_regression>`.

    Parameters
    ----------
    X : array-like or sparse matrix, shape (n_samples, n_features)
        Input data.

    y : array-like, shape (n_samples,)
        Input data, target values.

    Cs : int | array-like, shape (n_cs,)
        List of values for the regularization parameter or integer specifying
        the number of regularization parameters that should be used. In this
        case, the parameters will be chosen in a logarithmic scale between
        1e-4 and 1e4.

    pos_class : int, None
        The class with respect to which we perform a one-vs-all fit.
        If None, then it is assumed that the given problem is binary.

    fit_intercept : bool
        Whether to fit an intercept for the model. In this case the shape of
        the returned array is (n_cs, n_features + 1).

    max_iter : int
        Maximum number of iterations for the solver.

    tol : float
        Stopping criterion. For the newton-cg and lbfgs solvers, the iteration
        will stop when ``max{|g_i | i = 1, ..., n} <= tol``
        where ``g_i`` is the i-th component of the gradient.

    verbose : int
        Set verbose to any positive number for verbosity.

    coef : array-like, shape (n_features,), default None
        Initialization value for coefficients of logistic regression.

    class_weight : dict or 'balanced', optional
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    penalty : str, 'l1' or 'l2'
        Used to specify the norm used in the penalization. Currently only 'l2'
        is supported.

    check_input : bool, default True
        If False, the input arrays X and y will not be checked.

    sample_weight : array-like, shape(n_samples,) optional
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.

    Returns
    -------
    coefs : ndarray, shape (n_cs, n_features) or (n_cs, n_features + 1)
        List of coefficients for the Logistic Regression model. If
        fit_intercept is set to True then the second dimension will be
        n_features + 1, where the last item represents the intercept.

    Cs : ndarray
        Grid of Cs used for cross-validation.

    n_iter : array, shape (n_cs,)
        Actual number of iteration for each Cs.
    """
    if isinstance(Cs, numbers.Integral):
        Cs = np.logspace(-4, 4, Cs)

    # TODO: Validate inputs
    if isinstance(y, da.Array):
        y = y.compute()

    # Preprocessing.
    if check_input:
        X = check_array(X, accept_sparse='csr', dtype=np.float64)
        y = check_array(y, ensure_2d=False, dtype=None)
        check_consistent_length(X, y)
    n_features = X.shape[1]
    classes = np.unique(y)

    if (classes.size > 2):
        raise ValueError('To fit OvR, use the pos_class argument')
    # np.unique(y) gives labels in sorted order.
    pos_class = classes[1]

    # If sample weights exist, convert them to array (support for lists)
    # and check length
    # Otherwise leave as None, which is special-cased further on
    if sample_weight is not None:
        sample_weight = np.array(sample_weight, dtype=np.float64, order='C')
        check_consistent_length(y, sample_weight)

    # If class_weights is a dict (provided by the user), the weights
    # are assigned to the original labels. If it is "balanced", then
    # the class_weights are assigned after masking the labels with a OvR.
    le = LabelEncoder()

    # For doing a ovr, we need to mask the labels first.
    w0 = np.zeros(n_features + int(fit_intercept))
    mask_classes = np.array([-1, 1])
    mask = (y == pos_class)
    y_bin = np.ones(y.shape, dtype=np.float64)
    y_bin[~mask] = -1.
    if class_weight == "balanced":
        class_weight_ = compute_class_weight(class_weight, mask_classes, y_bin)
        sample_weight *= class_weight_[le.fit_transform(y_bin)]

    if coef is not None:
        # it must work both giving the bias term and not
        if coef.size not in (n_features, w0.size):
            raise ValueError(
                'Initialization coef is of shape %d, expected shape '
                '%d or %d' % (coef.size, n_features, w0.size))
        w0[:coef.size] = coef

    coefs = list()
    n_iter = np.zeros(len(Cs), dtype=np.int32)
    for i, C in enumerate(Cs):
        assert isinstance(X, da.Array)
        w0, loss, info = fmin_l_bfgs_b(_logistic_loss_and_grad, w0,
                                       fprime=None,
                                       args=(X, y_bin, 1/C, sample_weight),
                                       iprint=(verbose > 0) - 1,
                                       pgtol=tol, maxiter=max_iter)

        if info["warnflag"] == 1 and verbose > 0:
            warnings.warn("Failed to converge. Increase the number "
                          "of iterations.")
        n_iter_i = info['nit'] - 1

        coefs.append(w0.copy())

        n_iter[i] = n_iter_i

    return coefs, np.array(Cs), n_iter


class LogisticRegression(ImmediateDaskEstimator,
                         linear_model.LogisticRegression):

    def __init__(self, penalty='l2', tol=1e-4, C=1.0, fit_intercept=True,
                 class_weight=None, max_iter=100, verbose=0):
        self.penalty = penalty
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        if not isinstance(self.C, numbers.Number) or self.C < 0:
            raise ValueError("Penalty term must be positive; got (C=%r)"
                             % self.C)
        if not isinstance(self.max_iter, numbers.Number) or self.max_iter < 0:
            raise ValueError("Maximum number of iteration must be positive;"
                             " got (max_iter=%r)" % self.max_iter)
        if not isinstance(self.tol, numbers.Number) or self.tol < 0:
            raise ValueError("Tolerance for stopping criteria must be "
                             "positive; got (tol=%r)" % self.tol)

        X, y = check_X_y(X, y)
        check_classification_targets(y)
        n_samples, n_features = X.shape

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        classes_ = self.classes_
        if n_classes < 2:
            raise ValueError("This solver needs samples of at least 2 classes"
                             " in the data, but the data contains only one"
                             " class: %r" % classes_[0])

        if len(self.classes_) == 2:
            n_classes = 1
            classes_ = classes_[1:]

        self.coef_ = list()
        self.intercept_ = np.zeros(n_classes)

        kwargs = dict(Cs=[self.C], fit_intercept=self.fit_intercept,
                      tol=self.tol, verbose=self.verbose,
                      max_iter=self.max_iter, class_weight=self.class_weight,
                      check_input=False, sample_weight=sample_weight)

        fold_coefs_ = [logistic_regression_path(X, y, pos_class=class_,
                                                **kwargs)
                       for class_ in classes_]

        fold_coefs_, _, n_iter_ = zip(*fold_coefs_)
        self.n_iter_ = np.asarray(n_iter_, dtype=np.int32)[:, 0]

        self.coef_ = np.asarray(fold_coefs_)
        self.coef_ = self.coef_.reshape(n_classes, n_features +
                                        int(self.fit_intercept))

        if self.fit_intercept:
            self.intercept_ = self.coef_[:, -1]
            self.coef_ = self.coef_[:, :-1]

        return self
