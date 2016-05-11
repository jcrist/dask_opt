from __future__ import division, print_function, absolute_import

import dask.array as da
from dask.delayed import Delayed
from dask import delayed
import numpy as np
from sklearn import clone


__all__ = ['averaged_fit', 'chained_partial_fit', 'logistic_gradient']


def is_list_of(x, typ):
    """Is `x` a list of type `typ`"""
    return isinstance(x, list) and all(isinstance(i, typ) for i in x)


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


@delayed(pure=True)
def fit_chunk(cls, params, X, y, kwargs):
    return cls(**params).fit(X, y, **kwargs)


@delayed(pure=True)
def merge_classifiers(models):
    if len(models) == 1:
        return models[0]
    o = clone(models[0])
    classes = np.unique(np.concatenate([m.classes_ for m in models]))
    o.classes_ = classes
    n_classes = len(classes)
    if all(m.classes_.size == n_classes for m in models):
        o.coef_ = np.mean([m.coef_ for m in models], axis=0)
        o.intercept_ = np.mean([m.intercept_ for m in models], axis=0)
    else:
        # Not all models got all classes. Multiclass problems are fit using
        # ovr, which results in a row per class. Here we average the
        # coefficients for each class, using zero if that class wasn't fit.
        n_features = models[0].coef_.shape[1]
        coef = np.zeros((n_classes, n_features), dtype='f8')
        intercept = np.zeros(n_classes, dtype='f8')
        for m in models:
            ind = np.in1d(classes, m.classes_)
            coef[ind] += m.coef_
            intercept[ind] += m.intercept_
        o.coef_ = coef / len(models)
        o.intercept_ = intercept / len(models)
    return o


@delayed(pure=True)
def merge_regressors(models):
    if len(models) == 1:
        return models[0]
    o = clone(models[0])
    o.coef_ = np.mean([m.coef_ for m in models], axis=0)
    o.intercept_ = np.mean([m.intercept_ for m in models], axis=0)
    return o


def delayed_solver(f):
    def fit(self, X, y, compute=True, **kwargs):
        model = f(self, X, y, **kwargs)

        if compute:
            fitted = model.compute()
            self.coef_ = fitted.coef_
            self.intercept_ = fitted.intercept_
            if hasattr(fitted, 'classes_'):
                self.classes_ = fitted.classes_
            return self
        return model
    return fit


@delayed_solver
def averaged_fit(self, X, y, **kwargs):
    """Fit a model by averaging fits of the chunks.

    Fits a separate model to each chunk, and then averages the coefficients and
    intercepts to produce a final model.

    Parameters
    ----------
    self : estimator
        A dask wrapped linear classifier or regressor.
    X, y: dask.array.Array, list of dask.delayed.Delayed
        The X and y arrays to fit. Can be either instances of
        ``dask.array.Array`` or lists of ``dask.delayed.Delayed`` objects that
        represent delayed ``ndarray``s.

    Returns
    -------
    estimator : Delayed
        A dask.delayed.Delayed object holding the graph to compute the
        estimator.
    """
    x_parts, y_parts = xy_to_parts(X, y)
    params = self.get_params()
    params.pop('dask_solver')
    cls = self._estimator
    chunks = [fit_chunk(cls, params, xp, yp, kwargs) for (xp, yp) in
              zip(x_parts, y_parts)]
    if cls._estimator_type == 'classifier':
        return merge_classifiers(chunks)
    return merge_regressors(chunks)


@delayed(pure=True)
def partial_fit_chunk(model, x, y, kwargs):
    return model.partial_fit(x, y, **kwargs)


@delayed(pure=True)
def _unique_chunk(x):
    return np.unique(x)


@delayed(pure=True)
def _unique_merge(x):
    return np.unique(np.concatenate(x))


def unique(x):
    if is_list_of(x, Delayed):
        return _unique_merge([_unique_chunk(i) for i in x])
    return da.unique(x)


@delayed_solver
def chained_partial_fit(self, X, y, **kwargs):
    """Fit a model by chaining calls to ``partial_fit`` for each chunk.

    Parameters
    ----------
    self : estimator
        A dask wrapped linear classifier or regressor supporting
        ``partial_fit``.
    X, y: dask.array.Array, list of dask.delayed.Delayed
        The X and y arrays to fit. Can be either instances of
        ``dask.array.Array`` or lists of ``dask.delayed.Delayed`` objects that
        represent delayed ``ndarray``s.
    Returns
    -------
    estimator : Delayed
        A dask.delayed.Delayed object holding the graph to compute the
        estimator.
    """
    params = self.get_params()
    params.pop('dask_solver')
    model = self._estimator(**params)
    if not hasattr(model, 'partial_fit'):
        raise ValueError("model must support `partial_fit`")
    if 'classes' not in kwargs:
        kwargs['classes'] = unique(y)
    for x, y in zip(*xy_to_parts(X, y)):
        model = partial_fit_chunk(model, x, y, kwargs)
    return model


# Gradient Descent algorithm

def logistic_gradient(self, X, y, **kwargs):
    if not kwargs.pop('compute', True):
        raise ValueError("gradient solver doesn't support delayed compute")

    classes = kwargs.pop('classes', False) or da.unique(y)
    if len(classes) != 2:
        raise ValueError("gradient solver doesn't support multiclass")
    coded_y = (y == classes[1]).astype('i1')

    w, c = gradient_descent(X, coded_y, fit_intercept=self.fit_intercept,
                            max_iter=self.max_iter, tol=self.tol,
                            verbose=self.verbose)
    self.coef_ = w
    self.intercept_ = c
    self.classes_ = classes


def _intercept_dot(X, w, c):
    return X.dot(w) + c


def _intercept_norm(w, c):
    return ((w**2).sum() + c**2)**0.5


def gradient_descent(X, y, fit_intercept=True, max_iter=100, tol=1e-4,
                     verbose=False):
    """Solve a logistic regression problem using gradient descent"""

    log = print if verbose else lambda x: x
    # Tuning Params
    first_backtrack_mult = 0.1
    next_backtrack_mult = 0.5
    armijo_mult = 0.1
    step_growth = 1.25
    step_size = 1.0

    # Init
    w = np.zeros(X.shape[1])
    c = 0.0
    backtrack_mult = first_backtrack_mult

    log('##       -f        |df/f|    |dw/w|    step\n'
        '-------------------------------------------')
    for k in range(1, max_iter + 1):
        # Compute the gradient
        Xw = _intercept_dot(X, w, c)
        eXw = da.exp(Xw)
        f = da.log1p(eXw).sum() - y.dot(Xw)
        mult = eXw/(eXw + 1) - y
        grad = X.T.dot(mult)
        c_grad = mult.sum() if fit_intercept else 0.0
        Xgrad = _intercept_dot(X, grad, c_grad)

        Xw, f, grad, Xgrad, c_grad = da.compute(Xw, f, grad, Xgrad, c_grad)

        step_len = _intercept_norm(grad, c_grad)

        # Compute the step size using line search
        old_w = w
        old_c = c
        old_Xw = Xw
        old_f = f
        for ii in range(100):
            w = old_w - step_size * grad
            if fit_intercept:
                c = old_c - step_size * c_grad
            if ii and np.array_equal(w, old_w) and c == old_c:
                step_size = 0
                break
            Xw = old_Xw - step_size * Xgrad
            # This prevents overflow
            if np.all(Xw < 700):
                eXw = np.exp(Xw)
                f = np.log1p(eXw).sum() - np.dot(y, Xw)
                df = old_f - f
                if df >= armijo_mult * step_size * step_len**2:
                    break
            step_size *= backtrack_mult

        if step_size == 0:
            log('No more progress')
            break
        df /= max(f, old_f)
        dw = (step_size * step_len /
              (_intercept_norm(w, c) + step_size * step_len))
        log('%2d  %.6e %-.2e  %.2e  %.1e' % (k, f, df, dw, step_size))
        if df < tol:
            log('Converged')
            break
        step_size *= step_growth
        backtrack_mult = next_backtrack_mult
    else:
        log('Maximum Iterations')
    return w[None, :], np.atleast_1d(c)
