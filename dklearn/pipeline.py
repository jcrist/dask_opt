from __future__ import division, print_function, absolute_import

from dask.delayed import delayed, compute
from sklearn import clone, pipeline

from .core import LazyDaskEstimator
from .wrappers import fit_chunk, transform_chunk, score_chunk


@delayed(pure=True)
def fit_and_transform_chunk(est, X, y=None, **fit_params):
    """Call fit_transform, return fit and Xt"""
    est = clone(est)
    Xt = est.fit_transform(X, y, **fit_params)
    return est, Xt


class Pipeline(LazyDaskEstimator, pipeline.Pipeline):
    _delayed = None

    def _from_delayed(self, ests):
        o = clone(self)
        o._delayed = [(old[0], s) for old, s in zip(self.steps, ests)]
        return o

    @classmethod
    def from_sklearn(cls, sk_pipeline):
        return cls(sk_pipeline.steps)

    def to_delayed(self):
        return delayed(Pipeline, pure=True)(self._delayed)

    def compute(self, **kwargs):
        if self._delayed is None:
            return self
        self.steps = list(compute(*self._delayed, **kwargs))
        self._delayed = None
        return self

    @property
    def steps_(self):
        """Most recently fit estimators"""
        if self._delayed is not None:
            return self._delayed
        return self.steps

    def _pre_transform(self, Xt, y=None, **fit_params):
        # Separate out parameters
        fit_params_steps = dict((step, {}) for step, _ in self.steps)
        for pname, pval in fit_params.items():
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval
        # Call fit_transform on all but last estimator
        fit_steps = []
        for name, transform in self.steps[:-1]:
            kwargs = fit_params_steps[name]
            if isinstance(transform, LazyDaskEstimator):
                fit = transform.fit(Xt, y, compute=False, **kwargs)
                Xt = fit.transform(Xt)
            else:
                if hasattr(transform, "fit_transform"):
                    temp = fit_and_transform_chunk(transform, Xt, y, **kwargs)
                    fit, Xt = temp[0], temp[1]
                else:
                    fit = fit_chunk(transform, Xt, y, **kwargs)
                    Xt = transform_chunk(fit, Xt)
            fit_steps.append(fit)
        return fit_steps, Xt, fit_params_steps[self.steps[-1][0]]

    def fit(self, X, y=None, **fit_params):
        compute = fit_params.pop('compute', True)
        fit_steps, Xt, params = self._pre_transform(X, y, **fit_params)
        final = self._final_estimator
        if isinstance(final, LazyDaskEstimator):
            fit = self._final_estimator.fit(Xt, y, compute=False, **params)
        else:
            fit = fit_chunk(final, Xt, y, **params)
        fit_steps.append(fit)
        if not compute:
            return self._from_delayed(fit_steps)
        self._delayed = [(old[0], s) for old, s in zip(self.steps, fit_steps)]
        return self.compute()

    def score(self, X, y=None):
        if self._delayed is not None:
            for name, transform in self._delayed[:-1]:
                X = transform_chunk(transform, X)
            return score_chunk(self._final_estimator, X, y)
        else:
            return super(Pipeline, self).score(X, y)
