from __future__ import absolute_import, print_function, division

from dask.base import tokenize
from dask.delayed import delayed
from sklearn import pipeline

from .core import DaskBaseEstimator, from_sklearn


class Pipeline(DaskBaseEstimator, pipeline.Pipeline):
    _finalize = staticmethod(lambda res: Pipeline(res[0]))

    def __init__(self, steps):
        # Run the sklearn init to validate the pipeline steps
        steps = pipeline.Pipeline(steps).steps
        steps = [(k, from_sklearn(v)) for k, v in steps]
        object.__setattr__(self, 'steps', steps)
        self._reset()

    def _reset(self, full=False):
        if full:
            for n, s in self.steps:
                s._reset(full=True)
        self._name = 'pipeline-' + tokenize(self.steps)
        self._dask = None

    @property
    def dask(self):
        if self._dask is None:
            dsk = {}
            names = []
            tasks = []
            for n, s in self.steps:
                dsk.update(s.dask)
                names.append(n)
                tasks.append((s._finalize, s._keys()))
            dsk[self._name] = (list, (zip, names, tasks))
            self._dask = dsk
        return self._dask

    def _keys(self):
        return [self._name]

    @classmethod
    def from_sklearn(cls, est):
        if not isinstance(est, pipeline.Pipeline):
            raise TypeError("est must be a sklearn Pipeline")
        return cls(est.steps)

    def to_sklearn(self, compute=True):
        steps = [(n, s.to_sklearn(compute=False)) for n, s in self.steps]
        pipe = delayed(pipeline.Pipeline, pure=True)
        res = pipe(steps, dask_key_name='to_sklearn-' + self._name)
        if compute:
            return res.compute()
        return res

    def set_params(self, **params):
        super(Pipeline, self).set_params(**params)
        self._reset()
        return self

    def __setattr__(self, k, v):
        if k in ('_name', '_dask'):
            object.__setattr__(self, k, v)
        else:
            raise AttributeError("Attribute setting not permitted. "
                                 "Use `set_params` to change parameters")

    def _pre_transform(self, Xt, y=None, **fit_params):
        # Separate out parameters
        fit_params_steps = dict((step, {}) for step, _ in self.steps)
        for pname, pval in fit_params.items():
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval
        # Call fit_transform on all but last estimator
        for name, transform in self.steps[:-1]:
            kwargs = fit_params_steps[name]
            Xt = transform.fit_transform(Xt, y, **kwargs)
        return Xt, fit_params_steps[self.steps[-1][0]]

    def fit(self, X, y=None, **fit_params):
        # Reset graph
        self._reset(full=True)
        # Perform the fit
        name = 'fit-' + tokenize(self, X, y, fit_params)
        Xt, params = self._pre_transform(X, y, **fit_params)
        self._final_estimator.fit(Xt, y, **params)
        self._name = name
        return self

    def transform(self, X):
        for name, transform in self.steps:
            X = transform.transform(X)
        return X

    def predict(self, X):
        for name, transform in self.steps[:-1]:
            X = transform.transform(X)
        return self._final_estimator.predict(X)

    def score(self, X, y=None):
        for name, transform in self.steps[:-1]:
            X = transform.transform(X)
        return self._final_estimator.score(X, y)

    def fit_transform(self, X, y=None, **fit_params):
        # Reset graph
        self._reset(full=True)
        # Perform the fit and transform
        name = 'fit-' + tokenize(self, X, y, fit_params)
        Xt, params = self._pre_transform(X, y, **fit_params)
        Xt = self._final_estimator.fit_transform(Xt, y, **params)
        self._name = name
        return self, Xt


from_sklearn.dispatch.register(pipeline.Pipeline, Pipeline.from_sklearn)
