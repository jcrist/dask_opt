from dask_searchcv.adaptive import _top_k, Hyperband
from sklearn.linear_model import SGDClassifier
from dask_ml.datasets import make_classification
import numpy as np
import dask.array as da
from dask.distributed import Client
import pandas as pd
import pytest
import distributed
from pprint import pprint
import scipy.stats as stats
import random
from sklearn.linear_model import Lasso
from distributed.utils_test import cluster, loop
import time
from dask_ml.wrappers import Incremental


class ConstantFunction:
    def _fn(self):
        return self.value

    def get_params(self, deep=None, **kwargs):
        return {k: getattr(self, k) for k, v in kwargs.items()}

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    def partial_fit(self, *args, **kwargs):
        pass

    def score(self, *args, **kwargs):
        return self._fn()


def _get_client():
    try:
        return distributed.get_client()
    except ValueError:
        return Client()


def _with_client(fn):
    def fn_with_client(loop, *args, **kwargs):
        with cluster() as (s, [a, b]):
            with Client(s['address'], loop=loop):
                y = fn(*args, **kwargs)
        return y
    return fn_with_client


@_with_client
def test_hyperband_test_model(*args, **kwargs):
    X, y = make_classification(n_samples=20, n_features=20, chunks=20)
    model = ConstantFunction()
    max_iter = 81

    values = np.random.RandomState(42).rand(int(max_iter))
    params = {'value': values}
    with pytest.warns(UserWarning, match='model has no attribute warm_start'):
        alg = Hyperband(model, params, max_iter=max_iter, n_jobs=0)

    alg.fit(X, y)

    df = pd.DataFrame(alg.cv_results_)
    assert set(df.param_value) == set(values)
    assert (df.test_score == df.param_value).all()  # more of a ConstantFunction test
    assert alg.best_params_['value'] == alg.best_estimator_.value
    assert alg.best_params_['value'] == values.max()
    assert alg.cv_results_['test_score'][alg.best_index_] == values.max()


@_with_client
def test_hyperband_needs_partial_fit():
    X, y = make_classification(n_samples=20, n_features=20, chunks=20)
    model = Lasso()
    params = {'none': None}
    with pytest.raises(ValueError, match='Hyperband only supports models with partial_fit'):
        Hyperband(model, params)


@_with_client
def test_hyperband_n_jobs():
    X, y = make_classification(n_samples=20, n_features=20, chunks=20)
    model = ConstantFunction()
    params = {'value': [1, 2, 3]}

    with pytest.raises(ValueError, match='n_jobs must be'):
        alg = Hyperband(model, params, max_iter=3, n_jobs=1)


@_with_client
def test_hyperband_async():
    X, y = make_classification(n_samples=20, n_features=20, chunks=20)
    model = ConstantFunction()
    params = {'value': stats.uniform(0, 1)}

    alg = Hyperband(model, params, max_iter=3, n_jobs=-1)
    future = alg.fit(X, y)


    result = yield future



@_with_client
def test_info():
    X, y = make_classification(n_samples=20, n_features=20, chunks=20)

    model = ConstantFunction()
    max_iter = 9

    values = np.random.RandomState(42).rand(int(max_iter))
    params = {'value': values}
    with pytest.warns(UserWarning, match='model has no attribute warm_start'):
        alg = Hyperband(model, params, max_iter=max_iter, n_jobs=0)

    info = alg.info()
    expect = {'brackets': [{'bracket': 0.0,
                            'bracket_iter': 0.0,
                            'num_models': 3.0,
                            'partial_fit_iters': 9.0},
                           {'bracket': 1.0,
                            'bracket_iter': 0.0,
                            'num_models': 5.0,
                            'partial_fit_iters': 3.0},
                           {'bracket': 2.0,
                            'bracket_iter': 0.0,
                            'num_models': 9.0,
                            'partial_fit_iters': 1.0},
                           {'bracket': 2.0,
                            'bracket_iter': 1.0,
                            'num_models': 3.0,
                            'partial_fit_iters': 4.0}],
              'num_partial_fit_calls': 63.0,
              'num_models': 17,
              'num_cv_splits': 1}
    assert info == expect
    assert info['num_partial_fit_calls'] == sum(b['num_models'] * b['partial_fit_iters']
                                                for b in expect['brackets'])
    assert info['num_models'] == sum(b['num_models'] for b in expect['brackets']
                                     if b['bracket_iter'] == 0)
    assert expect['num_cv_splits'] == 1  # TODO: change this!


@_with_client
def test_hyperband_with_distributions():
    X, y = make_classification(n_samples=20, n_features=20, chunks=20)
    model = ConstantFunction()
    max_iter = 81

    values = stats.uniform(0, 1)

    params = {'value': values}
    with pytest.warns(UserWarning, match='model has no attribute warm_start'):
        alg = Hyperband(model, params, max_iter=max_iter, n_jobs=0)

    alg.fit(X, y)

    assert len(alg.cv_results_['param_value']) == alg.info()['num_models']


def test_hyperband_needs_client():
    X, y = make_classification(n_samples=20, n_features=20, chunks=20)
    model = Incremental(SGDClassifier(), warm_start=True)
    params = {'value': np.logspace(-3, 0, num=100)}

    alg = Hyperband(model, params, max_iter=81, n_jobs=0)
    with pytest.raises(ValueError, match='No global client'):
        with pytest.warns(UserWarning, match='No global distributed client found'):
            alg.fit(X, y)


def test_top_k(k=2):
    keys = range(10)
    scores = {str(i): -i for i in keys}
    models = {str(i): str(i) for i in keys}
    y = _top_k(models, scores, k=k)
    assert y == {str(i): str(i) for i in range(k)}


@_with_client
def test_hyperband_sklearn():
    # This test passes very incosistently, and often throws "RuntimeError:
    # IOLoop is closed". I rerun this by itself until it passes
    X, y = make_classification(n_samples=1000, chunks=500)
    classes = np.unique(y).tolist()
    model = Incremental(SGDClassifier(),
                        warm_start=True, loss='hinge', penalty='elasticnet')

    params = {'alpha': np.logspace(-3, 0, num=int(10e3)),
              'l1_ratio': np.linspace(0, 1, num=int(10e3))}
    alg = Hyperband(model, params, max_iter=9, n_jobs=0)

    alg.fit(X, y, dry_run=True, classes=classes)
    assert len(alg.history) == 20
    alg.fit(X, y, dry_run=True)
    assert len(alg.history) == 40

    alg.fit(X, y, classes=classes)
