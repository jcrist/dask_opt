from dask_searchcv.adaptive import _top_k, Hyperband
from dask_ml.linear_model import PartialSGDClassifier
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


class TestFunction:
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


def test_hyperband_sklearn():
    client = _get_client()
    X, y = make_classification(n_samples=1000, chunks=500)
    classes = da.unique(y)
    model = PartialSGDClassifier(warm_start=True, classes=classes,
                                 loss='hinge', penalty='elasticnet')

    params = {'alpha': np.logspace(-3, 0, num=int(10e3)),
              'l1_ratio': np.linspace(0, 1, num=int(10e3))}
    alg = Hyperband(model, params, max_iter=9, n_jobs=0)

    alg.fit(X, y, dry_run=True)
    assert len(alg.history) == 20
    alg.fit(X, y, dry_run=True)
    assert len(alg.history) == 40

    alg.fit(X, y, classes=classes)  # make sure no exceptions are raised


def test_hyperband_test_model():
    client = _get_client()
    X, y = make_classification(n_samples=20, n_features=20, chunks=20)
    model = TestFunction()
    max_iter = 81

    values = np.random.RandomState(42).rand(int(max_iter))
    params = {'value': values}
    with pytest.warns(UserWarning, match='model has no attribute warm_start'):
        alg = Hyperband(model, params, max_iter=max_iter, n_jobs=0)

    alg.fit(X, y)

    df = pd.DataFrame(alg.cv_results_)
    assert set(df.param_value) == set(values)
    assert (df.test_score == df.param_value).all()  # more of a TestFunction test
    assert alg.best_params_['value'] == alg.best_estimator_.value
    assert alg.best_params_['value'] == values.max()
    assert alg.cv_results_['test_score'][alg.best_index_] == values.max()


def test_hyperband_needs_partial_fit():
    client = _get_client()
    X, y = make_classification(n_samples=20, n_features=20, chunks=20)
    model = Lasso()
    params = {'none': None}
    with pytest.raises(ValueError, match='Hyperband relies on partial_fit'):
        alg = Hyperband(model, params, max_iter=81)


def test_hyperband_n_jobs():
    client = _get_client()
    X, y = make_classification(n_samples=20, n_features=20, chunks=20)
    model = TestFunction()
    params = {'value': [1, 2, 3]}

    with pytest.raises(ValueError, match='n_jobs must be'):
        alg = Hyperband(model, params, max_iter=3, n_jobs=1)


def test_info():
    client = _get_client()
    X, y = make_classification(n_samples=20, n_features=20, chunks=20)

    model = TestFunction()
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


def test_hyperband_with_distributions():
    client = _get_client()
    X, y = make_classification(n_samples=20, n_features=20, chunks=20)
    model = TestFunction()
    max_iter = 81

    values = stats.uniform(0, 1)

    params = {'value': values}
    with pytest.warns(UserWarning, match='model has no attribute warm_start'):
        alg = Hyperband(model, params, max_iter=max_iter, n_jobs=0)

    alg.fit(X, y)

    assert len(alg.cv_results_['param_value']) == alg.info()['num_models']


def test_hyperband_needs_client():
    # make sure there's no client active
    client = _get_client()
    client.close()

    X, y = make_classification(n_samples=20, n_features=20, chunks=20)
    model = PartialSGDClassifier(classes=da.unique(y), warm_start=True)
    params = {'value': np.logspace(-3, 0, num=100)}

    alg = Hyperband(model, params, max_iter=81, n_jobs=0)
    with pytest.raises(ValueError, match='No global distributed client found'):
        alg.fit(X, y)


def test_top_k(k=2):
    keys = range(10)
    scores = {str(i): -i for i in keys}
    models = {str(i): str(i) for i in keys}
    y = _top_k(models, scores, k=k)
    assert y == {str(i): str(i) for i in range(k)}


def _get_client():
    try:
        client = distributed.get_client()
    except ValueError:
        client = Client()
    return client
