from dask_searchcv.adaptive import _top_k, Hyperband
from sklearn.linear_model import SGDClassifier
from dask_ml.datasets import make_classification
import numpy as np
import dask.array as da
import pandas as pd
import pytest
from pprint import pprint
import scipy.stats as stats
import random
from sklearn.linear_model import Lasso
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

    def fit(self, *args):
        pass


def test_hyperband_test_model():
    X, y = make_classification(n_samples=20, n_features=20, chunks=20)
    model = ConstantFunction()
    max_iter = 9

    values = np.random.RandomState(42).rand(int(max_iter))
    params = {'value': values}
    with pytest.warns(UserWarning, match='model has no attribute warm_start'):
        alg = Hyperband(model, params, max_iter=max_iter)

    alg.fit(X, y)

    df = pd.DataFrame(alg.cv_results_)
    assert set(df.param_value) == set(values)
    assert (df.test_score == df.param_value).all()  # more of a ConstantFunction test
    assert alg.best_params_['value'] == alg.best_estimator_.value
    assert alg.best_params_['value'] == values.max()
    assert alg.cv_results_['test_score'][alg.best_index_] == values.max()


def test_hyperband_needs_partial_fit():
    X, y = make_classification(n_samples=20, n_features=20, chunks=20)
    model = Lasso()
    params = {'none': None}
    with pytest.raises(ValueError, match='models with partial_fit'):
        Hyperband(model, params)


@pytest.mark.parametrize("n_jobs", [-1, 0, 1, 2])
def test_hyperband_n_jobs(n_jobs):
    X, y = make_classification(n_samples=20, n_features=20, chunks=20)
    model = ConstantFunction()
    model.warm_start = True
    params = {'value': stats.uniform(0, 1)}

    if n_jobs in {-1, 1}:
        alg = Hyperband(model, params, max_iter=27, n_jobs=n_jobs)
        alg.fit(X, y)
    else:
        with pytest.raises(ValueError, match='n_jobs must be'):
            Hyperband(model, params, max_iter=27, n_jobs=n_jobs)


def test_score():
    X, y = make_classification(n_samples=20, n_features=20, chunks=20)

    model = ConstantFunction()
    max_iter = 9

    params = {'value': stats.uniform(0, 1)}
    with pytest.warns(UserWarning, match='warm_start'):
        alg = Hyperband(model, params, max_iter=max_iter)
    alg.fit(X, y).score(X, y)


def test_info():
    X, y = make_classification(n_samples=20, n_features=20, chunks=20)

    model = ConstantFunction()
    max_iter = 9

    values = np.random.RandomState(42).rand(int(max_iter))
    params = {'value': values}
    with pytest.warns(UserWarning, match='model has no attribute warm_start'):
        alg = Hyperband(model, params, max_iter=max_iter)
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
    X, y = make_classification(n_samples=20, n_features=20, chunks=20)
    model = ConstantFunction()

    values = stats.uniform(0, 1)

    params = {'value': values}
    with pytest.warns(UserWarning, match='model has no attribute warm_start'):
        alg = Hyperband(model, params, max_iter=9)

    alg.fit(X, y)

    assert len(alg.cv_results_['param_value']) == alg.info()['num_models']


def test_hyperband_defaults():
    X, y = make_classification(n_samples=20, n_features=20, chunks=20)
    model = SGDClassifier(warm_start=True, max_iter=5)
    params = {'alpha': np.logspace(-3, 0)}

    not_default = dict(iid=False, refit=False, cv=1,
                       error_score='warn',
                       return_train_score=False,
                       scheduler='whoops', cache_cv=False)
    for k, v in not_default.items():
        d = {k: v}
        with pytest.warns(UserWarning, match='Hyperband ignores'):
            Hyperband(model, params, max_iter=9, **d)


def test_top_k(k=2):
    keys = range(10)
    scores = {str(i): -i for i in keys}
    models = {str(i): str(i) for i in keys}
    y = _top_k(models, scores, k=k)
    assert y == {str(i): str(i) for i in range(k)}


from distributed.utils_test import loop, cluster
from distributed import Client
@pytest.mark.parametrize("n_jobs", [-1, 1])
def test_hyperband_sklearn(loop, n_jobs):
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop):

            X, y = make_classification(n_samples=100, chunks=50)
            classes = np.unique(y).tolist()
            model = Incremental(SGDClassifier(),
                                warm_start=True, loss='hinge', penalty='elasticnet',
                                max_iter=5)

            params = {'alpha': np.logspace(-3, 0, num=int(10e3)),
                      'l1_ratio': np.linspace(0, 1, num=int(10e3))}
            alg = Hyperband(model, params, max_iter=3)

            alg.fit(X, y, classes=classes)
            assert len(alg.history) == 5
            #  alg.fit(X, y)
            #  assert len(alg.history) == 10

            expected_keys = {'param_alpha', 'param_l1_ratio', 'bracket',
                             'bracket_iter', 'val_score', 'model_id',
                             'partial_fit_iters', 'num_models'}
            for hist_item in alg.history:
                assert set(hist_item.keys()) == expected_keys
