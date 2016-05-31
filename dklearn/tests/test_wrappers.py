from __future__ import division, print_function, absolute_import

from dask import compute
import dask.array as da
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import (LogisticRegression, SGDClassifier,
                                  PassiveAggressiveRegressor)
import pytest

from dklearn.wrappers import Averaged, Chained, DelayedEstimator

X_c, y_c = make_classification(1000)
dX_c = da.from_array(X_c, chunks=200)
dy_c = da.from_array(y_c, chunks=200)

X_r, y_r = make_regression(1000)
dX_r = da.from_array(X_r, chunks=200)
dy_r = da.from_array(y_r, chunks=200)


def test_dask_wrapper():
    est = LogisticRegression()
    dest = Averaged(est)

    res = dest.fit(dX_c, dy_c, compute=False)
    o = res.compute()
    assert isinstance(res, DelayedEstimator)
    assert isinstance(o, Averaged)
    assert o is not dest
    assert not hasattr(est, 'coef_')

    o2 = dest.fit(dX_c, dy_c)
    assert o2 is dest
    assert hasattr(est, 'coef_')


def test_averaged_classifier():
    # Classifier
    est = LogisticRegression()
    dest = Averaged(est)

    res = dest.fit(dX_c, dy_c, compute=False)
    res2 = dest.fit(dX_c, dy_c, compute=False)
    assert res.key == res2.key
    dest.set_params(estimator__C=10)
    res3 = dest.fit(dX_c, dy_c, compute=False)
    assert res.key != res3.key

    o = res.compute()
    assert isinstance(o, Averaged)
    assert o.estimator is not est
    assert hasattr(o.estimator, 'coef_')
    assert hasattr(o.estimator, 'intercept_')
    assert hasattr(o.estimator, 'classes_')

    # Regressor
    est = PassiveAggressiveRegressor()
    dest = Averaged(est)
    res = dest.fit(dX_r, dy_r, compute=False)

    o = res.compute()
    assert hasattr(o.estimator, 'coef_')
    assert hasattr(o.estimator, 'intercept_')
    assert not hasattr(o.estimator, 'classes_')


def test_chained():
    est = SGDClassifier()
    dest = Chained(est)

    res = dest.fit(dX_c, dy_c, compute=False)
    res2 = dest.fit(dX_c, dy_c, compute=False)
    assert res.key == res2.key
    dest.set_params(estimator__alpha=0.001)
    res3 = dest.fit(dX_c, dy_c, compute=False)
    assert res.key != res3.key

    o = res.compute()
    assert isinstance(o, Chained)
    assert o.estimator is not est
    assert hasattr(o.estimator, 'coef_')
    assert hasattr(o.estimator, 'intercept_')
    assert hasattr(o.estimator, 'classes_')


def test_delayed_estimator():
    est = LogisticRegression()
    dest = Averaged(est)

    res = dest.fit(dX_c, dy_c, compute=False)
    # getattr of passes to wrapped estimator
    assert isinstance(res.estimator, LogisticRegression)
    assert pytest.raises(AttributeError, lambda: res.fake_attr)
    # dir includes wrapped and base
    attrs = dir(res)
    assert 'estimator' in attrs
    assert 'compute' in attrs
    # immutable attributes
    assert pytest.raises(TypeError, lambda: setattr(res, 'estimator', est))
    # compute
    o = res.compute()
    assert isinstance(o, Averaged)
    assert hasattr(o.estimator, 'coef_')
    assert not hasattr(res.estimator, 'coef_')
    assert o.estimator is not est
    o2 = compute(res)[0]
    assert o is not o2
