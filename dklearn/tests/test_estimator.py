from __future__ import print_function, absolute_import, division

import pytest
import numpy as np
from dask.base import tokenize
from dask.delayed import Delayed
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression

from dklearn import from_sklearn
from dklearn.estimator import Estimator


clf1 = LogisticRegression(C=1000)
clf2 = LogisticRegression(C=5000)

iris = load_iris()
X_iris = iris.data[:, :2]
y_iris = iris.target


def test_tokenize_BaseEstimator():
    assert tokenize(clf1) == tokenize(clf1)
    assert tokenize(clf1) == tokenize(clone(clf1))
    assert tokenize(clf1) != tokenize(clf2)
    fit = clone(clf1).fit(X_iris, y_iris)
    assert tokenize(fit) == tokenize(fit)
    assert tokenize(fit) != tokenize(clf1)
    fit2 = clone(clf2).fit(X_iris, y_iris)
    assert tokenize(fit) != tokenize(fit2)


def test_Estimator_init():
    d = Estimator(clf1)
    d2 = from_sklearn(clf1)
    assert d._name == d2._name
    assert Estimator(clf1)._name == d._name
    assert Estimator(clf2)._name != d._name
    # estimators are copied instead of mutated
    assert d._base is not clf1

    with pytest.raises(TypeError):
        Estimator("not an estimator")


def test_clone():
    d = Estimator(clf1)
    d2 = clone(d)
    assert d.get_params() == d2.get_params()
    assert d._base is not d2._base


def test__estimator_type():
    d = Estimator(clf1)
    assert d._estimator_type == clf1._estimator_type


def test_get_params():
    d = Estimator(clf1)
    assert d.get_params() == clf1.get_params()
    assert d.get_params(deep=False) == clf1.get_params(deep=False)


def test_set_params():
    d = Estimator(LogisticRegression(C=1000))
    old_name = d._name
    d2 = d.set_params(C=5)
    assert d is d2
    assert d.C == 5
    assert d._name != old_name
    d.set_params(C=1000)
    assert d.C == 1000
    assert d._name == old_name


def test_setattr():
    d = Estimator(clf1)
    with pytest.raises(AttributeError):
        d.C = 10


def test_getattr():
    d = Estimator(clf1)
    assert d.C == clf1.C
    with pytest.raises(AttributeError):
        d.not_a_real_parameter


def test_dir():
    d = Estimator(clf1)
    attrs = dir(d)
    assert 'C' in attrs


def test_repr():
    d = Estimator(clf1)
    res = repr(d)
    assert res.startswith('Dask')


def test_to_sklearn():
    d = Estimator(clf1)
    res = d.to_sklearn()
    assert res is not clf1
    assert isinstance(res, LogisticRegression)

    res = d.to_sklearn(compute=False)
    assert isinstance(res, Delayed)
    assert isinstance(res.compute(), LogisticRegression)

    # After fitting
    d.fit(X_iris, y_iris)
    res = d.to_sklearn()
    assert isinstance(res, LogisticRegression)

    res = d.to_sklearn(compute=False)
    assert isinstance(res, Delayed)
    assert isinstance(res.compute(), LogisticRegression)


def test_fit():
    d = Estimator(clf1)
    fit = d.fit(X_iris, y_iris)
    assert fit is d
    assert not hasattr(d, 'coef_')

    res = d.compute()
    assert res is not d
    assert isinstance(res, Estimator)
    assert hasattr(res, 'coef_')
    assert not hasattr(d, 'coef_')


def test_predict():
    d = Estimator(clf1)
    d.fit(X_iris, y_iris)
    pred = d.predict(X_iris)
    assert isinstance(pred, Delayed)
    res = pred.compute()
    assert isinstance(res, np.ndarray)

    will_error = from_sklearn(clf1).predict(X_iris)
    with pytest.raises(NotFittedError):
        will_error.compute()


def test_score():
    d = Estimator(clf1)
    d.fit(X_iris, y_iris)
    s = d.score(X_iris, y_iris)
    assert isinstance(s, Delayed)
    res = s.compute()
    assert isinstance(res, float)

    will_error = from_sklearn(clf1).score(X_iris, y_iris)
    with pytest.raises(NotFittedError):
        will_error.compute()
