import logging

import numpy as np
import pytest
from dask.base import tokenize
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from dask_searchcv._normalize import normalize_estimator
from dask_searchcv.methods import feature_union_concat
from dask_searchcv.online_model_selection import do_fit, do_pipeline, \
    do_fit_transform, flesh_out_params, update_dsk, _normalize_scheduler
from dask_searchcv.utils import to_keys

loglevel = pytest.config.getoption("--log", 'INFO')

log = logging.getLogger(__name__)
logging.basicConfig(level=getattr(logging, loglevel))


def test_semi_mutable_mapping():
    d = {}
    update_dsk(d, 'a', 1)

    with pytest.raises(ValueError) as exc_info:
        update_dsk(d, 'a', 2)
        assert str(exc_info) == d._exception_string.format(d=d, k='a', vold=d['a'],
                                                           vnew=2)

    update_dsk(d, 'b', 2)

    assert set(d.items()) == {('b', 2), ('a', 1)}


class Thing1(BaseEstimator):
    def __init__(self, a=1, b=0):
        self.a = a
        self.b = b

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):
        return X


class Thing2(BaseEstimator):
    def __init__(self, r=None, s=1):
        self.r = r
        self.s = s

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):
        return X


num_cmp_cols = [
    'mean_test_score', 'mean_train_score', 'split0_test_score',
    'split0_train_score', 'split1_test_score', 'split1_train_score',
    'split2_test_score', 'split2_train_score', 'std_test_score',
    'std_train_score'
]


def estimator_name(estimator):
    return type(estimator).__name__.__lower__() + '-' + tokenize(
        normalize_estimator(estimator))


def test_do_fit():
    def check_val_and_graph(k1, k2, g1, g2):
        assert (k1 == k2), "Keys don't match!"
        assert (len(g1) == len(g2)), "Graphs don't match!"

    dsk = {}

    est = Thing1()
    X, y = np.ones((2, 2), dtype=np.float64), np.ones(2, dtype=np.float64)
    params = {}
    params = flesh_out_params(est, params)
    X_name, y_name = to_keys(dsk, X, y)

    fit_name = do_fit(dsk, est, X_name, y_name, params, fit_params={},
                      error_score='raise')
    g1 = dict(dsk).copy()

    # new estimator initialised with the same arguments should not change the graph:
    est2 = Thing1()
    fit_name2 = do_fit(dsk, est2, X_name, y_name, params, fit_params={},
                       error_score='raise')
    g2 = dict(dsk).copy()

    check_val_and_graph(fit_name, fit_name2, g1, g2)

    # a new fit with parameters that are the same as the estimator parameters should
    # not change the graph:
    params_same_as_default = {'a': 1, 'b': 0}
    fit_name3 = do_fit(dsk, est2, X_name, y_name, params_same_as_default,
                       fit_params={}, error_score='raise')
    g3 = dict(dsk).copy()

    check_val_and_graph(fit_name, fit_name3, g1, g3)

    # note: if parameters are wrong type (float vs int) a new key is constructed:
    params_same_as_default = {'a': 1., 'b': 0.}
    fit_name4 = do_fit(dsk, est2, X_name, y_name, params_same_as_default,
                       fit_params={}, error_score='raise')
    g4 = dict(dsk).copy()

    with pytest.raises(AssertionError):
        check_val_and_graph(fit_name, fit_name4, g1, g4)


def test_do_fit_transform():
    dsk = {}
    est = Thing1()
    X, y = np.ones((2, 2), dtype=np.float64), np.ones(2, dtype=np.float64)
    params = {}
    params = flesh_out_params(est, params)
    X_name, y_name = to_keys(dsk, X, y)
    fit_name, Xt_name = do_fit_transform(dsk, est, X_name, y_name, params,
                                         fit_params={}, error_score='raise')

    assert any('-fit-' in k for k in dsk)
    assert any('-fit-transform-' in k for k in dsk)
    assert len(dsk) == 6
    assert Xt_name.startswith('thing1-transform-')


def test_do_pipeline():
    pipeline = Pipeline([
        ('step1', Thing1(b=2)),
        ('step2', Thing2(r=1.4))
    ])

    dsk = {}
    X, y = np.ones((2, 2), dtype=np.float64), np.ones(2, dtype=np.float64)

    params = {
        'step1__a': 1,
        'step2__r': 2.5,
    }

    params = flesh_out_params(pipeline, params)
    X_name, y_name = to_keys(dsk, X, y)
    fit_name = do_pipeline(dsk, pipeline, X_name, y_name, params, fit_params={},
                           error_score='raise')
    assert isinstance(dsk, dict)
    log.info(fit_name)
    assert fit_name.startswith('pipeline-')

    with pytest.raises(ValueError) as exc_info:
        fit_name, Xt_name = do_pipeline(dsk, pipeline, X_name, y_name, params,
                                        fit_params={}, error_score='raise',
                                        transform=True)
        log.info(
            "We don't support changing the structure of the graph so pipeline "
            "fit->transform")
        log.info(str(exc_info))


def test_feature_union_concat_all_empty():
    tr_Xs = [None, None]
    n_samples_name = 4
    w = [None, None]
    scheduler = _normalize_scheduler(None, 1)
    concat_name = 'testing'
    dsk = {}
    dsk[concat_name] = (feature_union_concat, tr_Xs, n_samples_name, w)
    val = scheduler(dsk, concat_name)
    assert len(val) == 4  # ! not 0
    assert isinstance(val, np.ndarray)
