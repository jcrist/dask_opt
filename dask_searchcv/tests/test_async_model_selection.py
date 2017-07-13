from collections import defaultdict

from dask.base import tokenize
from distributed import Client
from distributed.utils_test import cluster, loop  # noqa: F401
from sklearn.datasets import load_digits
from sklearn.pipeline import Pipeline, FeatureUnion

from dask_searchcv import GridSearchCV
from dask_searchcv.async_model_selection import AsyncGridSearchCV, CachingPlugin
from dask_searchcv.utils_test import MockClassifier, ScalingTransformer
import pandas as pd
import numpy as np
import pytest

import logging

log = logging.getLogger(__name__)


@pytest.mark.parametrize(  # noqa: F811
    'model,param_grid',
    [
        (Pipeline(steps=[('clf', MockClassifier())]), {'clf__foo_param': [2]}),
        (Pipeline(steps=[
            ('clf1', MockClassifier()),
            ('clf2', MockClassifier()),
        ]), {'clf1__foo_param': [0, 1, 2, 3]}),
        (Pipeline(steps=[('clf1', FeatureUnion(transformer_list=[
                      ('scale1', ScalingTransformer())
            ])),
            ('clf2', MockClassifier()),
        ]), {'clf2__foo_param': [0, 1, 2, 3]})
    ]
)
def test_asyncgridsearchcv(model, param_grid, loop):
    digits = load_digits()

    n_splits = 3

    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop, set_as_default=False) as client:
            search = AsyncGridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=n_splits,
                threshold=1.1,  # hack to exhaust all parameters in the grid
                client=client
            )

            X = digits.data
            y = digits.target

            search.fit_async(X, y)
            cv_results_async = pd.DataFrame(search.cv_results_)
            # dsk_async = search.dask_graph_

    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop, set_as_default=False) as client:
            search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=n_splits,
                scheduler=client
            )

            X = digits.data
            y = digits.target

            search.fit(X, y)
            cv_results_sync = pd.DataFrame(search.cv_results_)
            # dsk_sync = search.dask_graph_

    # some manipulation required to compare the results:
    assert np.array_equal(
        cv_results_sync.assign(
            sort_token=cv_results_sync.params.apply(tokenize)).sort_values(
            'sort_token').values,
        cv_results_async.assign(
            sort_token=cv_results_async.params.apply(tokenize)).sort_values(
            'sort_token').values
    )

    # print('sync graph size', len(dsk_sync), 'async graph size', len(dsk_async))
    # assert dsk_async == dsk_sync


class MockScheduler(object):
    def __init__(self):
        self._keys = defaultdict(lambda: [])
        self._plugins = []

    def client_releases_keys(self, keys, client):
        self._keys[client] = [k for k in self._keys[client] if k not in keys]

    def client_desires_keys(self, keys, client):
        self._keys[client] = self._keys[client] + keys

    def add_plugin(self, plugin):
        self._plugins.append(plugin)


def test_cachingplugin():
    logging.basicConfig(level=logging.DEBUG)

    s = MockScheduler()

    caching_plugin = CachingPlugin(s, cache_size=40)

    s.add_plugin(caching_plugin)

    caching_plugin.transition('a', start='processing', finish='memory', nbytes=30,
                              startstops=[('compute', 0.1, 0.2)])

    assert caching_plugin.total_bytes == 30
    assert list(caching_plugin.scheduler._keys['fake-caching-client']) == ['a']

    caching_plugin.transition(
        'b', start='processing', finish='memory', nbytes=20,
        startstops=[('compute', 0.1, 0.2)]
    )

    assert caching_plugin.total_bytes == 20
    assert list(caching_plugin.scheduler._keys['fake-caching-client']) == ['b']

    # Touching keys from the client?
    # ------------------------------
    # Would like a method to affect the cost of the key from the client.
    # Cannot affect key cost from client.gather since SchedulerPlugin doens't provide
    #  that interface. Maybe try increment cost on the plugin via rpc from client.
    #
    # ... something like from local: caching_plugin.touch('b')
