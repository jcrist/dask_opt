from dask.base import tokenize
from distributed import Client
from distributed.utils_test import loop, cluster
from sklearn.datasets import load_digits
from sklearn.pipeline import Pipeline, FeatureUnion

from dask_searchcv import GridSearchCV
from dask_searchcv.async_model_selection import AsyncGridSearchCV
from dask_searchcv.utils_test import MockClassifier, ScalingTransformer
import pandas as pd
import numpy as np
import pytest


@pytest.mark.parametrize(
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
            dsk_async = search.dask_graph_

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
            dsk_sync = search.dask_graph_

    print(cv_results_sync)
    print(cv_results_async)

    assert np.array_equal(
        cv_results_sync.assign(
            sort_token=cv_results_sync.params.apply(tokenize)).sort_values(
            'sort_token').values,
        cv_results_async.assign(
            sort_token=cv_results_async.params.apply(tokenize)).sort_values(
            'sort_token').values
    )
    print('sync graph size', len(dsk_sync), 'async graph size', len(dsk_async))
    # assert dsk_async == dsk_sync


def test_persisted_scores_are_kept():
    pass