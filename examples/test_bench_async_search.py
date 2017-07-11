import numpy as np
import pandas as pd

from examples.bench_async_search import get_stats_for_eventstream, \
    get_duration_for_eventstream


def test_get_stats_for_eventstream():
    bf_async = pd.read_pickle('./async_bf_l6.pkl')

    r_async = get_stats_for_eventstream(bf_async)
    mean_occupancy = r_async.loc['mean_occupancy'].mean()
    mean_vacancy_rate = r_async.loc['mean_vacancy'].mean()

    assert np.isclose(mean_occupancy, 1.2040435)
    assert np.isclose(mean_vacancy_rate, 0.241180)


def test_get_duration_for_eventstream():
    bf_async = pd.read_pickle('./async_bf_l6.pkl')
    dur = get_duration_for_eventstream(bf_async)

    assert np.isclose(dur, 269.53993463)
