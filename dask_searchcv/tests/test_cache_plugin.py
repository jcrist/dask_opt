import logging
from collections import defaultdict

from dask_searchcv.async_model_selection import CachingPlugin

log = logging.getLogger(__name__)


# todo: mock this using mock library and Scheduler
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

    # Touching keys from the client
    # -----------------------------
    # Cannot affect key cost from client.gather since SchedulerPlugin doens't provide
    #  that interface. Have to increment cost on the plugin via rpc from client.
    #
    # caching_plugin.touch('b')
    # touch increases the score

