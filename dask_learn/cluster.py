from __future__ import absolute_import, division, print_function

from .base import wrap_model
from .solvers import chained_partial_fit

from sklearn import cluster

MiniBatchKMeans = wrap_model(cluster.MiniBatchKMeans,
                             {'chained': chained_partial_fit}, 'chained')
