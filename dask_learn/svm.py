from __future__ import absolute_import, division, print_function

from .base import wrap_model
from .solvers import average_fit

from sklearn import svm

LinearSVC = wrap_model(svm.LinearSVC, {'avg': average_fit}, 'avg')
