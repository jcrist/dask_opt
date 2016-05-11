from __future__ import absolute_import, print_function, division

from sklearn import linear_model

from .base import wrap_model
from .solvers import averaged_fit, chained_partial_fit, logistic_gradient


LogisticRegression = wrap_model(linear_model.LogisticRegression,
                                {'avg': averaged_fit,
                                 'gradient': logistic_gradient}, 'avg')


wrap = lambda mdl: wrap_model(mdl, {'avg': averaged_fit,
                                    'chained': chained_partial_fit}, 'avg')

SGDClassifier = wrap(linear_model.SGDClassifier)
SGDRegressor = wrap(linear_model.SGDRegressor)
Perceptron = wrap(linear_model.Perceptron)
PassiveAggressiveRegressor = wrap(linear_model.PassiveAggressiveRegressor)
PassiveAggressiveClassifier = wrap(linear_model.PassiveAggressiveClassifier)
