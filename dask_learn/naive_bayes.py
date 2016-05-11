from __future__ import absolute_import, division, print_function

from .base import wrap_model
from .solvers import chained_partial_fit

from sklearn import naive_bayes

solvers = {'chained': chained_partial_fit}
MultinomialNB = wrap_model(naive_bayes.MultinomialNB, solvers, 'chained')
BernoulliNB = wrap_model(naive_bayes.BernoulliNB, solvers, 'chained')
