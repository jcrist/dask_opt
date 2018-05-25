import math
import numpy as np
import scipy.stats as stats

import sklearn
from sklearn.model_selection import train_test_split
from sklearn import clone


def train(model, X, y, X_val, y_val, max_iter=1, classes=None):
    for iter in range(int(max_iter)):
        _ = model.partial_fit(X, y, classes)
        model._iters += 1
    return model

def predict_and_loss(model, X, y):
    y_hat = model.predict(X)
    return model.loss_fn(y_hat, y)

def bottom_k(choose_from, eval_with, k=1):
    assert k > 0, "k must be positive"
    assert set(choose_from.keys()) == set(eval_with.keys())

    keys = list(choose_from.keys())

    eval_with = [eval_with[key] for key in keys]
    idx = np.argsort(eval_with)

    return {keys[i]: choose_from[keys[i]] for i in idx[:k]}


def random_choice(params):
    choice = {}
    for k, v in params.items():
        dist = stats.uniform(v['min'], v['max'] - v['min'])
        r = dist.rvs()
        if v.get('type', 'linear') == 'log':
            value = 10**r
        choice[k] = value
    return choice


class Hyperband:
    def __init__(self, params, model, R, X, y, eta=3):
        """
        data : dict
            keys of ['train', 'test'] with values suitable for model.fit
        """
        self.params = params
        self.model = model
        self.R = R
        self.eta = eta
        self.best_val_loss = np.inf
        self.n_iter = 0
        self.classes = np.unique(y)

        n, d = X.shape
        train, val = train_test_split(range(n))
        self.data = {'train': (X[train], y[train]), 'val': (X[val], y[val])}
        self.history = []
        self._base_id = 0

    def _successive_halving(self, params, n, r, s):
        ids = [self._base_id + i for i in range(n)]
        self._base_id += len(ids)
        configs = {k: random_choice(self.params) for k in ids}
        models = {}
        for k, config in configs.items():
            models[k] = clone(self.model).set_params(**config)
            models[k].loss_fn = sklearn.metrics.hinge_loss
            models[k]._iters = 0

        self.out_data = []
        eta = self.eta
        for i in range(s + 1):
            n_i = math.floor(n * eta**-i)
            r_i = r * eta**i
            #  train_iters = r * (eta**i - eta**(i-1))

            models = {k: train(model, *self.data['train'], *self.data['val'],
                               max_iter=r_i, classes=self.classes)
                      for k, model in models.items()}
            val_losses = {k: predict_and_loss(model, *self.data['val'])
                          for k, model in models.items()}
            self.history += [{'s': s, 'i': i, 'val_loss': val_losses[k], 'model_id': k,
                              'iters': model._iters, **configs[k]}
                             for k, model, in models.items()]
            #  iters = [model._iters for model in models.values()]

            if len(models) == 1:
                break
            models = bottom_k(models, val_losses, k=max(1, math.floor(n_i / eta)))

            if min(val_losses.values()) < self.best_val_loss:
                min_loss_key = min(val_losses, key=val_losses.get)
                self.best_val_loss = val_losses[min_loss_key]
                self.best_config = configs[min_loss_key]
            self.out_data += [{'best_loss': self.best_val_loss,
                               'best_config': self.best_config,
                               'n_iters': self.n_iter}]

    def hyperband(self):
        R, eta = self.R, self.eta
        s_max = math.floor(math.log(self.R, self.eta))
        B = (s_max + 1) * self.R
        for s in reversed(range(s_max + 1)):
            n = math.ceil(B/R * eta**s / (s+1))
            r = R * eta ** -s
            self._successive_halving(self.params, n, r, s=s)
