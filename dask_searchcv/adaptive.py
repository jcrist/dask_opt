import math
import numpy as np
import scipy.stats as stats
from functools import partial

import sklearn
from sklearn.model_selection import train_test_split
from sklearn import clone


def _train(model, X, y, X_val, y_val, max_iter=1, classes=None):
    for iter in range(int(max_iter)):
        _ = model.partial_fit(X, y, classes)
        model._iters += 1
    return model

def _predict_and_loss(model, X, y):
    y_hat = model.predict(X)
    return model.loss_fn(y_hat, y)

def _bottom_k(choose_from, eval_with, k=1):
    assert k > 0, "k must be positive"
    assert set(choose_from.keys()) == set(eval_with.keys())

    keys = list(choose_from.keys())

    eval_with = [eval_with[key] for key in keys]
    idx = np.argsort(eval_with)

    return {keys[i]: choose_from[keys[i]] for i in idx[:k]}


def _random_choice(params):
    choice = {}
    for k, v in params.items():
        dist = stats.uniform(v['min'], v['max'] - v['min'])
        r = dist.rvs()
        if v.get('type', 'linear') == 'log':
            value = 10**r
        choice[k] = value
    return choice


class Hyperband:
    def __init__(self, params, model, iters, X, y, eta=3):
        """
        Parameters
        ----------
        params :
            Search space of various parameters. This mirrors the sklearn API.
        model :
            The base model to try.
        iters : int
            The maximum number of iterations to train a model on, as determined
            by ``model.partial_fit``.
        X, y : np.ndarray, np.ndarray
            Train data, features ``X`` and labels ``y``.
        eta : optional, int
            How aggressive to be while search. The default is 3, and theory
            suggests that ``eta=e=2.718...``. Changing this is not recommended.

        Notes
        -----
        This implements the `Hyperband model selection algorithm`_
        and only requires one input, ``iters`` which is porprotional to
        "computational effort" required. Notably it does not require balancing
        a trade-off between the number of configs to evaluate and how long to
        train each config.

        Hyperband has some theoritical guarantees on finding (close to) the
        optimal config from ``params`` given this computational effort ``iters``.

        This class implements an ``info`` method which can be used to obtain
        the amount of computational effort required, alongside some other
        information (number of splits, etc).

        .. _Hyperband model selection algorithm: https://arxiv.org/abs/1603.06560
        """
        self.params = params
        self.model = model
        self.R = iters
        self.eta = eta
        self.best_val_loss = np.inf
        self.n_iter = 0
        self.classes = np.unique(y)

        n, d = X.shape
        train, val = train_test_split(range(n))
        self.data = {'train': (X[train], y[train]), 'val': (X[val], y[val])}
        self.history = []
        self._base_id = 0

    def _successive_halving(self, params, n=None, r=None, s=None, verbose=False):
        if any(x is None for x in [n, r, s]):
            raise ValueError('n, r, and s are required')
        ids = [self._base_id + i for i in range(n)]
        self._base_id += len(ids)
        configs = {k: _random_choice(self.params) for k in ids}
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
            if verbose:
                print('Training {n} models for {r} iterations each'.format(n=n_i, r=r_i))

            models = {k: _train(model, *self.data['train'], *self.data['val'],
                               max_iter=r_i, classes=self.classes)
                      for k, model in models.items()}
            val_losses = {k: _predict_and_loss(model, *self.data['val'])
                          for k, model in models.items()}

            if len(models) == 1:
                break
            models = _bottom_k(models, val_losses, k=max(1, math.floor(n_i / eta)))

            if min(val_losses.values()) < self.best_val_loss:
                min_loss_key = min(val_losses, key=val_losses.get)
                self.best_val_loss = val_losses[min_loss_key]
                self.best_config = configs[min_loss_key]
                self.best_model = models[min_loss_key]
                if verbose:
                    print("New best val_loss={val_loss} found".format(val_loss=self.best_val_loss))
            self.history += [{'s': s, 'i': i, 'val_loss': val_losses[k], 'model_id': k,
                              'iters': model._iters, 'best_val_loss': self.best_val_loss,
                              **configs[k]}
                             for k, model, in models.items()]
        return True

    def fit(self, verbose=True):
        """
        Returns
        -------
        """
        R, eta = self.R, self.eta
        s_max = math.floor(math.log(self.R, self.eta))
        B = (s_max + 1) * self.R
        kwargs = []
        for s in reversed(range(s_max + 1)):
            n = math.ceil(B/R * eta**s / (s+1))
            r = R * eta ** -s
            kwargs += [{'s': s, 'n': n, 'r': r}]

        for kwarg in kwargs:
            self._successive_halving(self.params, **kwarg, verbose=verbose)


        return self.best_config, self.best_model
