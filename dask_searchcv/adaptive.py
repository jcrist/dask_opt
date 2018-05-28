import math
import numpy as np
import scipy.stats as stats
from functools import partial

import sklearn
try:
    from dask_ml.model_selection import train_test_split
except:
    from sklearn.model_selection import train_test_split
from sklearn import clone

import distributed
import dask.array as da


def _train(model, X, y, X_val, y_val, max_iter=1, classes=None, **kwargs):
    """
    Train function. Returns validation loss.

    Takes max_iters as parameters. THese are treated as a "scarce resource" --
    Hyperband is trying to minimize the total iters use for max_iters.
    """
    s = kwargs.pop('s', 1)
    i = kwargs.pop('i', 1)
    for iter in range(int(max_iter)):
        print("s={s}, i={i}, iter={iter}".format(iter=iter, s=s, i=i))
        _ = model.partial_fit(X, y, classes)
        model._iters += 1

    y_hat = model.predict(X_val)
    return {'val_loss': model.loss_fn(y_hat, y_val), 'model': model}

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
        print(X.shape, y.shape)
        print(X.dtype, y.dtype)
        # TODO: remove this
        if not isinstance(X, da.Array):
            raise ValueError('Hyperband currently only accepts data as dask arrays')
        if not isinstance(y, da.Array):
            raise ValueError('Hyperband currently only accepts data as dask arrays')

        X_train, X_val, y_train, y_val = train_test_split(X, y)

        self.data = {'train': (X_train, y_train), 'val': (X_val, y_val)}
        self.history = []
        self._base_id = 0

    def _successive_halving(self, params, n=None, r=None, s=None, verbose=False,
                            shared=None, eta=3):
        client = distributed.get_client()
        data = client.get_dataset('_dask_data')
        best = {k: shared[k] for k in ['val_loss', 'model', 'config']}
        history = shared['history']

        if any(x is None for x in [n, r, s]):
            raise ValueError('n, r, and s are required')
        ids = [self._base_id + i for i in range(n)]
        self._base_id += len(ids)
        configs = {k: _random_choice(self.params) for k in ids}
        models = {}
        final_val_losses = {k: np.inf for k in ids}
        for k, config in configs.items():
            models[k] = clone(self.model).set_params(**config)
            models[k].loss_fn = sklearn.metrics.hinge_loss
            models[k]._iters = 0

        for i in range(s + 1):
            n_i = math.floor(n * eta**-i)
            r_i = r * eta**i
            if verbose:
                msg = ('Training {n} models for {r} iterations for the {i}th '
                       'iteration of the {s}th bracket')
                print(msg.format(n=n_i, r=r_i, i=i, s=s))

            futures = {k: client.submit(_train, model, *data['train'],
                                        *data['val'], max_iter=r_i,
                                        classes=shared['classes'].get(), s=s, i=i)
                          for k, model in models.items()}
            results = client.gather(futures)
            val_losses = {k: v['val_loss'] for k, v in results.items()}
            models = {k: v['model'] for k, v in results.items()}
            final_val_losses.update(val_losses)

            models = _bottom_k(models, val_losses, k=max(1, math.floor(n_i / eta)))

            best_ = {k: v.get() for k, v in best.items()}
            if min(final_val_losses.values()) < best_['val_loss']:
                min_loss_key = min(final_val_losses, key=final_val_losses.get)
                best_['val_loss'] = val_losses[min_loss_key]
                best_['config'] = configs[min_loss_key]
                #  best_['model'] = models[min_loss_key]
                [best[k].set(v) for k, v in best_.items()]
                if verbose:
                    print("New best val_loss={} found".format(best_['val_loss']))

            history_ = history.get()
            history_ += [{'s': s, 'i': i, 'val_loss': val_losses[k], 'model_id': k,
                         'iters': model._iters, 'best_val_loss': best_['val_loss'],
                         **configs[k]}
                         for k, model, in models.items()]
            history.set(history_)
            if len(models) == 1:
                break
        return True

    def fit(self, verbose=True):
        """
        Returns
        -------
        """
        client = distributed.get_client()
        R, eta = self.R, self.eta
        s_max = math.floor(math.log(self.R, self.eta))
        B = (s_max + 1) * self.R
        kwargs = []
        for s in reversed(range(s_max + 1)):
            n = math.ceil(B/R * eta**s / (s+1))
            r = R * eta ** -s
            kwargs += [{'s': s, 'n': n, 'r': r}]

        variables = {'val_loss': np.inf, 'model': None, 'config': None,
                     'classes': self.classes.tolist(), 'eta': self.eta,
                     'history': []}
        shared = {}
        for key, value in variables.items():
            shared[key] = distributed.Variable(key)
            shared[key].set(value)

        if '_dask_data' in client.list_datasets():
            client.unpublish_dataset('_dask_data')
        client.publish_dataset(_dask_data=self.data)

        #  futures = [client.submit(self._successive_halving, self.params, eta=eta,
                                 #  **kwarg, shared=shared, verbose=verbose)
                   #  for kwarg in kwargs]
        #  client.gather(futures)
        futures = [self._successive_halving(self.params, eta=eta,
                                 **kwarg, shared=shared, verbose=verbose)
                   for kwarg in kwargs]

        #  client.gather(futures)
        self.history = shared['history'].get()

        return tuple([shared[k].get() for k in ['val_loss', 'config']])
