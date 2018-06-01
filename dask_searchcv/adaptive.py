import math
import numpy as np
import scipy.stats as stats
from functools import partial
from pprint import pprint

import sklearn
try:
    from dask_ml.model_selection import train_test_split
except:
    from sklearn.model_selection import train_test_split
from sklearn import clone
from sklearn.model_selection import ParameterSampler

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
    k = kwargs.pop('k', '_')
    verbose = kwargs.pop('verbose', False)
    for iter in range(int(max_iter)):
        if verbose:
            msg = ("Training model {k} in bracket s={s} iteration {i}. "
                   "This model is {percent:.1f}% trained for this bracket")
            print(msg.format(iter=iter, k=k, s=s, i=i, percent=iter * 100.0 / max_iter))
        _ = model.partial_fit(X, y, classes)

    return model.score(X_val, y_val)


def _top_k(choose_from, eval_with, k=1):
    assert k > 0, "k must be positive"
    assert set(choose_from.keys()) == set(eval_with.keys())

    keys = list(choose_from.keys())
    assert k <= len(keys)

    eval_with = [eval_with[key] for key in keys]
    idx = np.argsort(eval_with)

    return {keys[i]: choose_from[keys[i]] for i in idx[-k:]}


def _random_choice(params, n=1):
    configs = ParameterSampler(params, n)
    return [{k: v[0] if isinstance(v, np.ndarray) and v.ndim == 1 else v
             for k, v in config.items()} for config in configs]


def _successive_halving(params, model, n=None, r=None, s=None, verbose=False,
                        shared=None, eta=3, _prefix=''):
    client = distributed.get_client()
    data = client.get_dataset('_dask_data')
    best = {k: shared[k] for k in ['val_score', 'model', 'config']}
    classes = shared['classes'].get()

    if any(x is None for x in [n, r, s]):
        raise ValueError('n, r, and s are required')
    ids = [_prefix + '-' + str(i) for i in range(n)]
    params = _random_choice(params, n=len(ids))
    params = {k: param for k, param in zip(ids, params)}
    models = {}
    final_val_scores = {k: -np.inf for k in ids}
    for k, param in params.items():
        models[k] = clone(model).set_params(**param)

    # TODO: manage memory with below
    #  models = client.scatter(models)

    history = []
    iters = 0
    for i in range(s + 1):
        n_i = math.floor(n * eta**-i)
        r_i = r * eta**i
        iters += r_i
        if verbose:
            msg = ('Training {n} models for {r} iterations during iteration {i} '
                   'of bracket={s}')
            print(msg.format(n=n_i, r=r_i, i=i, s=s))

        futures = {k: client.submit(_train, model, *data['train'],
                                    *data['val'], max_iter=r_i, classes=classes,
                                    s=s, i=i, k=k, verbose=verbose)
                      for k, model in models.items()}
        val_scores = client.gather(futures)
        final_val_scores.update(val_scores)
        history += [{'s': s, 'i': i, 'val_score': val_scores[k], 'model_id': k,
                     'iters': iters, **params[k]}
                     for k, model, in models.items()]

        models = _top_k(models, val_scores, k=max(1, math.floor(n_i / eta)))
        d = {k: val_scores[k] for k in models}

        best_ = {k: v.get() for k, v in best.items()}
        if max(final_val_scores.values()) > best_['val_score']:
            max_loss_key = max(final_val_scores, key=final_val_scores.get)
            best_['val_score'] = final_val_scores[max_loss_key]
            best_['config'] = params[max_loss_key]
            [best[k].set(v) for k, v in best_.items()]
            if verbose:
                print("New best val_score={} found".format(best_['val_score']))

        if len(models) == 1:
            break
    return history

class Hyperband:
    def __init__(self, params, model, iters, X, y, eta=3):
        """
        An adaptive model selection method that trains models with
        ``model.partial_fit`` and evaluates models with ``model.score``.

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
        a trade-off between the number of params to evaluate and how long to
        train each config.

        Hyperband has some theoritical guarantees on finding (close to) the
        optimal config from ``params`` given this computational effort ``iters``.

        This class implements an ``info`` method which can be used to obtain
        the amount of computational effort required, alongside some other
        information (number of splits, etc).

        This is a general interface; ``model`` can be any class that

        * implements ``model.set_params(**kwargs) -> model``
        * implements ``model.partial_fit(X, y, classes)``.
          ``classes`` can be ``np.unique(y_total)``.
        * implements ``model.score(X, y) -> number``
        * supports ``sklearn.base.clone(model, safe=True)``. Support for this
          function comes down to having an function ``get_params``.

        .. _Hyperband model selection algorithm: https://arxiv.org/abs/1603.06560
        """
        self.params = params
        self.model = model
        self.R = iters
        self.eta = eta
        self.best_val_score = -np.inf
        self.n_iter = 0
        self.classes = np.unique(y)

        n, d = X.shape
        # TODO: remove this
        if not isinstance(X, da.Array):
            raise ValueError('Hyperband currently only accepts data as dask arrays')
        if not isinstance(y, da.Array):
            raise ValueError('Hyperband currently only accepts data as dask arrays')

        X_train, X_val, y_train, y_val = train_test_split(X, y)

        self.data = {'train': [X_train, y_train], 'val': [X_val, y_val]}
        self.history = []
        self._base_id = 0


    def fit(self, verbose=True):
        """
        Returns
        -------
        """
        client = distributed.get_client()

        variables = {'val_score': -np.inf, 'model': None, 'config': None,
                     'classes': self.classes.tolist(), 'eta': self.eta,
                     'history': []}
        shared = {}
        for key, value in variables.items():
            shared[key] = distributed.Variable(key)
            shared[key].set(value)

        if '_dask_data' in client.list_datasets():
            client.unpublish_dataset('_dask_data')
        for key in self.data.keys():
            self.data[key][0] = client.scatter(self.data[key][0])
            self.data[key][1] = client.scatter(self.data[key][1])
        client.publish_dataset(_dask_data=self.data)

        # now start hyperband
        R, eta = self.R, self.eta
        s_max = math.floor(math.log(self.R, self.eta))
        B = (s_max + 1) * self.R
        kwargs = []
        for s in reversed(range(s_max + 1)):
            n = math.ceil(B/R * eta**s / (s+1))
            r = R * eta ** -s
            kwargs += [{'s': s, 'n': n, 'r': r, 'shared': shared,
                        '_prefix': f's={s}', 'eta': eta, 'verbose': verbose}]

        #  futures = [client.submit(_successive_halving, self.params,
                                 #  self.model, **kwarg)
                   #  for kwarg in kwargs]
        #  history = sum(client.gather(futures), [])

        futures = [_successive_halving(self.params, self.model, **kwarg)
                   for kwarg in kwargs]
        history = sum(futures, [])

        self.history = history

        return tuple([shared[k].get() for k in ['val_score', 'config']])
