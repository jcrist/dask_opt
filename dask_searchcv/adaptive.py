import math
import numpy as np
import scipy.stats as stats
from functools import partial
from pprint import pprint
from toolz import reduce
import warnings

import sklearn
from dask_ml.model_selection._split import train_test_split

from sklearn import clone
from sklearn.model_selection import ParameterSampler

import distributed
import dask.array as da
from .model_selection import DaskBaseSearchCV, _RETURN_TRAIN_SCORE_DEFAULT


def _train(model, X, y, X_val, y_val, max_iter=1, dry_run=False, classes=None,
           s=1, i=1, k='_', verbose=False, **fit_kwargs):
    """
    Train function. Returns validation loss.

    Takes max_iters as parameters. THese are treated as a "scarce resource" --
    Hyperband is trying to minimize the total iters use for max_iters.
    """
    for iter in range(int(max_iter)):
        if verbose:
            msg = ("Training model {k} in bracket s={s} iteration {i}. "
                   "This model is {percent:.1f}% trained for this bracket")
            print(msg.format(iter=iter, k=k, s=s, i=i, percent=iter * 100.0 / max_iter))
        if not dry_run:
            _ = model.partial_fit(X, y, classes, **fit_kwargs)
        elif iter == 0:
            _ = model.partial_fit(X[0:2], y[0:2], classes)

    score = model.score(X_val, y_val, **fit_kwargs) if not dry_run else np.random.rand()
    return score


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
                        shared=None, eta=3, _prefix='', dry_run=False, **fit_kwargs):
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

        futures = {k: client.submit(_train, model, *data['train'], *data['val'],
                                    max_iter=r_i, classes=classes, s=s, i=i,
                                    k=k, verbose=verbose, dry_run=dry_run,
                                    **fit_kwargs)
                      for k, model in models.items()}
        val_scores = client.gather(futures)
        final_val_scores.update(val_scores)
        history += [{'s': s, 'i': i, 'val_score': val_scores[k], 'model_id': k,
                     'iters': iters, 'num_models': len(val_scores), **params[k]}
                     for k, model, in models.items()]

        models = _top_k(models, val_scores, k=max(1, math.floor(n_i / eta)))

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
    return {'history': history, 'final_val_scores': final_val_scores,
            'params': params, 'models': models}


class Hyperband(DaskBaseSearchCV):
    def __init__(self, model, params, max_iter=81, eta=3, **kwargs):
        """
        An adaptive model selection method that trains models with
        ``model.partial_fit`` and evaluates models with ``model.score``.

        Parameters
        ----------
        params :
            Search space of various parameters. This mirrors the sklearn API.

        model :
            The base model to try.

        max_iter : int
            The maximum number of iterations to train a model on, as determined
            by ``model.partial_fit``.

        X, y : np.ndarray, np.ndarray
            Train data, features ``X`` and labels ``y``.

        eta : optional, int
            How aggressive to be while search. The default is 3, and theory
            suggests that ``eta=e=2.718...``. Changing this is not recommended.

        Attributes
        ----------
        cv_results_ : list of dicts
            This is a list of all models tuned, and information about those
            models.  This list of dicts can be imported into Pandas with
            ``pd.DataFrame(self.cv_results_)``. Each dictionary contains the
            keys ``test_score``, ``params`` and ``param_{p}`` for each ``p``
            in params. That is, ``params['alpha'] == params_alpha``.

        best_params_ : dict
            The best parameters found by this model selection algorithm. This
            is determined by cross validation.

        best_estimator_ : type(model)
            The best estimator as determined by this model selection algorithm.
            This is determined by cross validation.

        best_index_ : int
            The index of the best performing estimator and parameters in
            ``best_estimator_`` and ``best_params_``

        history : list of dicts
            The recorded history for this adaptive process. It includes the
            evaluations on each step, and it's possible to use this to see each
            decision on which models to keep evaluating. Each dictionary in this
            variables has ``set(keys) == {}``

        Notes
        -----
        This implements the `Hyperband model selection algorithm`_
        and only requires one input, ``max_iter`` which is porprotional to
        "computational effort" required. Notably it does not require balancing
        a trade-off between the number of params to evaluate and how long to
        train each config.

        Hyperband has some theoritical guarantees on finding (close to) the
        optimal config from ``params`` given this computational effort
        ``max_iter``.

        .. _Hyperband model selection algorithm: https://arxiv.org/abs/1603.06560
        """
        self.params = params
        self.model = model
        self.R = max_iter
        self.eta = eta
        self.best_val_score = -np.inf
        self.n_iter = 0

        if not hasattr(model, 'warm_start'):
            warnings.warn('model has no attribute warm_start. Hyperband will assume it '
                          'is reusing previous calls to `partial_fit` during each call '
                          'to `partial_fit`')
        else:
            if not model.warm_start:
                raise ValueError('Hyperband relies on model(..., warm_start=True).')

        self.history = []

        super_default = dict(scoring=None, iid=True, refit=True, cv=None,
                             error_score='raise', return_train_score=_RETURN_TRAIN_SCORE_DEFAULT,
                             scheduler=None, n_jobs=-1, cache_cv=True)
        for k, v in super_default.items():
            if kwargs[k] != super_default[k]:
                warnings.warn('Hyperband ignores the keyword argument {k}.')
        super(Hyperband, self).__init__(model, **kwargs)

    def fit(self, X, y, dry_run=False, verbose=False, **fit_kwargs):
        """
        This implicitly assumes that higher scores are better.

        Returns
        -------
        val_score : float
            The best

        This is a general interface; ``model`` can be any class that

        * implements ``model.set_params(**kwargs) -> model``
        * implements ``model.partial_fit(X, y, classes)``.
          ``classes`` can be ``np.unique(y_total)``.
        * implements ``model.score(X, y) -> number``
        * supports ``sklearn.base.clone(model, safe=True)``. Support for this
          function comes down to having an function ``get_params``.

        """
        client = distributed.get_client()

        variables = {'val_score': -np.inf, 'model': None, 'config': None,
                     'classes': np.unique(y).tolist(), 'eta': self.eta,
                     'history': []}
        shared = {}
        for key, value in variables.items():
            shared[key] = distributed.Variable(key)
            shared[key].set(value)

        if '_dask_data' in client.list_datasets():
            client.unpublish_dataset('_dask_data')

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
        data = {'train': [X_train, y_train], 'val': [X_val, y_val]}

        for key in data.keys():
            data[key][0] = client.scatter(data[key][0])
            data[key][1] = client.scatter(data[key][1])
        client.publish_dataset(_dask_data=data)

        # now start hyperband
        R, eta = self.R, self.eta
        s_max = math.floor(math.log(self.R, self.eta))
        B = (s_max + 1) * self.R
        kwargs = []
        for s in reversed(range(s_max + 1)):
            n = math.ceil(B/R * eta**s / (s+1))
            r = R * eta ** -s
            kwargs += [{'s': s, 'n': n, 'r': r, 'dry_run': dry_run, 'eta': eta,
                        'shared': shared, '_prefix': f's={s}',
                        'verbose': verbose}]

        #  futures = [client.submit(_successive_halving, self.params,
                                 #  self.model, **kwarg)
                   #  for kwarg in kwargs]
        #  results = client.gather(futures)

        futures = [_successive_halving(self.params, self.model, **kwarg, **fit_kwargs)
                   for kwarg in kwargs]
        results = futures

        all_keys = reduce(lambda x, y: x + y, [list(r['params'].keys()) for r in results])

        history = sum([r['history'] for r in results], [])
        models = reduce(lambda x, y: {**x, **y},
                        [r['models'] for r in results])
        params = reduce(lambda x, y: {**x, **y},
                        [r['params'] for r in results])
        val_scores = reduce(lambda x, y: {**x, **y},
                            [r['final_val_scores'] for r in results])
        assert set(all_keys) == set(val_scores.keys())
        assert set(all_keys) == set(params.keys())

        cv_results_, best_idx = _get_cv_results(params=params, val_scores=val_scores)

        #  self.best_params_ = cv_results_[best_idx]['params']
        self.best_index_ = best_idx
        self.history += history
        self.best_estimator_ = _get_best_model(val_scores, models)
        self.cv_results_ = cv_results_

        return self

    def info(self):
        """
        This can be used to obtain the amount of computational effort required,
        alongside some other information (number of splits, etc).

        Returns
        -------
        info : dict
            Dictionary with keys

            * num_partial_fit_calls : total number of times partial_fit is called
            * num_models : the total number of models evaluated
            * num_splits : number of cross validation splits
            * brackets : a list of dicts, importable into pandas.DataFrame. Each value has a key of
                * bracket : the Hyperband bracket number. It runs several
                  brackets to balance the explore/exploit tradeoff (should you
                  train many models for few iterations or few models for many
                  iterations?)
                * bracket_iter : The iteration for this bracket. At each time
                  step, it "kills" 1 - 1/eta of the models.
                * partial_fit_iters: the number of times partial_fit will be
                  called for per model in this iteration in this bracket
                * num_models : the number of models that will be fit for
                  ``partial_fit_iters`` iterations
        """

        import pandas as pd
        X = da.from_array(np.random.rand(2, 2), 2)
        y = da.from_array(np.random.rand(2, 2), 2)
        self.fit(X, y, verbose=False)
        df = pd.DataFrame(self.history)

        brackets = df[['s', 'i', 'iters', 'num_models']]
        values = {(s, i, iter, n) for _, (s, i, iter, n) in brackets.iterrows()}
        values = sorted(values)
        brackets = [{'bracket': v[0], 'bracket_iter': v[1],
                     'partial_fit_iters': v[2], 'num_models': v[3]}
                    for v in values]

        return {'num_models': len(df.model_id.unique()),
                'total_iters': df.iters.sum(),
                'num_cv_splits': 1,
                'brackets': brackets}


def _LDtoDL(ld):
    """list of dicts to dict of lists"""
    keys = set(sum([list(d.keys()) for d in ld], []))
    values = {k: [d.get(k, None) for d in ld] for k in keys}
    return values


def _get_cv_results(params=None, val_scores=None):
    assert set(params.keys()) == set(val_scores.keys())

    cv_results = []
    best_idx = None
    best_score = -np.inf
    for i, k in enumerate(val_scores.keys()):
        result = {'params': params[k], 'test_score': val_scores[k]}
        result.update({'param_' + param: v for param, v in params[k].items()})
        cv_results += [result]

        if result['test_score'] > best_score:
            best_idx = i
            best_score = result['test_score']

    cv_results = _LDtoDL(cv_results)

    return cv_results, best_idx


def _get_best_model(val_scores, models):
    scores = {k: val_scores[k] for k in models}
    best_id = max(scores, key=scores.get)
    return models[best_id]
