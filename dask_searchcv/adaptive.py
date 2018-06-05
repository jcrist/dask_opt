import math
import numpy as np
import scipy.stats as stats
from functools import partial
from toolz import reduce
import toolz
import warnings
from timeit import default_timer
import logging

import sklearn
from dask_ml.model_selection._split import train_test_split

from sklearn import clone
from sklearn.model_selection import ParameterSampler

import dask.array as da
from .model_selection import DaskBaseSearchCV, GridSearchCV, _RETURN_TRAIN_SCORE_DEFAULT

logger = logging.getLogger(__name__)


def _train(model, X, y, X_val, y_val, max_iter=1, dry_run=False,
           s=1, i=1, k='_', **fit_kwargs):
    """
    Train function. Returns validation score.

    Parameters
    ----------
    model : sklearn model to train
    X, y, X_val, y_val : dask arrays that are training and validation sets
    max_iter : float, number of times to call partial_fit
    dry_run : whether to call partial_fit and score or not
    s, i, k : int or string, used for logging
    fit_kwargs : dict, kwargs to pass to partial_fit

    Takes max_iters as parameters. THese are treated as a "scarce resource" --
    Hyperband is trying to minimize the total iters use for max_iters.

    Returns
    -------
    val_score : float. Validation score
    times : dict. keys of ``fit_time`` and ``score_time``, values in seconds.

    """
    start_time = default_timer()
    for iter in range(int(max_iter)):
        msg = ("Training model {k} in bracket s={s} iteration {i}. "
               "This model is {percent:.1f}% trained for this bracket")
        logger.info(msg.format(iter=iter, k=k, s=s, i=i,
                               percent=iter * 100.0 / max_iter))
        if not dry_run:
            _ = model.partial_fit(X, y, **fit_kwargs)
    fit_time = default_timer() - start_time
    msg = ("Training model {k} for {max_iter} partial_fit calls took {secs} seconds")
    logger.info(msg.format(k=k, max_iter=max_iter, secs=fit_time))

    start_time = default_timer()
    score = model.score(X_val, y_val) if not dry_run else np.random.rand()
    score_time = default_timer() - start_time
    return score, {'score_time': score_time, 'fit_time': fit_time}


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
    return [{k: v if not isinstance(v, np.ndarray) else da.from_array(v)
            for k, v in config.items()} for config in configs]


def _successive_halving(params, model, n=None, r=None, s=None,
                        shared=None, eta=3, _prefix='', dry_run=False,
                        **fit_kwargs):
    """
    """
    client = _get_client()

    data = client.get_dataset('_dask_data')
    best = {k: shared[k] for k in ['val_score', 'model', 'config']}

    if any(x is None for x in [n, r, s]):
        raise ValueError('n, r, and s are required')
    ids = [_prefix + '-' + str(i) for i in range(n)]
    params = _random_choice(params, n=len(ids))
    params = {k: param for k, param in zip(ids, params)}
    models = {}
    final_val_scores = {k: -np.inf for k in ids}
    for k, param in params.items():
        models[k] = clone(model).set_params(**param)

    history = []
    iters = 0
    times = []
    for i in range(s + 1):
        n_i = math.floor(n * eta**-i)
        r_i = r * eta**i
        iters += r_i
        msg = ('Training {n} models for {r} iterations during iteration {i} '
               'of bracket={s}')
        logger.info(msg.format(n=n_i, r=r_i, i=i, s=s))

        futures = {k: client.submit(_train, model, *data['train'], *data['val'],
                                    max_iter=r_i, s=s, i=i,
                                    k=k, dry_run=dry_run,
                                    **fit_kwargs)
                      for k, model in models.items()}
        results = client.gather(futures)

        val_scores = {k: r[0] for k, r in results.items()}
        times += [{'id': k, **r[1]} for k, r in results.items()]

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
            #  logger.info("New best val_score={} found".format(best_['val_score']))

        if len(models) == 1:
            break

    times = toolz.groupby('id', times)
    times = {k: [toolz.keyfilter(lambda key: key != 'id', vi) for vi in v]
             for k, v in times.items()}
    times = {k: toolz.merge_with(sum, v) for k, v in times.items()}
    return {'history': history, 'final_val_scores': final_val_scores,
            'params': params, 'models': models, 'times': times}


class Hyperband(DaskBaseSearchCV):
    """
    An adaptive model selection method that trains models with
    ``model.partial_fit`` and evaluates models with ``model.score``.

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

    Methods
    -------
    fit(X, y, dry_run=False, **fit_kwargs)
        Find the best classifier.
    info()
        Get info about finding the best classifier (i.e., how many calls to
        ``partial_fit``?).


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

    References
    ----------
    - Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization, 2016.
      Lisha Li, Kevin Jamieson, Giulia DeSalvo, Afshin Rostamizadeh, Ameet Talwalkar
      https://arxiv.org/abs/1603.06560

    """
    def __init__(self, model, params, max_iter=81, eta=3,
                 n_jobs=-1, **kwargs):
        """
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
        n_jobs : int, optional
            Hyperband depends on another function, and they are completely
            indepdenent of one another and can be run embarassingly parallel.
            This function is submitted to the scheduler if ``n_jobs == -1``
            (which is the default behavior).
        """
        self.params = params
        self.model = model
        self.R = max_iter
        self.eta = eta
        self.best_val_score = -np.inf
        self.n_iter = 0

        if n_jobs not in {-1, 0}:
            raise ValueError('n_jobs has to be either -1 or 0 to run in '
                             'in parallel or serially respectively')
        self._run_in_parallel = (n_jobs == -1)

        if not hasattr(model, 'partial_fit'):
            raise ValueError('Hyperband relies on partial_fit. Without it '
                             'it would be no different than RandomizedSearchCV')
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
            if k in kwargs and kwargs[k] != super_default[k]:
                warnings.warn('Hyperband ignores the keyword argument {k}.')
        super(Hyperband, self).__init__(model, **kwargs)

    def fit(self, X, y, dry_run=False, **fit_kwargs):
        """
        This function requires an active dask.distribtued client as it uses
        the dask.distributed's concurrent.futures API.

        This function implicitly assumes that higher scores are better. Note:
        this matters if you are using a scoring function that measures loss,
        where lower values imply a better model.

        This function works with classes with a certain API, not only
        scikit-learn estimators. It is assumed that the class

        * implements ``partial_fit(self, X, y, **kwargs)``.
        * implements ``set_params(self, **kwargs) -> self``
        * implements ``score(self, X, y) -> number``
        * supports ``sklearn.base.clone(self, safe=True)``. Support for this
          function comes down to having an function ``get_params(self)``.

        """
        import distributed
        client = _get_client()

        variables = {'val_score': -np.inf, 'model': None, 'config': None,
                     'eta': self.eta, 'history': []}
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
                        'shared': shared, '_prefix': f's={s}'}]

        if self._run_in_parallel:
            futures = [client.submit(_successive_halving, self.params,
                                     self.model, **kwarg, **fit_kwargs)
                       for kwarg in kwargs]
            results = client.gather(futures)
        else:
            results = [_successive_halving(self.params, self.model, **kwarg,
                                           **fit_kwargs)
                       for kwarg in kwargs]

        all_keys = reduce(lambda x, y: x + y, [list(r['params'].keys()) for r in results])

        history = sum([r['history'] for r in results], [])
        models = toolz.merge([r['models'] for r in results])
        params = toolz.merge([r['params'] for r in results])
        times = toolz.merge([r['times'] for r in results])
        val_scores = toolz.merge([r['final_val_scores'] for r in results])

        assert set(all_keys) == set(val_scores.keys())
        assert set(all_keys) == set(params.keys())

        cv_results_, best_idx = _get_cv_results(params=params, val_scores=val_scores,
                                                times=times)

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
            * num_cv_splits : number of cross validation splits
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
        self.fit(X, y, dry_run=True)
        df = pd.DataFrame(self.history)

        brackets = df[['s', 'i', 'iters', 'num_models']]
        values = {(s, i, iter, n) for _, (s, i, iter, n) in brackets.iterrows()}
        values = sorted(values)
        brackets = [{'bracket': v[0], 'bracket_iter': v[1],
                     'partial_fit_iters': v[2], 'num_models': v[3]}
                    for v in values]

        return {'num_models': len(df.model_id.unique()),
                'num_partial_fit_calls': df.iters.sum(),
                'num_cv_splits': 1,
                'brackets': brackets}


def _LDtoDL(ld):
    """list of dicts to dict of lists"""
    keys = set(sum([list(d.keys()) for d in ld], []))
    values = {k: [d.get(k, None) for d in ld] for k in keys}
    return values


def _get_cv_results(params=None, val_scores=None, times=None):
    assert all([isinstance(x, dict) for x in [params, val_scores, times]])
    assert set(params.keys()) == set(val_scores.keys()) == set(times.keys())

    cv_results = []
    best = {'index': None, 'score': -np.inf}
    for i, k in enumerate(val_scores.keys()):
        result = {'params': params[k], 'test_score': val_scores[k], **times[k]}
        result.update({'param_' + param: v for param, v in params[k].items()})
        cv_results += [result]

        if result['test_score'] > best['score']:
            best = {'index': i, 'score': result['test_score']}

    cv_results = _LDtoDL(cv_results)
    return cv_results, best['index']


def _get_best_model(val_scores, models):
    scores = {k: val_scores[k] for k in models}
    best_id = max(scores, key=scores.get)
    return models[best_id]


def _get_client():
    import distributed
    try:
        return distributed.get_client()
    except ValueError:
        raise ValueError('No global distributed client found with '
                         'distributed.get_client. To resolve this error, '
                         'have an distributed client: '
                         'https://distributed.readthedocs.io/en/latest/client.html')
