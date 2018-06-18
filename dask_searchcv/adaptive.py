import sys
from timeit import default_timer
import logging
from warnings import warn
import math

import numpy as np
from sklearn import clone
from sklearn.model_selection import ParameterSampler
import dask
import dask.array as da
import toolz

from dask_ml.model_selection._split import train_test_split
from dask_ml.wrappers import ParallelPostFit
from dask_ml.metrics import get_scorer
import sklearn.metrics
import dask_ml.metrics
from dask_ml.wrappers import Incremental

from .model_selection import DaskBaseSearchCV, _RETURN_TRAIN_SCORE_DEFAULT


logger = logging.getLogger(__name__)


def _train(model, data, max_iter=1, dry_run=False, scorer=None,
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

    Takes max_iters as parameters, which are treated as a "scarce resource" --
    Hyperband is trying to minimize the total iters use for max_iters.

    Returns
    -------
    val_score : float. Validation score
    times : dict. keys of ``fit_time`` and ``score_time``, values in seconds.

    """
    for k, v in data.items():
        for i, vi in enumerate(v):
            if isinstance(vi, np.ndarray):
                data[k][i] = da.from_array(vi, chunks=10)
    X, y = data['train']
    X_val, y_val = data['val']
    start_time = default_timer()

    for iter in range(int(max_iter)):
        msg = ("Training model %s in bracket %d iteration %d. "
               "This model is %0.2f trained for this bracket")
        logger.info(msg, k, s, i, iter * 100.0 / max_iter)
        if not dry_run:
            model.partial_fit(X, y, **fit_kwargs)
    fit_time = default_timer() - start_time
    msg = ("Training model %s for %d partial_fit calls took %0.2f seconds")
    logger.info(msg, k, max_iter, fit_time)

    start_time = default_timer()
    score = scorer(model, X_val, y_val) if not dry_run else np.random.rand()
    score_time = default_timer() - start_time
    msg = "Fitting model %s took %0.3f seconds and scoring took %0.3f seconds"
    logger.info(msg, k, fit_time, score_time)
    return score, {'score_time': score_time, 'fit_time': fit_time}


def _top_k(choose_from, eval_with, k=1):
    """
    find the top k items from two dictionaries

    Arguments
    ---------
    choose_from : dict
        A dict of {k: object} pairs to choose. The "top k" of these objects
        will be selected. This dict must have the same keys as eval_with.
    eval_with : dict
        The scores to evaluate with.
    k : int, optional
        This function returns the top k objects as ranked by eval_with

    Returns
    -------
    top_k : dict
        A dict of {k: object}, where the objects in this dict are in the top k
        items as ranked by the scores in eval_with
    """
    assert k > 0, "k must be positive"
    assert set(choose_from.keys()) == set(eval_with.keys())

    keys = list(choose_from.keys())
    assert k <= len(keys)

    eval_with = [eval_with[key] for key in keys]
    idx = np.argsort(eval_with)

    return {keys[i]: choose_from[keys[i]] for i in idx[-k:]}


def _successive_halving(params, model, data, eta=3, n=None, r=None, s=None,
                        _prefix='', dry_run=False, scorer=None,
                        **fit_kwargs):
    """
    Perform "successive halving" on a set of models: partially fit the models,
    "kill" the worst performing 1/eta fraction, and repeat.

    Arguments
    ---------
    params : dict
        Parameter space to be sampled with sklearn.model_selection.ParameterSampler
    model : sklearn Estimator
        Class that has support for partial_fit, score and {get, set}_params
    data : dict
        Data to train/validate on. {'train': [X, y], 'val': [X_val, y_val]}
    eta : int
        This controls how aggressive the "killing" of models should be. The
        "infinite time" theory suggest eta == np.e, about 3.
    n : int
        Number of models to train initially
    r : int
        Maximum number of times to call partial_fit on any one model
    s : int
        The bracket. Used for intermediate variables
    _prefix : str
        A prefix to make the IDs for this call to _successive_halving unique
        from other brackets
    dry_run : bool
        Whether this is a dry run or not. If it is, do not train the model and
        allow the score to be random. Setting ``dry_run=True`` is useful for
        collecting stats.
    scorer : callable
        The function used to score any model.

    Returns
    -------
    results : dict, {str: value}
        Results of this call to _successive_halving. Has values of

        history : list
            The history of every call to the model. This will be a list of dicts,
            and eventually found in self.history. Has keys of bracket,
            bracket_iter, val_score, model_id, partial_fit_iters, num_models as
            well as all the parameter keys.
        final_val_scores :


    """
    if any(x is None for x in [n, r, s, model, scorer]):
        raise ValueError('n, r, and s are required')

    best = {'val_score': -np.inf, 'config': {}}

    ids = [_prefix + '-' + str(i) for i in range(n)]
    params = ParameterSampler(params, len(ids))
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
        msg = ('Training %d models for %d iterations during iteration %d '
               'of bracket=%d')
        logger.info(msg, n_i, r_i, i, s)

        delayed_results = {k: dask.delayed(_train)(model, data, max_iter=r_i,
                                                   s=s, i=i, k=k, dry_run=dry_run,
                                                   scorer=scorer, **fit_kwargs)
                           for k, model in models.items()}
        results = {k: v.compute() for k, v in delayed_results.items()}
        #  results = dask.compute(delayed_results)[0]

        val_scores = {k: r[0] for k, r in results.items()}
        times += [{'id': k, **r[1]} for k, r in results.items()]

        final_val_scores.update(val_scores)
        history += [{'bracket': s, 'bracket_iter': i, 'val_score': val_scores[k],
                     'model_id': k, 'partial_fit_iters': iters,
                     'num_models': len(val_scores),
                     **{'param_' + _k: _v for _k, _v in params[k].items()}}
                    for k, model, in models.items()]

        models = _top_k(models, val_scores, k=max(1, math.floor(n_i / eta)))

        best_ = best
        if max(final_val_scores.values()) > best_['val_score']:
            max_loss_key = max(final_val_scores, key=final_val_scores.get)
            best_['val_score'] = final_val_scores[max_loss_key]
            best_['config'] = params[max_loss_key]

            msg = ("For bracket s={s}, new best validation score of {a:0.3f} "
                   "found with params={b}")
            msg = msg.format(a=best_['val_score'], b=best_['config'], s=s)
            logger.info(msg)
            print(msg)

        if len(models) == 1:
            break

    times = toolz.groupby('id', times)
    times = {k: [{k_: v_ for k_, v_ in vi.items() if k_ != 'id'} for vi in v]
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
    def __init__(self, model, param_distributions, max_iter=81, eta=3,
                 n_jobs=-1, scoring=None, **kwargs):
        """
        Parameters
        ----------
        model :
            The base model to try.
        param_distributions :
            Search space of various parameters. This mirrors the sklearn API.
        max_iter : int
            The maximum number of iterations to train a model on, as determined
            by ``model.partial_fit``.
        eta : optional, int
            How aggressive to be while search. The default is 3, and theory
            suggests that ``eta=e=2.718...``. Changing this is not recommended.
        n_jobs : int, optional
            Hyperband depends on another function, and they are completely
            indepdenent of one another and can be run embarassingly parallel.
            This function is submitted to the scheduler if ``n_jobs == -1``
            (which is the default behavior). Setting ``n_jobs == 1`` can help
            preserve memory on your machine.
        scoring : str | callable
            Scoring to use on the estimator. Higher is presumed to be better.
        """
        self.params = param_distributions
        self.model = model
        self.R = max_iter
        self.eta = eta
        self.best_val_score = -np.inf
        self.n_iter = 0
        self.scoring = scoring

        if n_jobs not in {-1, 1}:
            raise ValueError('n_jobs must be -1 (for full parallelization with '
                             'dask) or 1 (for regular list comphrension). '
                             'Setting n_jobs=1 can relieve memory pressure.')
        self.n_jobs = n_jobs

        est = model if not isinstance(model, ParallelPostFit) else model.estimator
        if not hasattr(est, 'partial_fit'):
            raise ValueError('Hyperband only supports models with partial_fit, '
                             'which provides support for incremental learning. '
                             'For non-incremental models, please consider '
                             'RandomizedSearchCV instead')
        if not hasattr(est, 'warm_start'):
            warn('model has no attribute warm_start. Hyperband will assume it '
                 'is reusing previous calls to `partial_fit` during each call '
                 'to `partial_fit`')
        else:
            if not est.warm_start:
                raise ValueError('Hyperband relies on model(..., warm_start=True).')

        self.history = []

        super_default = dict(iid=True, refit=True, cv=None,
                             error_score='raise',
                             return_train_score=_RETURN_TRAIN_SCORE_DEFAULT,
                             scheduler=None, cache_cv=True)
        for k, v in super_default.items():
            if k in kwargs and kwargs[k] != super_default[k]:
                msg = 'Hyperband ignores the keyword argument {k}={v}.'
                warn(msg.format(k=k, v=v))
        super(Hyperband, self).__init__(model, n_jobs=n_jobs,
                                        scoring=scoring, **kwargs)

    def fit(self, X, y, dry_run=False, **fit_kwargs):
        """
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

        To relieve memory pressure, it is recommended to scatter the input data
        (``X`` and ``y``) before calling ``fit``.

        """
        if not self.scoring:
            self.scorer_, _ = self._check_scorer(self.model)
        elif isinstance(self.model, Incremental):
            self.scorer_ = dask_ml.metrics.get_scorer(self.scoring)
        else:
            self.scorer_ = sklearn.metrics.get_scorer(self.scoring)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
        data = {'train': [X_train, y_train], 'val': [X_val, y_val]}

        # now start hyperband
        R, eta = self.R, self.eta
        s_max = math.floor(math.log(self.R, self.eta))
        B = (s_max + 1) * self.R
        all_kwargs = []
        for s in reversed(range(s_max + 1)):
            n = math.ceil((B / R) * eta**s / (s + 1))
            r = R * eta ** -s
            all_kwargs += [{'s': s, 'n': n, 'r': r, 'dry_run': dry_run,
                            'eta': eta, '_prefix': f's={s}',
                            'scorer': self.scorer_}]

        if self.n_jobs == -1:
            delayed_results = [dask.delayed(_successive_halving)(self.params,
                                                                 self.model,
                                                                 data,
                                                                 **kwargs,
                                                                 **fit_kwargs)
                               for kwargs in all_kwargs]
            results = [r.compute() for r in delayed_results]
            #  results = dask.compute(delayed_results)[0]
        else:
            results = [_successive_halving(self.params, self.model, data,
                                           **kwargs, **fit_kwargs)
                       for kwargs in all_kwargs]

        all_keys = [key for r in results for key in r['params'].keys()]
        history = sum([r['history'] for r in results], [])
        models = toolz.merge([r['models'] for r in results])
        params = toolz.merge([r['params'] for r in results])
        times = toolz.merge([r['times'] for r in results])
        val_scores = toolz.merge([r['final_val_scores'] for r in results])

        assert set(all_keys) == set(val_scores.keys())
        assert set(all_keys) == set(params.keys())

        cv_results_, best_idx = _get_cv_results(params=params, val_scores=val_scores,
                                                times=times)

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
            * brackets : a list of dicts, importable into pandas.DataFrame.
              Each value has a key of
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

        brackets = df[['bracket', 'bracket_iter', 'partial_fit_iters', 'num_models']]
        values = {(s, i, iter, n) for _, (s, i, iter, n) in brackets.iterrows()}
        values = sorted(values)
        brackets = [{'bracket': v[0], 'bracket_iter': v[1],
                     'partial_fit_iters': v[2], 'num_models': v[3]}
                    for v in values]

        return {'num_models': len(df.model_id.unique()),
                'num_partial_fit_calls': df.partial_fit_iters.sum(),
                'num_cv_splits': 1,
                'brackets': brackets}


def _LDtoDL(ld):
    """list of dicts to dict of lists"""
    keys = set(sum([list(d.keys()) for d in ld], []))
    values = {k: [d.get(k, None) for d in ld] for k in keys}
    return values


def _get_cv_results(params=None, val_scores=None, times=None):
    """
    Format to sklearn's cross validation results

    Parameters
    ----------
    params : dict
        Dict of parameters for each model
    val_scores : dict
        Dict of validation scores for each model
    times : dict
        Dict of times for each model, or really any values for each model.
        All values are added to each result.

    The keys for each dict are the IDs for each model.

    Returns
    -------
    cv_results : list
        A list of dicts describing each model
    best_index_ : int
        The integer in cv_results that represents the best performing model


    """
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
    """
    Parameters
    ----------
    val_scores : dict
        Floats describing how well each model performed.
    models : dict
        A dict of models.

    The keys for both dicts are the IDs for each model

    Returns
    ------
    The best performing model accoring to val_scores

    """
    scores = {k: val_scores[k] for k in models}
    best_id = max(scores, key=scores.get)
    return models[best_id]
