"""
Online model selection

Book-keeping is achieved by tokenization of args with an assertion to 
avoid mutation of existing items with different values. I.e. this ensures that our 
tokenize(func, args) works.

"""


import logging
import numbers
import operator as op
from multiprocessing import cpu_count

import dask
import dask.array as da
import toolz as tz
from dask import delayed
from dask.base import tokenize, Base
from dask.delayed import Delayed
from dask.threaded import get as threaded_get
from dask_searchcv._normalize import normalize_estimator
from dask_searchcv.methods import fit_transform, fit, score, cv_split, cv_extract, \
    feature_union_concat, feature_union, pipeline
from dask_searchcv.utils import to_keys, to_indexable
from sklearn import model_selection
from sklearn.base import is_classifier
from sklearn.model_selection import LeaveOneGroupOut, LeavePGroupsOut, LeavePOut, LeaveOneOut
from sklearn.model_selection._split import _CVIterableWrapper, PredefinedSplit, _BaseKFold, \
    BaseShuffleSplit, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _num_samples

log = logging.getLogger(__name__)


def _to_keys(dsk, *args):
    """Safe to_keys"""
    for x in args:
        if x is None:
            yield None
        elif isinstance(x, da.Array):
            x = delayed(x)
            dsk.update(x.dask)
            yield x.key
        elif isinstance(x, Delayed):
            dsk.update(x.dask)
            yield x.key
        else:
            assert not isinstance(x, Base)
            key = 'array-' + tokenize(x)
            dsk[key] = x
            yield key


def _persist_fit_params(dsk, fit_params):
    """Persist all input fit params to the cv cache and return keys in place with full name 
    """
    if fit_params:
        # A mapping of {name: (name, graph-key)}
        param_values = to_indexable(*fit_params.values(), allow_scalars=True)
        _fit_params = {k: (k, token) for (k, token) in zip(fit_params, _to_keys(dsk, *param_values))}
    else:
        _fit_params = {}
    return _fit_params


def _cv_extract_params(cvs, keys, full_names, tokens, n):
    # key doesn't matter (even if it's been modified) only token
    return {k: cvs.extract_param(name, token, n) for (k, name, token) in zip(keys, full_names, tokens)}


def _fit_params_for_n(fit_params, cv_name, n):
    return {k: (cv_name, n, full_name, token) for k, (full_name, token) in fit_params.items()}


def _group_fit_params(steps, fit_params):
    param_lk = {n: {} for n, _ in steps}
    for pname, pval in fit_params.items():
        step, param = pname.split('__', 1)
        param_lk[step][param] = pval
    return param_lk


def _get_fit_params(fit_params):
    """Reconstruct the parameters at time of submitting fit to graph"""
    if fit_params:
        assert len({(n, cv_name) for _, (cv_name, n, full_name, token) in fit_params.items()})
        cv_name, n, _, _ = next(iter(fit_params.values()))
        keys, full_names, tokens = (
            list(fit_params.keys()),
            list(tz.pluck(2, fit_params.values())),
            list(tz.pluck(3, fit_params.values()))
        )
        return _cv_extract_params, cv_name, keys, full_names, tokens, n
    else:
        return {}


def check_cv(cv=3, y=None, classifier=False):
    """Dask aware version of ``sklearn.model_selection.check_cv``

    Same as the scikit-learn version, but works if ``y`` is a dask object.
    """
    if cv is None:
        cv = 3

    # If ``cv`` is not an integer, the scikit-learn implementation doesn't
    # touch the ``y`` object, so passing on a dask object is fine
    if not isinstance(y, Base) or not isinstance(cv, numbers.Integral):
        return model_selection.check_cv(cv, y, classifier)

    if classifier:
        # ``y`` is a dask object. We need to compute the target type
        target_type = delayed(type_of_target, pure=True)(y).compute()
        if target_type in ('binary', 'multiclass'):
            return StratifiedKFold(cv)
    return KFold(cv)


def compute_n_splits(cv, X, y=None, groups=None):
    """Return the number of splits.

    Parameters
    ----------
    cv : BaseCrossValidator
    X, y, groups : array_like, dask object, or None

    Returns
    -------
    n_splits : int
    """
    if not any(isinstance(i, Base) for i in (X, y, groups)):
        return cv.get_n_splits(X, y, groups)

    if isinstance(cv, (_BaseKFold, BaseShuffleSplit)):
        return cv.n_splits

    elif isinstance(cv, PredefinedSplit):
        return len(cv.unique_folds)

    elif isinstance(cv, _CVIterableWrapper):
        return len(cv.cv)

    elif isinstance(cv, (LeaveOneOut, LeavePOut)) and not isinstance(X, Base):
        # Only `X` is referenced for these classes
        return cv.get_n_splits(X, None, None)

    elif (isinstance(cv, (LeaveOneGroupOut, LeavePGroupsOut)) and not
          isinstance(groups, Base)):
        # Only `groups` is referenced for these classes
        return cv.get_n_splits(None, None, groups)

    else:
        return delayed(cv).get_n_splits(X, y, groups).compute()


def _normalize_n_jobs(n_jobs):
    if not isinstance(n_jobs, int):
        raise TypeError("n_jobs should be an int, got %s" % n_jobs)
    if n_jobs == -1:
        n_jobs = None  # Scheduler default is use all cores
    elif n_jobs < -1:
        n_jobs = cpu_count() + 1 + n_jobs
    return n_jobs


_scheduler_aliases = {'sync': 'synchronous',
                      'sequential': 'synchronous',
                      'threaded': 'threading'}


def _normalize_scheduler(scheduler, n_jobs, loop=None):
    # Default
    if scheduler is None:
        scheduler = dask.context._globals.get('get')
        if scheduler is None:
            scheduler = dask.get if n_jobs == 1 else threaded_get
        return scheduler

    # Get-functions
    if callable(scheduler):
        return scheduler

    # Support name aliases
    if isinstance(scheduler, str):
        scheduler = _scheduler_aliases.get(scheduler, scheduler)

    if scheduler in ('threading', 'multiprocessing') and n_jobs == 1:
        scheduler = dask.get
    elif scheduler == 'threading':
        scheduler = threaded_get
    elif scheduler == 'multiprocessing':
        from dask.multiprocessing import get as scheduler
    elif scheduler == 'synchronous':
        scheduler = dask.get
    else:
        try:
            from dask.distributed import Client
            # We pass loop to make testing possible, not needed for normal use
            return Client(scheduler, set_as_default=False, loop=loop).get
        except Exception as e:
            msg = ("Failed to initialize scheduler from parameter %r. "
                   "This could be due to a typo, or a failure to initialize "
                   "the distributed scheduler. Original error is below:\n\n"
                   "%r" % (scheduler, e))
        # Re-raise outside the except to provide a cleaner error message
        raise ValueError(msg)
    return scheduler


def _str_cmp(x, y):
    return str(x) == str(y)

_exception_string = (
        'graph already has a key {k} with a different value: old: {vold}, new: {vnew}'
    )


def update_dsk(dsk, k, v, cmp=_str_cmp):
    if k in dsk:
        if not cmp(dsk[k], v):
            raise ValueError(
                _exception_string.format(d=dsk, k=k, vold=dsk[k], vnew=v))
    else:
        dsk[k] = v


def assoc_params_with_steps(params, steps):
    plist = []
    for step, _ in steps:
        _params = {k: v for k, v in params.items() if k.split('__')[0] == step and k != step}
        plist.append((step, {'__'.join(k.split('__')[1:]): v for k, v in _params.items()}))
    return plist


def flesh_out_params(estimator, params):
    if any(k.endswith('transformer_list') for k in params.keys()):
        raise(NotImplementedError("Setting FeatureUnion.transformer_list "
                                  "in a DaskBaseSearchCV search"))
    if any(k.endswith('steps') for k in params.keys()):
        raise(NotImplementedError("Setting Pipeline.steps "
                                  "in a DaskBaseSearchCV search"))

    default_params = estimator.get_params()
    # we dissoc the transformer lists, todo: do this for estimators as well
    # default_params = {k: v for k, v in default_params.items() if not k.endswith('transformer_list')}

    # note: there can be new parameters if they are
    # get new default parameters implied by subestimators in a pipeline
    # todo: sub_estimators = []
    # if not all(k in default_params for k in params):
    #     # log.warn('All parameters should be relevant to the estimator - pruning extras')
    #     # params = {k: v for k, v in params.items() if k in default_params}
    #     raise ValueError('All parameters should be relevant to the estimator')

    return tz.merge(default_params, params)


def check_estimator_parameters(est, params):
    if not all(k in est.get_params() for k in params):
        raise ValueError("We only accept parameters that are in est")


def do_fit(dsk, est, X_name, y_name, params, fit_params, error_score):
    check_estimator_parameters(est, params)

    if isinstance(est, Pipeline):
        return do_pipeline(
            dsk, est, X_name, y_name, params, fit_params, error_score, transform=False)
    elif isinstance(est, FeatureUnion):
        return tz.first(do_featureunion(dsk, est, X_name, y_name, params, fit_params, error_score))
    else:
        est_type = type(est).__name__.lower()
        est_name = est_type + '-' + tokenize(normalize_estimator(est))
        update_dsk(dsk, est_name, est)
        fit_name = '%s-fit-%s' % (est_type, tokenize(est_name, X_name, y_name, params, fit_params))

        fit_params_ = _get_fit_params(fit_params)
        update_dsk(dsk, fit_name, (
            fit, est_name, X_name, y_name, error_score, list(params.keys()), list(params.values()),
            fit_params_))
        return fit_name


def do_fit_transform(dsk, est, X_name, y_name, params, fit_params, error_score):
    check_estimator_parameters(est, params)
    if isinstance(est, Pipeline):
        return do_pipeline(
            dsk, est, X_name, y_name, params, fit_params, error_score, transform=True)
    elif isinstance(est, FeatureUnion):
        return do_featureunion(dsk, est, X_name, y_name, params, fit_params, error_score)
    else:
        est_type = type(est).__name__.lower()
        est_name = None if est is None else est_type + '-' + tokenize(normalize_estimator(est))
        update_dsk(dsk, est_name, est)

        token = tokenize(est_name, X_name, y_name, params, fit_params)
        fit_Xt_name = '%s-fit-transform-%s' % (est_type, token)
        fit_name = '%s-fit-%s' % (est_type, token)
        Xt_name = '%s-transform-%s' % (est_type, token)

        fit_params_ = _get_fit_params(fit_params)
        update_dsk(dsk, fit_Xt_name, (
            fit_transform, est_name, X_name, y_name, error_score, list(params.keys()), list(params.values()),
            fit_params_))
        update_dsk(dsk, fit_name, (op.getitem, fit_Xt_name, 0))
        update_dsk(dsk, Xt_name, (op.getitem, fit_Xt_name, 1))

        return fit_name, Xt_name


def do_pipeline(dsk, est, X_name, y_name, params, fit_params, error_score, transform=False):
    check_estimator_parameters(est, params)

    params_dict = dict(assoc_params_with_steps(params, est.steps))
    fit_params_dict = _group_fit_params(est.steps, fit_params)
    fit_steps = []
    Xt_name = X_name
    for step_name, step in est.steps[:-1]:
        fit_params_ = _get_fit_params(fit_params_dict[step_name])
        fit_name, Xt_name = do_fit_transform(
            dsk, step, Xt_name, y_name,
            params_dict[step_name], fit_params_, error_score
        ) if step else (None, Xt_name)  # ... pass-through for pipeline
        fit_steps.append(fit_name)
    step_name, step = est.steps[-1]
    # fit_params_ = _get_fit_params(fit_params_dict[step_name])

    if transform:
        fit_name, Xt_name = do_fit_transform(
            dsk, step, Xt_name, y_name,
            params_dict[step_name], fit_params_dict[step_name], error_score
        ) if step else (None, Xt_name)
    else:
        fit_name = do_fit(
            dsk, step, Xt_name, y_name,
            params_dict[step_name], fit_params_dict[step_name], error_score
        ) if step else None
    fit_steps.append(fit_name)

    est_type = type(est).__name__.lower()
    # fit_params = _get_fit_params(fit_params)  # we keep the keys for tokenizing
    token = tokenize(X_name, y_name, params, fit_params, error_score, transform)
    fit_name = est_type + '-' + token
    step_names = list(tz.pluck(0, est.steps))
    update_dsk(dsk, fit_name, (pipeline, step_names, fit_steps))

    return (fit_name, Xt_name) if transform else fit_name


def do_featureunion(dsk, est, X_name, y_name, params, fit_params, error_score):
    check_estimator_parameters(est, params)

    est_type = type(est).__name__.lower()
    est_name = est_type + '-' + tokenize(normalize_estimator(est))
    update_dsk(dsk, est_name, est)

    token = tokenize(est_name, X_name, params, fit_params, error_score)
    n_samples_name = 'n_samples-' + token
    update_dsk(dsk, n_samples_name, (_num_samples, X_name))

    fit_steps = []
    tr_Xs = []
    params_dict = dict(assoc_params_with_steps(params, est.transformer_list))
    fit_params_dict = _group_fit_params(est.transformer_list, fit_params)
    for (step_name, step) in est.transformer_list:
        params_ = params_dict[step_name]
        fit_params_ = _get_fit_params(fit_params_dict[step_name])
        if step is None:
            fit_name, Xt = None, None
        else:
            fit_name, Xt = do_fit_transform(
                dsk, step, X_name, y_name,
                params_, fit_params_, error_score
            )
        fit_steps.append(fit_name)
        tr_Xs.append(Xt)

    w = params.get('transformer_weights', est.transformer_weights)
    wdict = w or {}
    weight_list = [wdict.get(n) for n, _ in est.transformer_list]

    fit_name = est_type + '-' + token
    tr_name = 'feature-union-concat-' + token
    step_names, steps = zip(*est.transformer_list)
    update_dsk(dsk, fit_name, (feature_union, step_names, fit_steps, wdict))
    update_dsk(dsk, tr_name, (feature_union_concat, tr_Xs, n_samples_name, weight_list))

    return fit_name, tr_name


def do_score(dsk, fit_name, X_name, y_name, X_test_name, y_test_name, scorer, return_train_score):
    score_name = 'score-' + tokenize(fit_name, X_test_name, y_test_name, scorer)
    update_dsk(dsk, score_name, (
        score, fit_name, X_test_name, y_test_name, X_name if return_train_score else None, y_name,
        scorer
    ))
    return score_name


def do_fit_and_score(
    dsk, est, Xtrain_name, ytrain_name, Xtest_name, ytest_name, params, fit_params, scorer,
    return_train_score, error_score
):
    # n_and_fit_params = _get_fit_params(cv, fit_params, n_splits)
    fit_name = do_fit(dsk, est, Xtrain_name, ytrain_name, params, fit_params, error_score)
    score_name = do_score(
        dsk, fit_name, Xtrain_name, ytrain_name, Xtest_name, ytest_name, scorer, return_train_score)

    return score_name


def build_graph(estimator, X, y, cv, groups, cache_cv=True):
    dsk = {}
    X, y, groups = to_indexable(X, y, groups)
    X_name, y_name, groups_name = to_keys(dsk, X, y, groups)
    cv = check_cv(cv, y, is_classifier(estimator))
    n_splits = compute_n_splits(cv, X, y, groups)
    is_pairwise = getattr(estimator, '_pairwise', False)
    # accumulating all keys that influence input data:
    data_token = tokenize(cv, X_name, y_name, groups_name, is_pairwise)
    # should we put cache_cv in the data_token, (does it have any effect on the computation graph)?
    cv_name = 'cv-split-' + data_token
    update_dsk(dsk, cv_name, (cv_split, cv, X_name, y_name, groups_name, is_pairwise, cache_cv))

    return dsk, X_name, y_name, cv_name, n_splits


def update_graph(dsk, estimator, X_name, y_name, params, fit_params, cv_name, n_splits, scorer,
                 return_train_score, error_score):
    # munge parameters, for unique keys:
    params = flesh_out_params(estimator, params)
    fit_params = _persist_fit_params(dsk, fit_params)  # todo: safe update of graph??

    cv_score_names = []
    for n in range(n_splits):
        xtrain = (cv_extract, cv_name, X_name, y_name, True, True, n)
        ytrain = (cv_extract, cv_name, X_name, y_name, False, True, n)
        xtest = (cv_extract, cv_name, X_name, y_name, True, False, n)
        ytest = (cv_extract, cv_name, X_name, y_name, False, False, n)
        fit_params_n = _fit_params_for_n(fit_params, cv_name, n)

        score_name = do_fit_and_score(dsk, estimator, xtrain, ytrain, xtest, ytest, params,
                                      fit_params_n, scorer, return_train_score, error_score)
        cv_score_names.append(score_name)

    return cv_score_names
