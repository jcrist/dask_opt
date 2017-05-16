"""
Online model selection

book-keeping for caching jobs that have already been submitted is achieved by tokenization of 
estimators and input parameter giving unique keys on the dask graph. The default parameters of 
estimators are combined with the input parameters to give unique keys.

we allow estimator duplicates since a pipeline or feature union may contain multiple instances with 
different ids (... we could change the value comparison algorithm to achieve this too)

"""


import logging
import numbers
import operator as op
from multiprocessing import cpu_count

import dask
import dask.array as da
import toolz as tz
import numpy as np
from dask import delayed
from dask.base import tokenize, Base
from dask.delayed import Delayed
from dask.threaded import get as threaded_get
from dask.utils import derived_from
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection._search import BaseSearchCV, _check_param_grid
from sklearn.utils.metaestimators import if_delegate_has_method

from dask_searchcv._normalize import normalize_estimator
from dask_searchcv.methods import fit_transform, fit, score, cv_split, cv_extract, \
    feature_union_concat, feature_union, pipeline, cv_n_samples, create_cv_results, fit_best
from dask_searchcv.utils import to_keys, to_indexable
from sklearn import model_selection
from sklearn.base import is_classifier, BaseEstimator, MetaEstimatorMixin, clone
from sklearn.model_selection import LeaveOneGroupOut, LeavePGroupsOut, LeavePOut, LeaveOneOut
from sklearn.model_selection._split import _CVIterableWrapper, PredefinedSplit, _BaseKFold, \
    BaseShuffleSplit, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _num_samples, check_is_fitted, NotFittedError

log = logging.getLogger(__name__)

try:
    from cytoolz import get, pluck, concat
except:  # pragma: no cover
    from toolz import get, pluck, concat


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
    if fit_params is not None:
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
    # todo: improve this comparison function
    if hasattr(x, 'fit') and hasattr(y, 'fit'):
        return normalize_estimator(x) == normalize_estimator(y)
    else:
        return str(x) == str(y)

_exception_string = (
        'graph already has a key {k} with a different value: old: {vold}, new: {vnew}'
    )


def update_dsk(dsk, k, v, cmp=_str_cmp):
    """update the dask graph, checking that """
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
    # we dissoc the transformer lists
    # default_params = {k: v for k, v in default_params.items() if not k.endswith('transformer_list')}
    # note: there can be new parameters if they are
    # get new default parameters implied by subestimators in a pipeline
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


class DaskBaseSearchCV(BaseEstimator, MetaEstimatorMixin):
    """Base class for hyper parameter search with cross-validation."""

    def __init__(self, estimator, scoring=None, iid=True, refit=True, cv=None,
                 error_score='raise', return_train_score=True, scheduler=None,
                 n_jobs=-1, cache_cv=True):
        self.scoring = scoring
        self.estimator = estimator
        self.iid = iid
        self.refit = refit
        self.cv = cv
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.scheduler = scheduler
        self.n_jobs = n_jobs
        self.cache_cv = cache_cv

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    @property
    def best_params_(self):
        check_is_fitted(self, 'cv_results_')
        return self.cv_results_['params'][self.best_index_]

    @property
    def best_score_(self):
        check_is_fitted(self, 'cv_results_')
        return self.cv_results_['mean_test_score'][self.best_index_]

    def _check_is_fitted(self, method_name):
        if not self.refit:
            msg = ('This {0} instance was initialized with refit=False. {1} '
                   'is available only after refitting on the best '
                   'parameters.').format(type(self).__name__, method_name)
            raise NotFittedError(msg)
        else:
            check_is_fitted(self, 'best_estimator_')

    @property
    def classes_(self):
        self._check_is_fitted("classes_")
        return self.best_estimator_.classes_

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    @derived_from(BaseSearchCV)
    def predict(self, X):
        self._check_is_fitted('predict')
        return self.best_estimator_.predict(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    @derived_from(BaseSearchCV)
    def predict_proba(self, X):
        self._check_is_fitted('predict_proba')
        return self.best_estimator_.predict_proba(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    @derived_from(BaseSearchCV)
    def predict_log_proba(self, X):
        self._check_is_fitted('predict_log_proba')
        return self.best_estimator_.predict_log_proba(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    @derived_from(BaseSearchCV)
    def decision_function(self, X):
        self._check_is_fitted('decision_function')
        return self.best_estimator_.decision_function(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    @derived_from(BaseSearchCV)
    def transform(self, X):
        self._check_is_fitted('transform')
        return self.best_estimator_.transform(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    @derived_from(BaseSearchCV)
    def inverse_transform(self, Xt):
        self._check_is_fitted('inverse_transform')
        return self.best_estimator_.transform(Xt)

    @derived_from(BaseSearchCV)
    def score(self, X, y=None):
        if self.scorer_ is None:
            raise ValueError("No score function explicitly defined, "
                             "and the estimator doesn't provide one %s"
                             % self.best_estimator_)
        return self.scorer_(self.best_estimator_, X, y)

    def fit(self, X, y=None, groups=None, block=False, **fit_params):
        """Run fit with all sets of parameters.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.
        groups : array-like, shape = [n_samples], optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        **fit_params
            Parameters passed to the ``fit`` method of the estimator
        """
        estimator = self.estimator
        self.scorer_ = check_scoring(estimator, scoring=self.scoring)
        error_score = self.error_score
        if not (isinstance(error_score, numbers.Number) or
                error_score == 'raise'):
            raise ValueError("error_score must be the string 'raise' or a"
                             " numeric value.")

        dsk, X_name, y_name, cv_name, n_splits = build_graph(
            estimator, X, y, self.cv, groups, self.cache_cv)

        self.dask_graph_ = dsk
        self.n_splits_ = n_splits

        fit_params = _persist_fit_params(dsk, fit_params)

        # populate the graph with jobs
        keys, params_list = [], []
        for params in self._get_param_iterator():
            cv_score_names = update_graph(dsk, estimator, X_name, y_name, params, fit_params,
                                          cv_name, n_splits,
                                          self.scorer_, self.return_train_score,
                                          error_score=error_score)
            keys.append(cv_score_names)
            params_list.append(params)

        n_jobs = _normalize_n_jobs(self.n_jobs)
        scheduler = _normalize_scheduler(self.scheduler, n_jobs)

        scores = scheduler(dsk, keys, num_workers=n_jobs)
        scores = tuple(concat(zip(*scores)))  # fixme: ugly hack for the moment to compare

        if self.iid:
            weights_name = 'cv-n-samples-' + tokenize(
                normalize_estimator(estimator), X_name, y_name, params, fit_params,
                cv_name, n_splits)
            update_dsk(dsk, weights_name, (cv_n_samples, cv_name))
            weights = scheduler(dsk, weights_name)
        else:
            weights = None

        self.cv_results_ = results = create_cv_results(
            scores, params_list, n_splits, error_score, weights=weights
        )

        self.best_index_ = np.flatnonzero(results["rank_test_score"] == 1)[0]

        if self.refit:
            best_params = results['params'][self.best_index_]
            # we do this locally (should be possible by key (from fit_name)):
            token = tokenize(normalize_estimator(estimator), best_params, X_name, y_name, fit_params)
            best_fit_key = 'best-fit-{}'.format(token)
            dsk[best_fit_key] = (fit_best, clone(estimator), best_params, X_name, y_name, fit_params)
            self.best_estimator_ = scheduler(dsk, best_fit_key)

        return self

    def visualize(self, filename='mydask', format=None, **kwargs):
        """Render the task graph for this parameter search using ``graphviz``.

        Requires ``graphviz`` to be installed.

        Parameters
        ----------
        filename : str or None, optional
            The name (without an extension) of the file to write to disk.  If
            `filename` is None, no file will be written, and we communicate
            with dot using only pipes.
        format : {'png', 'pdf', 'dot', 'svg', 'jpeg', 'jpg'}, optional
            Format in which to write output file.  Default is 'png'.
        **kwargs
            Additional keyword arguments to forward to ``dask.dot.to_graphviz``.

        Returns
        -------
        result : IPython.diplay.Image, IPython.display.SVG, or None
            See ``dask.dot.dot_graph`` for more information.
        """
        check_is_fitted(self, 'dask_graph_')
        return dask.visualize(self.dask_graph_, filename=filename,
                              format=format, **kwargs)


_DOC_TEMPLATE = """{oneliner}

{name} implements a "fit" and a "score" method.
It also implements "predict", "predict_proba", "decision_function",
"transform" and "inverse_transform" if they are implemented in the
estimator used.

{description}

Parameters
----------
estimator : estimator object.
    This is assumed to implement the scikit-learn estimator interface.
    Either estimator needs to provide a ``score`` function,
    or ``scoring`` must be passed.

{parameters}

scoring : string, callable or None, default=None
    A string (see model evaluation documentation) or
    a scorer callable object / function with signature
    ``scorer(estimator, X, y)``.
    If ``None``, the ``score`` method of the estimator is used.

iid : boolean, default=True
    If True, the data is assumed to be identically distributed across
    the folds, and the loss minimized is the total loss per sample,
    and not the mean loss across the folds.

cv : int, cross-validation generator or an iterable, optional
    Determines the cross-validation splitting strategy.
    Possible inputs for cv are:
        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a ``(Stratified)KFold``,
        - An object to be used as a cross-validation generator.
        - An iterable yielding train, test splits.

    For integer/None inputs, if the estimator is a classifier and ``y`` is
    either binary or multiclass, ``StratifiedKFold`` is used. In all
    other cases, ``KFold`` is used.

refit : boolean, default=True
    Refit the best estimator with the entire dataset.
    If "False", it is impossible to make predictions using
    this {name} instance after fitting.

error_score : 'raise' (default) or numeric
    Value to assign to the score if an error occurs in estimator fitting.
    If set to 'raise', the error is raised. If a numeric value is given,
    FitFailedWarning is raised. This parameter does not affect the refit
    step, which will always raise the error.

return_train_score : boolean, default=True
    If ``'False'``, the ``cv_results_`` attribute will not include training
    scores.

scheduler : string, callable, or None, default=None
    The dask scheduler to use. Default is to use the global scheduler if set,
    and fallback to the threaded scheduler otherwise. To use a different
    scheduler, specify it by name (either "threading", "multiprocessing",
    or "synchronous") or provide the scheduler ``get`` function. Other
    arguments are assumed to be the address of a distributed scheduler,
    and passed to ``dask.distributed.Client``.

n_jobs : int, default=-1
    Number of jobs to run in parallel. Ignored for the synchronous and
    distributed schedulers. If ``n_jobs == -1`` [default] all cpus are used.
    For ``n_jobs < -1``, ``(n_cpus + 1 + n_jobs)`` are used.

cache_cv : bool, default=True
    Whether to extract each train/test subset at most once in each worker
    process, or every time that subset is needed. Caching the splits can
    speedup computation at the cost of increased memory usage per worker
    process.

    If True, worst case memory usage is ``(n_splits + 1) * (X.nbytes +
    y.nbytes)`` per worker. If False, worst case memory usage is
    ``(n_threads_per_worker + 1) * (X.nbytes + y.nbytes)`` per worker.

Examples
--------
{example}

Attributes
----------
cv_results_ : dict of numpy (masked) ndarrays
    A dict with keys as column headers and values as columns, that can be
    imported into a pandas ``DataFrame``.

    For instance the below given table

    +------------+-----------+------------+-----------------+---+---------+
    |param_kernel|param_gamma|param_degree|split0_test_score|...|rank.....|
    +============+===========+============+=================+===+=========+
    |  'poly'    |     --    |      2     |        0.8      |...|    2    |
    +------------+-----------+------------+-----------------+---+---------+
    |  'poly'    |     --    |      3     |        0.7      |...|    4    |
    +------------+-----------+------------+-----------------+---+---------+
    |  'rbf'     |     0.1   |     --     |        0.8      |...|    3    |
    +------------+-----------+------------+-----------------+---+---------+
    |  'rbf'     |     0.2   |     --     |        0.9      |...|    1    |
    +------------+-----------+------------+-----------------+---+---------+

    will be represented by a ``cv_results_`` dict of::

        {{
        'param_kernel': masked_array(data = ['poly', 'poly', 'rbf', 'rbf'],
                                        mask = [False False False False]...)
        'param_gamma': masked_array(data = [-- -- 0.1 0.2],
                                    mask = [ True  True False False]...),
        'param_degree': masked_array(data = [2.0 3.0 -- --],
                                        mask = [False False  True  True]...),
        'split0_test_score'  : [0.8, 0.7, 0.8, 0.9],
        'split1_test_score'  : [0.82, 0.5, 0.7, 0.78],
        'mean_test_score'    : [0.81, 0.60, 0.75, 0.82],
        'std_test_score'     : [0.02, 0.01, 0.03, 0.03],
        'rank_test_score'    : [2, 4, 3, 1],
        'split0_train_score' : [0.8, 0.9, 0.7],
        'split1_train_score' : [0.82, 0.5, 0.7],
        'mean_train_score'   : [0.81, 0.7, 0.7],
        'std_train_score'    : [0.03, 0.03, 0.04],
        'params'             : [{{'kernel': 'poly', 'degree': 2}}, ...],
        }}

    NOTE that the key ``'params'`` is used to store a list of parameter
    settings dict for all the parameter candidates.

best_estimator_ : estimator
    Estimator that was chosen by the search, i.e. estimator
    which gave highest score (or smallest loss if specified)
    on the left out data. Not available if refit=False.

best_score_ : float
    Score of best_estimator on the left out data.

best_params_ : dict
    Parameter setting that gave the best results on the hold out data.

best_index_ : int
    The index (of the ``cv_results_`` arrays) which corresponds to the best
    candidate parameter setting.

    The dict at ``search.cv_results_['params'][search.best_index_]`` gives
    the parameter setting for the best model, that gives the highest
    mean score (``search.best_score_``).

scorer_ : function
    Scorer function used on the held out data to choose the best
    parameters for the model.

n_splits_ : int
    The number of cross-validation splits (folds/iterations).

Notes
------
The parameters selected are those that maximize the score of the left out
data, unless an explicit score is passed in which case it is used instead.
"""

# ------------ #
# GridSearchCV #
# ------------ #

_grid_oneliner = """\
Exhaustive search over specified parameter values for an estimator.\
"""
_grid_description = """\
The parameters of the estimator used to apply these methods are optimized
by cross-validated grid-search over a parameter grid.\
"""
_grid_parameters = """\
param_grid : dict or list of dictionaries
    Dictionary with parameters names (string) as keys and lists of
    parameter settings to try as values, or a list of such
    dictionaries, in which case the grids spanned by each dictionary
    in the list are explored. This enables searching over any sequence
    of parameter settings.\
"""
_grid_example = """\
>>> import dask_searchcv as dcv
>>> from sklearn import svm, datasets
>>> iris = datasets.load_iris()
>>> parameters = {'kernel': ['linear', 'rbf'], 'C': [1, 10]}
>>> svc = svm.SVC()
>>> clf = dcv.GridSearchCV(svc, parameters)
>>> clf.fit(iris.data, iris.target)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
GridSearchCV(cache_cv=..., cv=..., error_score=...,
        estimator=SVC(C=..., cache_size=..., class_weight=..., coef0=...,
                      decision_function_shape=..., degree=..., gamma=...,
                      kernel=..., max_iter=-1, probability=False,
                      random_state=..., shrinking=..., tol=...,
                      verbose=...),
        iid=..., n_jobs=..., param_grid=..., refit=..., return_train_score=...,
        scheduler=..., scoring=...)
>>> sorted(clf.cv_results_.keys())  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
['mean_test_score', 'mean_train_score', 'param_C', 'param_kernel',...
 'params', 'rank_test_score', 'split0_test_score', 'split0_train_score',...
 'split1_test_score', 'split1_train_score', 'split2_test_score',...
 'split2_train_score', 'std_test_score', 'std_train_score'...]\
"""


class GridSearchCV(DaskBaseSearchCV):
    __doc__ = _DOC_TEMPLATE.format(name="GridSearchCV",
                                   oneliner=_grid_oneliner,
                                   description=_grid_description,
                                   parameters=_grid_parameters,
                                   example=_grid_example)

    def __init__(self, estimator, param_grid, scoring=None, iid=True,
                 refit=True, cv=None, error_score='raise',
                 return_train_score=True, scheduler=None, n_jobs=-1,
                 cache_cv=True):
        super(GridSearchCV, self).__init__(estimator=estimator,
                scoring=scoring, iid=iid, refit=refit, cv=cv,
                error_score=error_score, return_train_score=return_train_score,
                scheduler=scheduler, n_jobs=n_jobs, cache_cv=cache_cv)

        _check_param_grid(param_grid)
        self.param_grid = param_grid

    def _get_param_iterator(self):
        """Return ParameterGrid instance for the given param_grid"""
        return model_selection.ParameterGrid(self.param_grid)


# ------------------ #
# RandomizedSearchCV #
# ------------------ #

_randomized_oneliner = "Randomized search on hyper parameters."
_randomized_description = """\
In contrast to GridSearchCV, not all parameter values are tried out, but
rather a fixed number of parameter settings is sampled from the specified
distributions. The number of parameter settings that are tried is
given by n_iter.

If all parameters are presented as a list, sampling without replacement is
performed. If at least one parameter is given as a distribution, sampling
with replacement is used. It is highly recommended to use continuous
distributions for continuous parameters.\
"""
_randomized_parameters = """\
param_distributions : dict
    Dictionary with parameters names (string) as keys and distributions
    or lists of parameters to try. Distributions must provide a ``rvs``
    method for sampling (such as those from scipy.stats.distributions).
    If a list is given, it is sampled uniformly.

n_iter : int, default=10
    Number of parameter settings that are sampled. n_iter trades
    off runtime vs quality of the solution.

random_state : int or RandomState
    Pseudo random number generator state used for random uniform sampling
    from lists of possible values instead of scipy.stats distributions.\
"""
_randomized_example = """\
>>> import dask_searchcv as dcv
>>> from scipy.stats import expon
>>> from sklearn import svm, datasets
>>> iris = datasets.load_iris()
>>> parameters = {'C': expon(scale=100), 'kernel': ['linear', 'rbf']}
>>> svc = svm.SVC()
>>> clf = dcv.RandomizedSearchCV(svc, parameters, n_iter=100)
>>> clf.fit(iris.data, iris.target)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
RandomizedSearchCV(cache_cv=..., cv=..., error_score=...,
        estimator=SVC(C=..., cache_size=..., class_weight=..., coef0=...,
                      decision_function_shape=..., degree=..., gamma=...,
                      kernel=..., max_iter=..., probability=...,
                      random_state=..., shrinking=..., tol=...,
                      verbose=...),
        iid=..., n_iter=..., n_jobs=..., param_distributions=...,
        random_state=..., refit=..., return_train_score=...,
        scheduler=..., scoring=...)
>>> sorted(clf.cv_results_.keys())  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
['mean_test_score', 'mean_train_score', 'param_C', 'param_kernel',...
 'params', 'rank_test_score', 'split0_test_score', 'split0_train_score',...
 'split1_test_score', 'split1_train_score', 'split2_test_score',...
 'split2_train_score', 'std_test_score', 'std_train_score'...]\
"""


class RandomizedSearchCV(DaskBaseSearchCV):
    __doc__ = _DOC_TEMPLATE.format(name="RandomizedSearchCV",
                                   oneliner=_randomized_oneliner,
                                   description=_randomized_description,
                                   parameters=_randomized_parameters,
                                   example=_randomized_example)

    def __init__(self, estimator, param_distributions, n_iter=10,
                 random_state=None, scoring=None, iid=True, refit=True,
                 cv=None, error_score='raise', return_train_score=True,
                 scheduler=None, n_jobs=-1, cache_cv=True):

        super(RandomizedSearchCV, self).__init__(estimator=estimator,
                scoring=scoring, iid=iid, refit=refit, cv=cv,
                error_score=error_score, return_train_score=return_train_score,
                scheduler=scheduler, n_jobs=n_jobs, cache_cv=cache_cv)

        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state

    def _get_param_iterator(self):
        """Return ParameterSampler instance for the given distributions"""
        return model_selection.ParameterSampler(self.param_distributions,
                self.n_iter, random_state=self.random_state)
