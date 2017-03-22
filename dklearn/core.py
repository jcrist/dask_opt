from __future__ import absolute_import, division, print_function

from operator import getitem
from collections import defaultdict
import numbers

from cytoolz import get, pluck
from dask.base import tokenize, Base
from dask.delayed import delayed
from sklearn.base import is_classifier, clone
from sklearn.model_selection import check_cv as _sklearn_check_cv
from sklearn.model_selection._split import (_BaseKFold,
                                            BaseShuffleSplit,
                                            KFold,
                                            StratifiedKFold,
                                            LeaveOneOut,
                                            LeaveOneGroupOut,
                                            LeavePOut,
                                            LeavePGroupsOut,
                                            PredefinedSplit,
                                            _CVIterableWrapper)
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import type_of_target

from .methods import (fit, fit_transform, pipeline, fit_best, get_best_params,
                      create_cv_results, cv_split, cv_n_samples, cv_extract,
                      cv_extract_pairwise, score, MISSING)
from ._normalize import normalize_estimator

from .utils import to_indexable, to_keys, unzip


class TokenIterator(object):
    def __init__(self, base_token):
        self.token = base_token
        self.counts = defaultdict(int)

    def __call__(self, est):
        typ = type(est)
        c = self.counts[typ]
        self.counts[typ] += 1
        return self.token if c == 0 else self.token + str(c)


def build_graph(estimator, cv, scorer, candidate_params, X, y=None,
                groups=None, iid=True, refit=True,
                error_score='raise', return_train_score=True):

    X, y, groups = to_indexable(X, y, groups)
    cv = check_cv(cv, y, is_classifier(estimator))
    # "pairwise" estimators require a different graph for CV splitting
    is_pairwise = getattr(estimator, '_pairwise', False)

    (dsk, n_splits,
     X_name, y_name,
     X_train, y_train,
     X_test, y_test, weights) = initialize_graph(cv, X, y, groups, is_pairwise)

    fields, tokens, params = normalize_params(candidate_params)

    # Token used for all fit and score steps in the graph.
    main_token = tokenize(normalize_estimator(estimator), fields, params,
                          X_train, y_train, X_test, y_test,
                          error_score == 'raise', return_train_score)

    # Fit the estimator on the training data
    X_trains = [(X_train,)] * len(params)
    y_trains = [(y_train,)] * len(params)
    fit_ests = do_fit(dsk, TokenIterator(main_token), estimator,
                      fields, tokens, params,
                      X_trains, y_trains, n_splits, error_score)

    test_score = 'test-score-' + main_token

    test_scores = []
    test_scores_append = test_scores.append
    for (name, m) in fit_ests:
        for n in range(n_splits):
            dsk[(test_score, m, n)] = (score, (name, m, n), (X_test, n),
                                       (y_test, n), scorer)
            test_scores_append((test_score, m, n))

    if return_train_score:
        train_score = 'train-score-' + main_token
        train_scores = []
        train_scores_append = train_scores.append
        for (name, m) in fit_ests:
            for n in range(n_splits):
                dsk[(train_score, m, n)] = (score, (name, m, n), (X_train, n),
                                            (y_train, n), scorer)
                train_scores_append((train_score, m, n))
    else:
        train_scores = None

    cv_results = 'cv-results-' + main_token
    dsk[cv_results] = (create_cv_results, test_scores, train_scores,
                       candidate_params, n_splits, error_score, weights)
    keys = [cv_results]

    if refit:
        best_params = 'best-params-' + main_token
        dsk[best_params] = (get_best_params, candidate_params, cv_results)
        best_estimator = 'best-estimator-' + main_token
        dsk[best_estimator] = (fit_best, clone(estimator), best_params,
                               X_name, y_name)
        keys.append(best_estimator)

    return dsk, keys, n_splits


def normalize_params(params):
    """Take a list of dictionaries, and tokenize/normalize."""
    # Collect a set of all fields
    fields = set()
    for p in params:
        fields.update(p)
    fields = sorted(fields)

    params2 = list(pluck(fields, params, MISSING))
    # Non-basic types (including MISSING) are unique to their id
    tokens = [tuple(x if isinstance(x, (int, float, str)) else id(x)
                    for x in p) for p in params2]

    return fields, tokens, params2


def do_fit(dsk, next_token, est, fields, tokens, params, Xs, ys, n_splits,
           error_score):
    if isinstance(est, Pipeline) and params is not None:
        return _do_pipeline(dsk, next_token, est, fields, tokens, params,
                            Xs, ys, n_splits, error_score, False)
    else:
        token = next_token(est)
        fit_name = '%s-fit-%s' % (type(est).__name__.lower(), token)
        seen = {}
        m = 0
        out = []
        out_append = out.append
        if params is None:
            for X, y in zip(Xs, ys):
                if (X, y) in seen:
                    out_append(seen[X, y])
                else:
                    for n in range(n_splits):
                        dsk[(fit_name, m, n)] = (fit, est,
                                                 X + (n,), y + (n,),
                                                 error_score)
                    seen[(X, y)] = (fit_name, m)
                    out_append((fit_name, m))
                    m += 1
        else:
            for X, y, t, p in zip(Xs, ys, tokens, params):
                if (X, y, t) in seen:
                    out_append(seen[X, y, t])
                else:
                    for n in range(n_splits):
                        dsk[(fit_name, m, n)] = (fit, est, X + (n,), y + (n,),
                                                 error_score, fields, p)
                    seen[(X, y, t)] = (fit_name, m)
                    out_append((fit_name, m))
                    m += 1
        return out


def do_fit_transform(dsk, next_token, est, fields, tokens, params, Xs, ys,
                     n_splits, error_score):
    if isinstance(est, Pipeline) and params is not None:
        return _do_pipeline(dsk, next_token, est, fields, tokens, params,
                            Xs, ys, n_splits, error_score, True)
    else:
        name = type(est).__name__.lower()
        token = next_token(est)
        fit_Xt_name = '%s-fit-transform-%s' % (name, token)
        fit_name = '%s-fit-%s' % (name, token)
        Xt_name = '%s-transform-%s' % (name, token)
        seen = {}
        m = 0
        out = []
        out_append = out.append
        if params is None:
            for X, y in zip(Xs, ys):
                if (X, y) in seen:
                    out_append(seen[X, y])
                else:
                    for n in range(n_splits):
                        dsk[(fit_Xt_name, m, n)] = (fit_transform, est,
                                                    X + (n,), y + (n,),
                                                    error_score)
                        dsk[(fit_name, m, n)] = (getitem, (fit_Xt_name, m, n), 0)
                        dsk[(Xt_name, m, n)] = (getitem, (fit_Xt_name, m, n), 1)
                    seen[(X, y)] = m
                    out_append(m)
                    m += 1
        else:
            for X, y, t, p in zip(Xs, ys, tokens, params):
                if (X, y, t) in seen:
                    out_append(seen[X, y, t])
                else:
                    for n in range(n_splits):
                        dsk[(fit_Xt_name, m, n)] = (fit_transform, est,
                                                    X + (n,), y + (n,),
                                                    error_score, fields, p)
                        dsk[(fit_name, m, n)] = (getitem, (fit_Xt_name, m, n), 0)
                        dsk[(Xt_name, m, n)] = (getitem, (fit_Xt_name, m, n), 1)
                    seen[X, y, t] = m
                    out_append(m)
                    m += 1
        return [(fit_name, i) for i in out], [(Xt_name, i) for i in out]


def _do_pipeline(dsk, next_token, est, fields, tokens, params, Xs, ys,
                 n_splits, error_score, is_transform):
    if 'steps' in fields:
        raise NotImplementedError("Setting Pipeline.steps in a gridsearch")

    # Group the fields into a mapping of {stepname: [(newname, orig_index)]}
    field_to_index = dict(zip(fields, range(len(fields))))
    step_fields_lk = {s: [] for s, _ in est.steps}
    for f in fields:
        if '__' in f:
            step, param = f.split('__', 1)
            step_fields_lk[step].append((param, field_to_index[f]))
        elif f not in step_fields_lk:
            raise ValueError("Unknown parameter: `%s`" % f)

    # A list of (step, is_transform)
    instrs = [(s, True) for s in est.steps[:-1]]
    instrs.append((est.steps[-1], is_transform))

    fit_steps = []
    for (step_name, step), transform in instrs:
        sub_fields, sub_inds = map(list, unzip(step_fields_lk[step_name], 2))

        if step_name in field_to_index:
            # The estimator may change each call
            new_fits = {}
            new_Xs = {}
            est_index = field_to_index[step_name]
            id_groups = defaultdict(lambda: [].append)
            for n, step_token in enumerate(pluck(est_index, tokens)):
                id_groups[step_token](n)
            for ids in (i.__self__ for i in id_groups.values()):
                # Get the estimator for this subgroup
                sub_est = params[ids[0]][est_index]
                if sub_est is MISSING:
                    sub_est = step

                # If an estimator is `None`, there's nothing to do
                if sub_est is None:
                    new_fits.update(dict.fromkeys(ids, None))
                    if transform:
                        new_Xs.update(zip(ids, get(ids, Xs)))
                else:
                    # Extract the proper subset of Xs, ys
                    sub_Xs = get(ids, Xs)
                    sub_ys = get(ids, ys)
                    # Only subset the parameters/tokens if necessary
                    if sub_fields:
                        sub_tokens = list(pluck(sub_inds, get(ids, tokens)))
                        sub_params = list(pluck(sub_inds, get(ids, params)))
                    else:
                        sub_tokens = sub_params = None

                    if transform:
                        sub_Xs, sub_fits = do_fit_transform(dsk, next_token, sub_est,
                                                            sub_fields, sub_tokens,
                                                            sub_params, sub_Xs,
                                                            sub_ys, n_splits,
                                                            error_score)
                        new_Xs.update(zip(ids, sub_Xs))
                        new_fits.update(zip(ids, sub_fits))
                    else:
                        sub_fits = do_fit(dsk, next_token, sub_est, sub_fields,
                                          sub_tokens, sub_params, sub_Xs,
                                          sub_ys, n_splits, error_score)
                        new_fits.update(zip(ids, sub_fits))
            # Extract lists of transformed Xs and fit steps
            all_ids = list(range(len(Xs)))
            if transform:
                Xs = get(all_ids, new_Xs)
            fits = get(all_ids, new_fits)
        elif step is None:
            # Nothing to do
            fits = [None] * len(Xs)
        else:
            # Only subset the parameters/tokens if necessary
            if sub_fields:
                sub_tokens = list(pluck(sub_inds, tokens))
                sub_params = list(pluck(sub_inds, params))
            else:
                sub_tokens = sub_params = None

            if transform:
                fits, Xs = do_fit_transform(dsk, next_token, step, sub_fields,
                                            sub_tokens, sub_params, Xs, ys,
                                            n_splits, error_score)
            else:
                fits = do_fit(dsk, next_token, step, sub_fields, sub_tokens,
                              sub_params, Xs, ys, n_splits, error_score)
        fit_steps.append(fits)

    # Rebuild the pipelines
    step_names = [n for n, _ in est.steps]
    out_ests = []
    out_ests_append = out_ests.append
    name = 'pipeline-' + next_token(est)
    m = 0
    seen = {}
    for steps in zip(*fit_steps):
        if steps in seen:
            out_ests_append(seen[steps])
        else:
            for n in range(n_splits):
                dsk[(name, m, n)] = (pipeline, step_names,
                                     [None if s is None else s + (n,)
                                      for s in steps])
            seen[steps] = (name, m)
            out_ests_append((name, m))
            m += 1

    if is_transform:
        return out_ests, Xs
    return out_ests


# ------------ #
# CV splitting #
# ------------ #

def check_cv(cv=3, y=None, classifier=False):
    """Dask aware version of ``sklearn.model_selection.check_cv``

    Same as the scikit-learn version, but works if ``y`` is a dask object.
    """
    if cv is None:
        cv = 3

    # If ``cv`` is not an integer, the scikit-learn implementation doesn't
    # touch the ``y`` object, so passing on a dask object is fine
    if not isinstance(y, Base) or not isinstance(cv, numbers.Integral):
        return _sklearn_check_cv(cv, y, classifier)

    if classifier:
        # ``y`` is a dask object. We need to compute the target type
        target_type = delayed(type_of_target, pure=True)(y).compute()
        if target_type in ('binary', 'multiclass'):
            return StratifiedKFold(cv)
    return KFold(cv)


def initialize_graph(cv, X, y=None, groups=None, is_pairwise=False, iid=True):
    """Initialize the CV dask graph.

    Given input data X, y, and groups, build up a dask graph performing the
    initial CV splits.

    Parameters
    ----------
    cv : BaseCrossValidator
    X, y, groups : array_like, dask object, or None
    is_pairwise : bool
        Whether the estimator being evaluated has ``_pairwise`` as an
        attribute (which affects how the CV splitting is done).
    """
    dsk = {}
    X_name, y_name, groups_name = to_keys(dsk, X, y, groups)
    n_splits = compute_n_splits(cv, X, y, groups)

    cv_token = tokenize(cv, X_name, y_name, groups_name, is_pairwise)
    cv_name = 'cv-split-' + cv_token
    train_name = 'cv-split-train-' + cv_token
    test_name = 'cv-split-test-' + cv_token

    dsk[cv_name] = (cv_split, cv, X_name, y_name, groups_name)

    if iid:
        weights = 'cv-n-samples-' + cv_token
        dsk[weights] = (cv_n_samples, cv_name)
    else:
        weights = None

    for n in range(n_splits):
        dsk[(cv_name, n)] = (getitem, cv_name, n)
        dsk[(train_name, n)] = (getitem, (cv_name, n), 0)
        dsk[(test_name, n)] = (getitem, (cv_name, n), 1)

    # Extract the test-train subsets
    Xy_train = 'Xy-train-' + cv_token
    X_train = 'X-train-' + cv_token
    y_train = 'y-train-' + cv_token
    Xy_test = 'Xy-test-' + cv_token
    X_test = 'X-test-' + cv_token
    y_test = 'y-test-' + cv_token

    # Build a helper function to insert the extract tasks
    if is_pairwise:
        def extract_train(X, y, train, test):
            return (cv_extract_pairwise, X, y, train, train)

        def extract_test(X, y, train, test):
            return (cv_extract_pairwise, X, y, test, train)
    else:
        def extract_train(X, y, train, test):
            return (cv_extract, X, y, train)

        def extract_test(X, y, train, test):
            return (cv_extract, X, y, test)

    for n in range(n_splits):
        dsk[(Xy_train, n)] = extract_train(X_name, y_name, (train_name, n),
                                           (test_name, n))
        dsk[(X_train, n)] = (getitem, (Xy_train, n), 0)
        dsk[(y_train, n)] = (getitem, (Xy_train, n), 1)
        dsk[(Xy_test, n)] = extract_test(X_name, y_name, (train_name, n),
                                         (test_name, n))
        dsk[(X_test, n)] = (getitem, (Xy_test, n), 0)
        dsk[(y_test, n)] = (getitem, (Xy_test, n), 1)

    return (dsk, n_splits,
            X_name, y_name,
            X_train, y_train,
            X_test, y_test,
            weights)


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
