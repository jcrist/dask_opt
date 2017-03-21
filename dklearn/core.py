from __future__ import absolute_import, division, print_function

from operator import getitem
from collections import defaultdict
from cytoolz import get, pluck

from sklearn.pipeline import Pipeline

from .methods import fit, fit_transform, pipeline, MISSING


from itertools import count

next_token = iter(map(str, count())).__next__


def unzip(itbl, n):
    return map(list, zip(*itbl)) if itbl else [[] for i in range(n)]


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

    return tokens, params2, fields


def do_fit(dsk, est, fields, tokens, params, Xs, ys):
    if isinstance(est, Pipeline) and params is not None:
        return _do_pipeline(dsk, est, fields, tokens, params, Xs, ys, False)
    else:
        token = next_token()
        fit_name = '%s-fit-%s' % (type(est).__name__.lower(), token)
        seen = {}
        n = 0
        out = []
        out_append = out.append
        if params is None:
            for X, y in zip(Xs, ys):
                if (X, y) in seen:
                    out_append(seen[X, y])
                else:
                    dsk[(fit_name, n)] = (fit, est, X, y)
                    seen[(X, y)] = (fit_name, n)
                    out_append((fit_name, n))
                    n += 1
        else:
            for X, y, t, p in zip(Xs, ys, tokens, params):
                if (X, y) in seen:
                    out_append(seen[X, y, t])
                else:
                    dsk[(fit_name, n)] = (fit, est, X, y, fields, p)
                    seen[(X, y, t)] = (fit_name, n)
                    out_append((fit_name, n))
                    n += 1
        return out


def do_fit_transform(dsk, est, fields, tokens, params, Xs, ys):
    if isinstance(est, Pipeline) and params is not None:
        return _do_pipeline(dsk, est, fields, tokens, params, Xs, ys, True)
    else:
        name = type(est).__name__.lower()
        token = next_token()
        fit_Xt_name = '%s-fit-transform-%s' % (name, token)
        fit_name = '%s-fit-%s' % (name, token)
        Xt_name = '%s-transform-%s' % (name, token)
        seen = {}
        n = 0
        out = []
        out_append = out.append
        if params is None:
            for X, y in zip(Xs, ys):
                if (X, y) in seen:
                    out_append(seen[X, y])
                else:
                    dsk[(fit_Xt_name, n)] = (fit_transform, est, X, y)
                    dsk[(fit_name, n)] = (getitem, (fit_Xt_name, n), 0)
                    dsk[(Xt_name, n)] = (getitem, (fit_Xt_name, n), 1)
                    seen[(X, y)] = n
                    out_append(n)
                    n += 1
        else:
            for X, y, t, p in zip(Xs, ys, tokens, params):
                if (X, y, t) in seen:
                    out_append(seen[X, y, t])
                else:
                    dsk[(fit_Xt_name, n)] = (fit_transform, est, X, y, fields, p)
                    dsk[(fit_name, n)] = (getitem, (fit_Xt_name, n), 0)
                    dsk[(Xt_name, n)] = (getitem, (fit_Xt_name, n), 1)
                    seen[X, y, t] = n
                    out_append(n)
                    n += 1
        return [(fit_name, i) for i in out], [(Xt_name, i) for i in out]


def _do_pipeline(dsk, est, fields, tokens, params, Xs, ys, is_transform):
    if 'steps' in fields:
        raise NotImplementedError("Setting Pipeline.steps in a gridsearch")

    # Group the fields into a mapping of {stepname: [(newname, orig_index)]}
    field_to_index = dict(zip(fields, range(len(fields))))
    step_fields_lk = {s: [] for s, _ in est.steps}
    for f in fields:
        if '__' in f:
            step, param = f.split('__', 1)
            step_fields_lk[step].append((f, field_to_index[f]))
        elif f not in step_fields_lk:
            raise ValueError("Unknown parameter: `%s`" % f)

    # A list of (step, is_transform)
    instrs = [(s, True) for s in est.steps[:-1]]
    instrs.append((est.steps[-1], is_transform))

    fit_steps = []
    for (step_name, step), transform in instrs:
        sub_fields, sub_inds = unzip(step_fields_lk[step_name], 2)

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
                        sub_Xs, sub_fits = do_fit_transform(dsk, sub_est,
                                                            sub_fields, sub_tokens,
                                                            sub_params, sub_Xs,
                                                            sub_ys)
                        new_Xs.update(zip(ids, sub_Xs))
                        new_fits.update(zip(ids, sub_fits))
                    else:
                        sub_fits = do_fit(dsk, sub_est, sub_fields, sub_tokens,
                                          sub_params, sub_Xs, sub_ys)
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
                Xs, fits = do_fit_transform(dsk, step, sub_fields, sub_tokens,
                                            sub_params, Xs, ys)
            else:
                fits = do_fit(dsk, step, sub_fields, sub_tokens, sub_params,
                              Xs, ys)
        fit_steps.append(fits)

    # Rebuild the pipelines
    step_names = [n for n, _ in est.steps]
    out_ests = []
    out_ests_append = out_ests.append
    name = 'pipeline-' + next_token()
    n = 0
    seen = {}
    for steps in zip(*fit_steps):
        if steps in seen:
            out_ests_append(seen[steps])
        else:
            dsk[(name, n)] = (pipeline, step_names, list(steps))
            seen[steps] = (name, n)
            out_ests_append((name, n))
            n += 1

    if is_transform:
        return out_ests, Xs
    return out_ests
