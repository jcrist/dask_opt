from __future__ import division, print_function, absolute_import

from functools import wraps

from sklearn.externals.funcsigs import signature, Parameter


class DaskEstimator(object):
    """Base class for dask estimators"""
    pass


extra_kw_doc = """
    dask_solver : str, optional
        Dask solution method to use. Options are {0} with a default of '{1}'.
"""


def wrap_model(model, solvers, default_solver):
    # Derive the init signature from the model
    init_sig = signature(model)
    params = init_sig.parameters.values()
    params.append(Parameter('dask_solver', Parameter.POSITIONAL_OR_KEYWORD,
                  default_solver))
    init_sig = init_sig.replace(parameters=params)
    # Extract defaults for the init function
    defaults = dict((k, v.default) for (k, v) in init_sig.parameters.items())
    # Compose a new class docstring
    top, attributes, bottom = model.__doc__.partition('    Attributes')
    solver_strings = "'{0}'".format("', '".join(solvers))
    doc = '\n'.join([top.strip(),
                     extra_kw_doc.format(solver_strings, default_solver),
                     attributes + bottom])

    @wraps(model.fit)
    def fit(self, X, y, **kwargs):
        try:
            solver = solvers[self.dask_solver]
        except KeyError:
            raise KeyError("Unsupported solver: {0}".format(self.dask_solver))

        return solver(self, X, y, **kwargs)

    def __init__(self, **kwargs):
        extra = set(kwargs).difference(defaults)
        if extra:
            raise TypeError("Unexpected keyword arguments: {0}".format(extra))
        for k, v in defaults.items():
            setattr(self, k, kwargs.get(k, v))

    __init__.__signature__ = init_sig

    dct = {'__doc__': doc,
           '__init__': __init__,
           '_estimator': model,
           'fit': fit}
    return type(model.__name__, (model, DaskEstimator), dct)
