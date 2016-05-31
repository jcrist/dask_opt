from __future__ import division, print_function, absolute_import


def copy_to(source, target):
    """Copy attributes of source estimator to target estimator."""
    source_dict = vars(source)
    source_params = source.get_params(deep=False)
    for k, v in source_dict.items():
        if k not in source_params:
            setattr(target, k, v)
    return target
