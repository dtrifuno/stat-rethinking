from collections import namedtuple

import numpy as np
import scipy.stats as stats
import pandas as pd
import pymc3 as pm

QuapResult = namedtuple("QuapResult", ["vars", "mean", "cov"])


def quap(vars=None, start=None, model=None):
    from pymc3.util import is_transformed_name

    if model is None:
        try:
            from pymc3.model import Model

            model = Model.get_context()
        except TypeError:
            raise TypeError(
                "No PyMC3 model supplied nor present on context stack. "
                "Use quap in a 'with pm.Model():' block, or pass a model "
                "argument."
            )

    if vars is None:
        vars = [
            var for var in model.unobserved_RVs if not is_transformed_name(var.name)
        ]

    mean = pm.find_MAP(vars=vars, start=start, model=model)
    hess = pm.find_hessian(mean, vars=vars, model=model)
    cov = np.linalg.inv(hess)

    vars_names = [var.name for var in vars]
    mean_array = np.array([mean[var_name] for var_name in vars_names])
    return QuapResult(vars=vars_names, mean=mean_array, cov=cov)


def cov_to_cor(covariance_matrix):
    v = np.sqrt(np.diag(covariance_matrix))
    outer_v = np.outer(v, v)
    correlation = covariance_matrix / outer_v
    correlation[covariance_matrix == 0] = 0
    return correlation


def extract_samples(result, n, random_state=None):
    mean = result.mean
    cov = result.cov
    samples = stats.multivariate_normal.rvs(
        mean=mean, cov=cov, size=n, random_state=random_state
    )
    as_dict = {}
    for idx, var in enumerate(result.vars):
        as_dict[var] = samples[:, idx]
    return pd.DataFrame(as_dict)


def link(result, fn_dict, data, n=10_000, random_state=None):
    params = extract_samples(result, n, random_state=random_state)
    df = params.merge(data, how="cross")
    for fn_name, fn in fn_dict.items():
        df[fn_name] = fn(df)
    return df
