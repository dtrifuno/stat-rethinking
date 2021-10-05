import arviz as az
import numpy as np
import pandas as pd


def pi(samples, prob=0.89) -> pd.Series:
    a_lower = (1 - prob) / 2
    a_upper = (1 + prob) / 2

    index = [f"{x * 100:.3f}".rstrip("0").rstrip(".") + "%" for x in (a_lower, a_upper)]

    return pd.Series(data=np.quantile(samples, (a_lower, a_upper)), index=index)


def hdpi(samples, prob=0.89) -> pd.Series:
    a_lower = (1 - prob) / 2
    a_upper = (1 + prob) / 2

    index = [f"{x * 100:.3f}".rstrip("0").rstrip(".") + "%" for x in (a_lower, a_upper)]
    return pd.Series(data=az.hdi(samples, hdi_prob=prob), index=index)
