from typing import Union

import numpy as np
import scipy.stats as stats
import pandas as pd

from .quap import QuapResult

SupportedTypes = Union[QuapResult, pd.DataFrame]


def precis(result, ci_prob: float = 0.89, precision: int = 2) -> None:
    if isinstance(result, QuapResult):
        return precis_quap_result(result, ci_prob, precision)
    elif isinstance(result, pd.DataFrame):
        return precis_df(result, ci_prob, precision)


def precis_quap_result(result: QuapResult, ci_prob: float, precision: int) -> None:
    mean = result.mean
    std = np.sqrt(np.diag(result.cov))
    z = stats.norm.ppf(0.5 * (1 + ci_prob))
    left = mean - z * std
    right = mean + z * std

    lower_percentile = f"{0.5 * (1 - ci_prob) * 100:g}%"
    upper_percentile = f"{0.5 * (1 + ci_prob) * 100:g}%"

    df = pd.DataFrame(
        {"mean": mean, "std": std, lower_percentile: left, upper_percentile: right},
        index=result.vars,
    )
    with pd.option_context("display.precision", precision):
        print(df)


def precis_df(result: pd.DataFrame, ci_prob: float, precision: int) -> None:
    a_lower = (1 - ci_prob) / 2
    a_upper = (1 + ci_prob) / 2

    df = result.describe(percentiles=(a_lower, a_upper)).transpose()
    df.drop(["count", "min", "max", "50%"], axis=1, inplace=True)
    with pd.option_context("display.precision", precision):
        print(df)

    _ = result.hist()
