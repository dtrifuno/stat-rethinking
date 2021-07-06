import numpy as np
import scipy.stats as stats
import pandas as pd


def precis(result, ci_prob=0.89, precision=2):
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
