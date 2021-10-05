from typing import Any, Callable, Dict, Optional

import matplotlib.axes
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats


def _create_set_args(
    xlabel: Optional[str], ylabel: Optional[str], title: Optional[str]
) -> Dict[str, str]:
    set_args = {}
    if xlabel:
        set_args["xlabel"] = xlabel
    if ylabel:
        set_args["ylabel"] = ylabel
    if title:
        set_args["title"] = title
    return set_args


def curve(
    fn: Callable,
    start: float,
    end: float,
    *,
    num: int = 101,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    **kwargs
) -> matplotlib.axes.Axes:
    """Plots a curve corresponding to the graph of a function over an interval.

    Args:
        fn: The function to graph.
        start: The left endpoint of the interval.
        end: The right endpoint of the interval.
        num: Number of places on the interval on which to evaluate the function.
        xlabel: Label text for horizontal axis.
        ylabel: Label text for vertical axis.
        title: Text to use for the title.
        ax: Pre-existing axes for the plot. Otherwise, calls matplotlib.pyplot.gca().
        **kwargs: Arbitrary keyword arguments. These will be passed to seaborn.lineplot.

    Returns:
        The matplotlib axes containing the plot.
    """
    x = np.linspace(start, end, num)
    y = np.vectorize(fn)(x)
    ax = sns.lineplot(x=x, y=y, ax=ax, **kwargs)

    set_args = _create_set_args(xlabel, ylabel, title)
    if set_args:
        ax.set(**set_args)

    return ax


def plot(
    x: Any,
    y: Any,
    *,
    data: Optional[pd.DataFrame] = None,
    plot_fn: Callable = sns.lineplot,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    **kwargs
) -> matplotlib.axes.Axes:
    """FIXME

    Args:
        x, y: Values that specify positions on the x and y axes.
        data: Data for x and y in data frame format. If used, the x and y parameters should instead be keys in the data frame.
        xlabel: Label text for horizontal axis.
        ylabel: Label text for vertical axis.
        title: Text to use for the title.
        ax: Pre-existing axes for the plot. Otherwise, calls matplotlib.pyplot.gca().
        **kwargs: Arbitrary keyword arguments. These will be passed to plot_fn.

    Returns:
        The matplotlib axes containing the plot.
    """
    ax = plot_fn(x=x, y=y, data=data, ax=ax, **kwargs)

    set_args = _create_set_args(xlabel, ylabel, title)
    if set_args:
        ax.set(**set_args)

    return ax


def dens(
    data: Any,
    norm_comp: bool = False,
    *,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = "density",
    title: Optional[str] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    **kwargs
) -> matplotlib.axes.Axes:
    """Convenience function for plotting density estimates from data.

    Args:
        data: Collection of values from which to construct density.
        norm_comp: If True, overlays a fitted normal density for comparison.
        xlabel: Label text for horizontal axis.
        ylabel: Label text for vertical axis.
        title: Text to use for the title.
        ax: Pre-existing axes for the plot. Otherwise, calls matplotlib.pyplot.gca().
        **kwargs: Arbitrary keyword arguments. These will be passed to seaborn.kdeplot.

    Returns:
        The matplotlib axes containing the plot.
    """
    if kwargs.get("label") is None and norm_comp:
        kwargs["label"] = "empirical distribution"

    ax = sns.kdeplot(data, ax=ax, alpha=0.9, **kwargs)
    if norm_comp:
        start, end = ax.get_xlim()
        ax = curve(
            lambda x: stats.norm.pdf(x, np.mean(data), np.std(data)),
            start=start,
            end=end,
            linestyle="dashed",
            linewidth=0.75,
            ax=ax,
            label="normal approximation",
        )
        ax.legend(loc="upper left")

    set_args = _create_set_args(xlabel, ylabel, title)
    if set_args:
        ax.set(**set_args)

    return ax


def simplehist(
    data,
    round: bool = True,
    *,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = "frequency",
    title: Optional[str] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    **kwargs
) -> matplotlib.axes.Axes:
    """Plots a simple integer-valued histogram for displaying count distributions.

    Args:
        data: Collection of values to construct histogram from.
        round: If True, round values from data before plotting.
        xlabel: Label text for horizontal axis.
        ylabel: Label text for vertical axis.
        title: Text to use for the title.
        ax: Pre-existing axes for the plot. Otherwise, calls matplotlib.pyplot.gca().
        **kwargs: Arbitrary keyword arguments. These will be passed to seaborn.kdeplot.

    Returns:
        The matplotlib axes containing the plot.
    """
    if round:
        data = np.round(data)

    ax = sns.histplot(data, discrete=True, ax=ax, **kwargs)

    set_args = _create_set_args(xlabel, ylabel, title)
    if set_args:
        ax.set(**set_args)

    return ax
