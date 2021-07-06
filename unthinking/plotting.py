import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


def curve(
    fn, start, end, n=101, xlabel=None, ylabel=None, title=None, ax=None, **kwargs
):
    """Draws a curve corresponding to a function over an interval.

    Args:
        fn: The function to graph.
        start: The left endpoint of the interval.
        end: The right endpoint of the interval.

    """
    ax = plt.gca() if ax is None else ax
    x = np.linspace(start, end, n)
    y = np.vectorize(fn)(x)
    l = sns.lineplot(x=x, y=y, ax=ax, **kwargs)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    return l


def plot(
    x,
    y,
    data=None,
    plot_fn=sns.lineplot,
    xlabel=None,
    ylabel=None,
    title=None,
    ax=None,
    **kwargs
):
    ax = plt.gca() if ax is None else ax
    l = plot_fn(x=x, y=y, data=data, ax=ax, **kwargs)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    return l


def dens(
    data, xlabel=None, ylabel="density", title=None, ax=None, norm_comp=False, **kwargs
):
    """Convenience function for plotting density estimates from data.

    Args:
        data: Collection of values to construct density from.
    """
    ax = plt.gca() if ax is None else ax

    l1 = sns.kdeplot(data, ax=ax, alpha=0.9, **kwargs, label="empirical distribution")
    if norm_comp:
        l2 = curve(
            lambda x: stats.norm.pdf(x, np.mean(data), np.std(data)),
            start=np.min(data),
            end=np.max(data),
            linestyle="dashed",
            linewidth=0.75,
            ax=ax,
            label="normal approximation",
        )
        ax.legend(loc="upper left")

    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
