"""Utilities for drawing plots"""


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve


def multipleplots(data, x, nrows=1, ncols=1, figsize=(8, 6),
                  plot_type='scatterplot', **params):
    """Paints any specified plot from the seaborn library multiple times.

    Parameters
    ----------
    data: DataFrame
        Data which columns will be plotted.

    x: list
        List of columns to plot one after another.

    n{rows, cols}: int (default: 1)
        Number of {rows, cols} transferred to plt.subplots().

    figsize: list-like
        Size of every figure created on subsequent axes.

    plot_type: string
        Specific plot from the seaborn library to be plotted multiple times
        against every given element from x.

    params: dict
        Dictionary of optional parameters passed to the chosen seaborn plot.

    Returns
    -------
    None

    """
    assert isinstance(data, pd.DataFrame), \
        'Input "data" must be an instance of pandas.DataFrame()!'
    assert isinstance(x, list), \
        'Input "x" must be the list of data columns to plot!'

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()
    for idx, ax in enumerate(axes):
        plt.sca(ax)
        try:
            getattr(sns, plot_type)(x[idx], data=data, **params)
        except AttributeError:
            print('Consider changing the "plot_type" parameter.')
            raise
