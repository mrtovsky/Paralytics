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


def plot_learning_curve(
    estimator, title, X, y, ylim=None, cv=None, scoring=None,
    n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), random_state=None
):
    """Generate a simple plot of the test and training learning curve.

    Based on:
    https://scikit-learn.org/stable/auto_examples/model_selection/
    plot_learning_curve.html

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure(figsize=(12, 8))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs,
        train_sizes=train_sizes, random_state=random_state)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
