"""Utilities for input validation."""


import numpy as np
import pandas as pd

from pandas.api.types import is_categorical_dtype, is_numeric_dtype


__all__ = [
    "check_uniq",
    "check_column_existence",
    "check_is_dataframe",
    "check_is_series",
    "is_numeric",
    "find_sparsity",
    "check_continuity"
]


def check_uniq(series):
    """Check whether all input data values are unique.

    Parameters
    ----------
    series: array-like, shape = (n_samples, )
        Vector to check whether it cointains unique values.

    Returns
    -------
    boolean:
        Whether all input data values are unique.

    """
    unique = set()
    return not any(elem in unique or unique.add(elem) for elem in series)


def check_column_existence(df, columns):
    """Check whether all listed columns are in a given DataFrame.

    Parameters
    ----------
    df: pandas.DataFrame
        Data with columns to be checked for occurrence.

    columns: single label or list-like
        Columns' labels to check.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If one of the elements of `columns` is not found in the `df` columns.

    """
    if isinstance(columns, str):
        columns = [columns]

    exist = all(col in df.columns for col in columns)

    if not exist:
        cols_error = list(set(columns) - set(df.columns))
        raise ValueError(
            "Columns not found in the DataFrame: {}"
            .format(", ".join(cols_error))
        )


def check_is_dataframe(df):
    """Check whether object is a pandas.DataFrame.

    Parameters
    ----------
    df: object
        Object suspected of being a pandas.DataFrame.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If object is not a pandas.DataFrame.

    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be an instance of pandas.DataFrame.")


def check_is_series(series):
    """Check whether object is a pandas.Series or can be project onto one.

    Parameters
    ----------
    series: object
        Object suspected of being a pandas.Series.

    Returns
    -------
    series: pandas.Series
        Input object projected onto pandas.Series.

    Raises
    ------
    TypeError
        If object is not a pandas.Series and cannot be projected onto one.

    """
    try:
        series = pd.Series(series)
    except Exception as err:
        raise TypeError(
            "Input is not an instance of pandas.Series "
            "and cannot be projected onto one of them."
        ).with_traceback(err.__traceback__)

    return series


def is_numeric(series, project=True):
    """Checks whether given vector contains numeric-only values excluding
    boolean vectors.

    Parameters
    ----------
    series: array-like, shape = (n_samples, )
        Vector where n_samples is the number of samples.

    project: bool, optional (default=True)
        If True tries to project on a numeric type unless categorical dtype is
        passed.

    Returns
    -------
    boolean:
        Whether series is numeric or can be projected to numeric if ``project``
        is set to True.

    """
    if project and not is_categorical_dtype(series):
        try:
            series = np.array(series).astype(np.number)
        except ValueError:
            return False

    return is_numeric_dtype(series) and not set(series) <= {0, 1}


def find_sparsity(df, thresh=.01):
    """Finds columns with highly sparse categories.

    For categorical and binary features finds columns where categories with
    relative frequencies under the threshold are present.

    For numerical features (excluding binary variables) returns columns
    where NaNs or 0 are dominating in the given dataset.

    Parameters
    ----------
    df: pandas.DataFrame
        Data to be checked for sparsity.

    thresh: float, optional (default=.01)
        Fraction of one of the categories under which the sparseness will be
        reported.

    Returns
    -------
    sparse_{num, bin, cat}: list
        List of {numerical, binary, categorical} df column names where high
        sparsity was detected.

    """
    assert isinstance(df, pd.DataFrame), \
        'Input must be an instance of pandas.DataFrame()'
    assert len(df) > 0, 'Input data can not be empty!'

    sparse_num, sparse_bin, sparse_cat = [[] for _ in range(3)]

    for col in df.columns:
        tab_counter = df[col].value_counts(normalize=True, dropna=False)
        if is_numeric(df[col]):
            most_freq = tab_counter.index[0]
            if most_freq != most_freq or most_freq == 0:
                sparse_num.append(col)
        else:
            min_frac = tab_counter.iloc[-1]
            if min_frac < thresh:
                if set(df[col]) <= {0, 1}:
                    sparse_bin.append(col)
                else:
                    sparse_cat.append(col)

    return sparse_num, sparse_bin, sparse_cat


def check_continuity(series, thresh=.5):
    """Checks whether input variable is continuous.

    Parameters
    ----------
    series: array-like, shape = (n_samples, )
        Vector to check for continuity.

    thresh: float, optional (default=.5)
        Fraction of non-unique values under which lack of continuity will be
        reported.

    Returns
    -------
    boolean: Whether variable is continuous.

    """
    numerator = len(np.unique(series))
    denominator = len(series) >= 1 - thresh
    return is_numeric(series) and numerator / denominator
