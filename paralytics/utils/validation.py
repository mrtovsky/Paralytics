"""Utilities for input validation."""


import pandas as pd

from pandas.api.types import is_numeric_dtype


__all__ = [
    'check_uniq',
    'check_column_existence',
    'is_numeric',
    'find_sparsity'
]


def check_uniq(X):
    """Checks whether all input data values are unique.

    Parameters
    ----------
    X: array-like, shape = (n_samples, )
        Vector to check whether it cointains unique values.

    Returns
    -------
    boolean: Whether or not all input data values are unique.

    """
    s = set()
    return not any(x in s or s.add(x) for x in X)


def check_column_existence(X, cols):
    """Checks whether all listed columns are in a given DataFrame.

    Parameters
    ----------
    X: DataFrame
        Data with columns to be checked for occurrence.

    cols: single label or list-like
        Column labels to check.

    Returns
    -------
    boolean: Whether or not X contains all of given column labels.

    """
    assert isinstance(X, pd.DataFrame), \
        'Input must be an instance of pandas.DataFrame()'

    if isinstance(cols, str):
        cols = [cols]

    out = all(col in X.columns for col in cols)

    if not out:
        cols_error = list(set(cols) - set(X.columns))
        print('Columns not found in the DataFrame: %s' % cols_error)

    return out


def is_numeric(X):
    """Checks whether given vector contains numeric-only values excluding
    boolean vectors.

    Parameters
    ----------
    X: array-like, shape = (n_samples, )
        Vector where n_samples is the number of samples.

    Returns
    -------
    bool

    """
    return is_numeric_dtype(X) and not set(X) <= {0, 1}


def find_sparsity(X, thresh=.01):
    """Finds columns with highly sparse categories.

    For categorical and binary features finds columns where categories with
    relative frequencies under the threshold are present.

    For numerical features (excluding binary variables) returns columns
    where NaNs or 0 are dominating in the given dataset.

    Parameters
    ----------
    X: DataFrame
        Data to be checked for sparsity.

    thresh: float, optional (default=.01)
        Fraction of one of the categories under which the sparseness will be
        reported.

    Returns
    -------
    sparse_{num, bin, cat}: list
        List of {numerical, binary, categorical} X column names where high
        sparsity was detected.

    """
    assert isinstance(X, pd.DataFrame), \
        'Input must be an instance of pandas.DataFrame()'
    assert len(X) > 0, 'Input data can not be empty!'

    sparse_num, sparse_bin, sparse_cat = [[] for _ in range(3)]

    for col in X.columns:
        tab_counter = X[col].value_counts(normalize=True, dropna=False)
        if is_numeric(X[col]):
            most_freq = tab_counter.index[0]
            if most_freq != most_freq or most_freq == 0:
                sparse_num.append(col)
        else:
            min_frac = tab_counter.iloc[-1]
            if min_frac < thresh:
                if set(X[col]) <= {0, 1}:
                    sparse_bin.append(col)
                else:
                    sparse_cat.append(col)

    return sparse_num, sparse_bin, sparse_cat
