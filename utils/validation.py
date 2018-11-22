"""Utilities for input validation"""


import pandas as pd 


from pandas.api.types import is_numeric_dtype


def check_uniq(X):
    """Checks whether all input data values are unique.

    Parameters
    ----------
    X: array-like, shape (n_samples, )
        Vector to check whether it cointains unique values. 
    
    Returns
    -------
    bool

    """
    s = set()
    return not any(x in s or s.add(x) for x in X)


def check_column_existance(X, cols):
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
    X: array-like, shape (n_samples, )
        Vector where n_samples is the number of samples.

    Returns
    -------
    bool

    """
    return is_numeric_dtype(X) and not set(X) <= {0, 1}