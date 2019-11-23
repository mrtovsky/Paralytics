"""Utilities to facilitate the storage of data files."""


import numpy as np
import pandas as pd

from . import check_is_dataframe


__all__ = [
    "downcast_dataframe"
]


def downcast_dataframe(df, tolerance=1e-5, errors="ignore", category_thresh=.3):
    """Downcast pandas.DataFrame's columns to the smallest possible types.

    Reduce memory usage by applying downcasting to every column and for
    numerical columns, that could have been downcasted to integer dtype, unless
    NaN values were present, fills missing values with the smallest column
    value - 1 and returns a mapping dictionary that let's you revert the
    changes by simply calling `pandas.DataFrame.replace
    <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html>`_
    as follows:
    >>> import numpy as np
    >>> import pandas as pd
    >>>
    >>> from paralytics.utils import downcast_dataframe
    >>>
    >>>
    >>> df = pd.DataFrame(...) # DataFrame to call the downcasting on.
    >>> # Downcast the df dtypes.
    >>> df_dc, nan_mapping = downcast_dataframe(df)
    >>> # Revert filling NaN values for int dtypes.
    >>> df_rev = df_dc.replace(nan_mapping, np.nan)

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame designed to downcasting dtypes.

    tolerance: float, optional (default=1e-5)
        Reference value below which the total absolute column difference
        between the base numeric values and the corresponding values projected
        onto integers will not be counted as a significant difference and hence
        projected onto ``integer`` dtype.

    errors: string {ignore, raise, coerce}, optional (default="ignore")
        Methods of handling errors occured when trying to downcast a numeric
        value. Passed to `pandas.to_numeric
        <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_numeric.html>`_
        function's parameter: ``errors``.

    category_thresh: float, optional (default=.3)
        Reference value below which the ratio of unique observations of the
        ``object`` dtype column to its length will be counted as low granulated
        and hence projected onto ``category`` dtype.

    Returns
    -------
    df_dc: pandas.DataFrame
        Downcasted input DataFrame.

    nan_mapping: dictionary
        Mapping to revert missing values imputation with use of
        pandas.DataFrame.replace method.

    Raises
    ------
    TypeError
        If ``df`` is not a pandas.DataFrame.

    """
    check_is_dataframe(df)

    df_num = df.select_dtypes(include=np.number).copy()

    if df_num.shape[1]:
        df_num_nan_count = df_num.isnull().sum()
        num_with_nan_columns = df_num_nan_count.index[
            df_num_nan_count > 0
        ].tolist()

        # In the presence of NaN values only in the column, fill it with -1.
        _nan_filler = (df_num.min() - 1).fillna(-1)
        _nan_mapping = _nan_filler.to_dict()

        df_num.fillna(_nan_filler, inplace=True)
        df_num_int = df_num.astype(np.int64)

        # Find float columns that can be expressed as integers.
        df_num_diff = (df_num - df_num_int).abs().sum()
        int_columns = df_num_diff.index[df_num_diff < tolerance].tolist()

        nan_mapping = {
            column: int(value) for column, value in _nan_mapping.items()
            if column in set(int_columns).intersection(num_with_nan_columns)
        }

        df_dc = df.copy()
        for column in int_columns:
            series_int = df_num_int[column]
            if series_int.min() >= 0:
                df_dc[column] = pd.to_numeric(
                    series_int, errors=errors, downcast="unsigned"
                )
            else:
                df_dc[column] = pd.to_numeric(
                    series_int, errors=errors, downcast="signed"
                )
        for column in list(set(df_num.columns).difference(int_columns)):
            df_dc[column] = pd.to_numeric(
                df_dc[column], errors=errors, downcast="float"
            )

    df_obj = df.select_dtypes(include="object").copy()

    if df_obj.shape[1]:
        # Find columns where number of unique values divided by total column
        # length is smaller than the threshold.
        # Those columns will be projected onto 'category' dtype.
        df_obj_unique_ratio = df_obj.nunique(dropna=False) / df_obj.shape[0]
        category_columns = df_obj_unique_ratio.index[
            df_obj_unique_ratio < category_thresh
        ].tolist()
        if category_columns:
            df_dc[category_columns] = df[category_columns].astype("category")

    return df_dc, nan_mapping
