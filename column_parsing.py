import numpy as np 
import pandas as pd 

from sklearn.base import BaseEstimator, TransformerMixin
from pandas.api.types import is_numeric_dtype


class ColumnProjector(BaseEstimator, TransformerMixin):
    """Projects variable types onto basic dtypes.

    If not specified projects numeric features onto float, boolean onto bool and
    categorical onto 'category' dtypes.

    Parameters
    ---------- 
    manual_projection: dictionary
        Dictionary where keys are dtype names onto which specified columns 
        will be projected and values are lists containing names of variables to
        be projected onto given dtype.
            Example: manual_projection={
                                        float: ['foo', 'bar'],
                                        'category': ['baz'],
                                        int: ['qux'],
                                        bool: ['quux']
                     }

    num_to_float: boolean (default: True)
        Specifies whether numerical variables should be projected onto float
        (if True) or onto int (if False).

    """
    def __init__(self, manual_projection=None, num_to_float=True):
        self.manual_projection = manual_projection
        self.num_to_float = num_to_float

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Apply variable projection on X.

        Parameters
        ----------
        X: DataFrame, shape (n_samples, n_features)
            New data with n_samples as its number of samples.

        Returns
        -------
        X: DataFrame, shape (n_samples, n_features)
            X data with projected values onto specified dtype.

        """
        assert isinstance(X, pd.DataFrame), \
            'Input must be an instance of pandas.DataFrame()'

        columns = X.columns.values
        if self.manual_projection is not None:
            assert isinstance(manual_projection, dict), \
                'manual_projection must be an instance of the dictionary!'
            for col_type, col_names in manual_projection.items():
                assert isinstance(col_names, list), (
                    'Values of manual_projection must be an instance ' 
                    'of the list!'
                )
                try:
                    X[col_names] = X[col_names].astype(col_type)
                    columns = [col for col in columns 
                               if col not in col_names]
                except KeyError:
                    cols_error = list(set(col_names) - set(X.columns.values))
                    raise KeyError("C'mon, those columns ain't in "
                                   "the DataFrame: %s" % cols_error)

        for col in columns:
            if set(X[col]) <= {0, 1}:
                X[col] = X[col].astype(bool)
            elif num_to_float and is_numeric_dtype(X):
                X[col] = X[col].astype(float)
            elif is_numeric_dtype(X):
                X[col] = X[col].astype(int)
            else:
                X[col] = X[col].astype('category')

        return X