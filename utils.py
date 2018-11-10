import numpy as np
import pandas as pd

from pandas import Series
from pandas.api.types import is_numeric_dtype


class Utils(object):
    """A collection of minor methods.

    """
    @staticmethod
    def reduce_corr(X, thresh=.9, method='pearson'):
        """Removes correlated columns exceeding the thresh value.
        
        """
        assert isinstance(X, pd.DataFrame), \
            'Input must be an instance of pandas.DataFrame()'

        df = X.corr()
        # Create matrix of ones of the same size as the dataframe
        arr = np.ones(shape=df.shape, dtype=bool)
        # Set the value above the main diagonal to zero creating L-matrix
        L_arr = np.tril(arr)
        df.mask(L_arr, other=0., inplace=True)
        corr_cols = (df.abs() >= thresh).any()
        cols_out = corr_cols[corr_cols == True].index
        X_new = X.drop(cols_out, axis=1)

        return X_new

    @staticmethod
    def miss_split():
        """
        """
        pass

    @staticmethod
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

    @staticmethod
    def check_mono(X):
        """
        """
        pass

    @staticmethod
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
