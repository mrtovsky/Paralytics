import numpy as np 
import pandas as pd 

from inspect import currentframe, getargvalues
from sklearn.base import BaseEstimator, TransformerMixin


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Encodes categorical features with the corresponding target aggregated
     value.

    Accepts DataFrames with categorical features only. 
    
    Parameters
    ----------
    encoding_func: function or string {'mean', 'median'} (default: 'mean')
        Function passed to the agg method called on the GroupBy object
        on the basis of which the coding is made.
    
    nan_as_category: boolean (default: True)
        If True includes NaNs as one of the categories and also applies 
        mean encoding for this subgroup.
        
    Attributes
    ----------
    cat_aggval_: dict, length = n_cat_features
        Dictionary of dictionaries of corresponding aggregated values to given
        subgroups. The key is the column name and the value is the dictionary
        in which the key is the subgroup name and the value is the fitted
        target aggregated value.
    
    """
    def __init__(self, encoding_func='mean', nan_as_category=True):
        icf = currentframe()    
        args, _, _, values = getargvalues(icf)
        values.pop('self')

        for param, value in values.items():
            setattr(self, param, value)        
    
    def fit(self, X, y):
        """Fits corresponding target aggregated values to categorical subgroups.
        
        Parameters
        ----------
        X: DataFrame, shape (n_samples, n_features)
            Training data of independent categorical variables.
        
        y: array-like, shape (n_samples, )
            Vector of target variable values corresponding to X data.
        
        Returns
        -------
        self: object
            Returns the instance itself.
        
        """
        assert isinstance(X, pd.DataFrame), \
            'Input must be an instance of pandas.DataFrame()!'
        assert len(X) == len(y), 'X and y must be the same length!'
        
        self.cat_aggval_ = {}
        df = X.assign(target=y)
        
        for col in X.columns:
            agg_dict = df.groupby(col).target.agg(self.encoding_func).to_dict()
            if self.nan_as_category:
                nan_val = df[df[col].isnull()].target.agg(self.encoding_func)
                agg_dict[np.nan] = nan_val
            self.cat_aggval_[col] = agg_dict
        
        return self
    
    def transform(self, X):
        """Applies target encoding on X.

        X is target encoded with the aggregated values kept in the cat_aggval_.

        Parameters
        ----------
        X: DataFrame, shape (n_samples, n_features)
            New data with n_samples as its number of samples.

        Returns
        -------
        X_new: DataFrame, shape (n_samples, n_features)
            X data with substituted values to their respective target aggregated
            values.

        """
        try:
            getattr(self, 'cat_aggval_')
        except AttributeError:
            raise RuntimeError('Could not find the attribute.\n'
                               'Fitting is necessary before you do '
                               'the transformation!')  
        assert isinstance(X, pd.DataFrame), \
            'Input must be an instance of pandas.DataFrame()!'
        
        X_new = pd.DataFrame()
        for col in X.columns:
            X_new[col] = X[col].map(self.cat_aggval_[col])
        
        return X_new