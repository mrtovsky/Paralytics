import numpy as np
import pandas as pd 
import inspect 

from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor


class VIFSelector(BaseEstimator, TransformerMixin):
    """Makes feature selection based on Variance Inflation Factor.
    
    Calculates Variance Inflation Factor for a given dataset, in each iteration 
    discarding the variable with the highest VIF value and repeats this 
    process until it is not below the declared threshold.

    Based on:
    https://www.kaggle.com/ffisegydd/sklearn-multicollinearity-class

    Parameters
    ----------
    thresh: float (default: 5.0)
        Threshold value after which further rejection of variables is 
        discontinued.

    impute: boolean (default: True)
        Declares whether missing values imputation should be performed.

    impute_strat: string {'mean', 'median', 'most_frequent'} (default: 'mean')
        Declares imputation strategy for the scikit-learn SimpleImputer 
        transformation.

    Attributes
    ---------
    imputer_: estimator
        The estimator by means of which missing values imputation is performed.
    
    dropped_cols_: list, length = n_features
        List of removed features from a given dataset.

    """

    def __init__(self, thresh=5.0, impute=True, impute_strat='mean'):
        icf = inspect.currentframe()    
        args, _, _, values = inspect.getargvalues(icf)
        values.pop('self')

        for param, value in values.items():
            setattr(self, param, value)

        if impute:
            self.imputer_ = SimpleImputer(strategy=impute_strat)

    def fit(self, X, y=None):
        """If specified fit the imputer on X.

        Parameters
        ----------
        X: DataFrame, shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and n_features
            is the number of features.

        Returns
        -------
        self: object
            Returns the instance itself.

        """
        if hasattr(self, 'imputer_'):
            self.imputer_.fit(X)
        
        return self

    def transform(self, X):
        """Apply feature selection based on Variance Inflation Factor.

        Until the maximum VIF in the given dataset does not exceed the declared
        threshold, in every iteration independent variables' VIF values are 
        calculated and the variable with the highest VIF value is removed.

        Parameters
        ----------
        X: DataFrame, shape (n_samples, n_features)
            Input data on which variables elimination will be applied.

        Returns
        -------
        X_new: DataFrame, shape (n_samples, n_features_new)
            X data with variables remaining after applying feature elimination.

        """
        cols = X.columns.tolist()
        if hasattr(self, 'imputer_'):
            X = pd.DataFrame(self.imputer_.transform(X), columns=cols)
        X_new, self.dropped_cols_ = self._viffing(X)
        
        return X_new

    def _viffing(self, X):
        """In every iteration removes variable with the highest VIF value.  

        """
        assert isinstance(X, pd.DataFrame), 'Given input must be a DataFrame.'
        assert ~(X.isnull().values).any(), \
            'DataFrame cannot contain any missing values, consider changing ' \
             + 'impute parameter to True.'
        assert all(is_numeric_dtype(X[col]) for col in X.columns), \
            'Only numeric dtypes are acceptable.'
        
        X_new = X.copy()
        dropped_cols = []

        keep_digging = True
        while keep_digging:
            
            if len(X_new.columns) == 1:
                print("Last variable survived, I'm stopping it right now!")
                break

            keep_digging = False
            vifs = [variance_inflation_factor(X_new.values,
                X_new.columns.get_loc(var)) for var in X_new.columns]
            
            max_vif = max(vifs)
            if max_vif > self.thresh:
                max_loc = vifs.index(max_vif)
                col_out = X_new.columns[max_loc]
                print(f'Dropping {col_out} with vif={max_vif}')
                X_new.drop([col_out], axis=1, inplace=True)
                dropped_cols.append(col_out)
                keep_digging=True
        
        return X_new, dropped_cols