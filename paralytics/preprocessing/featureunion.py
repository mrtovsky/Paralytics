import numpy as np
import pandas as pd
from sklearn.externals.joblib import Parallel, delayed
from sklearn.pipeline import FeatureUnion, _fit_transform_one, _transform_one
from scipy import sparse

"""
Based on
https://github.com/marrrcin/pandas-feature-union

The main goal of this transformer it to return pandas DataFrame object using 
FeatureUnion capabilities from scikit-learn.
"""

class PandasFeatureUnion(FeatureUnion):
    def fit_transform(
                      self,
                      X: pd.DataFrame,
                      y: pd.DataFrame=None,
                      **fit_params
    ) -> pd.DataFrame:
        """Fits and transforms data based on transformers inside pipeline.
            The transformer has to return pandas DataFrame.

           Parameters
           ----------
           X: DataFrame, shape (n_samples, n_features)
               Data with n_samples as its number of samples and n_features as its
               number of features.
           Returns
           -------
           X_new: DataFrame, shape (k_samples, k_features)
               X data with substituted binary-like category columns with its
                   corresponding binary values.
           """
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(
                transformer=trans,
                X=X,
                y=y,
                weight=weight,
                **fit_params
            )
            for name, trans, weight in self._iter())

        if not result:
            # If all transformers are None return array of zeros
            return np.zeros((X.shape[0], 0))
        X_new, transformers = zip(*result)
        self._update_transformer_list(transformers)
        if any(sparse.issparse(f) for f in X_new):
            X_new = sparse.hstack(X_new).tocsr()
        else:
            X_new = self.merge_dataframes_by_column(X_new)
        return X_new

    def merge_dataframes_by_column(
                                   self,
                                   X: pd.DataFrame
    ) -> pd.DataFrame:
        """Concatenates dataframes which resulted from different operations inside
            PandasFeatureUnion.

           Parameters
           ----------
           X: DataFrame, shape (n_samples, n_features)
               Data with n_samples as its number of samples and n_features as its
               number of features.
           Returns
           -------
           X_new: DataFrame, shape (n_samples, n_features)
               X data with substituted binary-like category columns with its
                   corresponding binary values.
           """
        X_new = pd.concat(X, axis="columns", copy=False)
        return X_new

    def transform(
                  self,
                  X: pd.DataFrame
    ) -> pd.DataFrame:
        """Applies conversions which are found in transformer_list and returns pandas DataFrame
           Parameters
           ----------
           X: DataFrame, shape (n_samples, n_features)
               Data with n_samples as its number of samples and n_features as its
               number of features.
           Returns
           -------
           X_new: DataFrame, shape (n_samples, n_features)
               X data with substituted binary-like category columns with its
                   corresponding binary values.
           """
        X_new = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(
                transformer=trans,
                X=X,
                y=None,
                weight=weight)
            for name, trans, weight in self._iter())
        if not X_new:
            # If all transformers are None return array of zeros
            return np.zeros((X.shape[0], 0))
        if any(sparse.issparse(f) for f in X_new):
            X_new = sparse.hstack(X_new).tocsr()
        else:
            X_new = self.merge_dataframes_by_column(X_new)
        return X_new