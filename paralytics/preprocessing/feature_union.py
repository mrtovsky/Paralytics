import numpy as np
import pandas as pd

from scipy import sparse
from sklearn.externals.joblib import Parallel, delayed
from sklearn.pipeline import FeatureUnion, _fit_transform_one, _transform_one


class PandasFeatureUnion(FeatureUnion):
    """Concatenates results of multiple pandas.DataFrame transformers.

    Using FeatureUnion capabilities from scikit-learn applies multiple
    transformers always returning pandas.DataFrame object.

    References
    ----------
    [1] marrrcin, `pandas-feature-union
    <https://github.com/marrrcin/pandas-feature-union>`_, 2018

    """
    def fit_transform(self, X, y=None, **fit_params):
        """Fits and transforms data based on transformers inside pipeline.

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

        Notes
        -----
        The transformer has to return pandas.DataFrame object.

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

    def merge_dataframes_by_column(self, X):
        """Concatenates dataframes which resulted from different operations.

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

    def transform(self, X):
        """Applies conversions which are found in transformer_list.

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

        Notes
        -----
        Returns pandas.DataFrame object.

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
