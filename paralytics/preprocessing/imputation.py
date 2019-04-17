import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

from ..utils.validation import is_numeric


__all__ = [
    'Imputer'
]


class Imputer(BaseEstimator, TransformerMixin):
    """Imputes missing values of the dataframe.

    Imputes missing values with the method adjusted based on the column type.
    For numerical columns imputes missings with the value calculated based on
    the `numerical_method`. For categorical methods imputes missings with the
    most frequent value in the column.

    Parameters
    ----------
        columns: list, optional (default=None)
            Defines columns which missings will be imputed. If not specified
            imputes all of the dataframe columns.

        numerical_method: string {mean, median}, optional (default='mean')
            Method that will be applied to impute numerical columns. Accepts
            all of the pd.DataFrame methods returning some statistic.

        categorical_method: string {mode}, optional (default='mode')
            Method that will be applied to impute categorical columns. Accepts
            all of the pd.DataFrame methods returning some statistic.

    Attributes
    ----------
        imputing_dict_: dict, length = n_features
            Dictionary of values to be imputed in place of NaN's. The key is
            the column name and the value is the value to impute for NaN in
            the corresponding column.

    """
    def __init__(self, columns=None, numerical_method='mean',
                 categorical_method='mode'):
        self.columns = columns
        self.numerical_method = numerical_method
        self.categorical_method = categorical_method

    def fit(self, X, y=None):
        """Fits corresponding imputation values to the X columns.

        Parameters
        ----------
        X: DataFrame, shape = (n_samples, n_features)
            Training data with missing values.

        y: ignore

        Returns
        -------
        self: object
            Returns the instance itself.

        """
        assert isinstance(X, pd.DataFrame), \
            'Input must be an instance of pandas.DataFrame()!'

        if self.columns is None:
            self.columns = X.columns

        self.imputing_dict_ = dict()

        for col in self.columns:
            if is_numeric(X[col]):
                self.imputing_dict_[col] = getattr(
                    X[col], self.numerical_method
                )()
            else:
                self.imputing_dict_[col] = getattr(
                    X[col], self.categorical_method
                )()[0]

        return self

    def transform(self, X):
        """Applies missing values imputation to X.

        Parameters
        ----------
        X: DataFrame, shape = (n_samples, n_features)
            New data with n_samples as its number of samples.

        Returns
        -------
        X_new: DataFrame, shape = (n_samples, n_features)
            X data with substituted missing values to their respective
            imputation values from `imputing_dict_`.

        """
        try:
            getattr(self, 'imputing_dict_')
        except AttributeError:
            raise RuntimeError(
                'Could not find the attribute.\nFitting is necessary before '
                'you do the transformation!'
            )
        X_new = X.copy()
        X_new.fillna(self.imputing_dict_, inplace=True)

        return X_new
