import numpy as np
import pandas as pd

from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except ImportError as e:
    variance_inflation_factor = e

from .preprocessing import Imputer
from .utils import check_is_dataframe


__all__ = [
    'VIFSelector',
    'CorrelationReducer'
]


class VIFSelector(BaseEstimator, TransformerMixin):
    """Makes feature selection based on Variance Inflation Factor.

    Calculates Variance Inflation Factor for a given dataset, in each iteration
    discarding the variable with the highest VIF value and repeats this
    process until it is not below the declared threshold.

    Parameters
    ----------
    thresh: float, optional (default=5.0)
        Threshold value after which further rejection of variables is
        discontinued.

    impute: boolean, optional (default=True)
        Declares whether missing values imputation should be performed.

    impute_method: string, optional (default="mean")
        Declares numerical imputation method for the
        `paralytics.preprocessing.Imputer`.

    fit_intercept: bool, optional (default=True)
        Specifies if the constant (a.k.a. bias or intercept) should be added to
        the decision functions.

    verbose: int, optional (default=0)
        Controls verbosity of output. If 0 there is no output, if 1 displays

    Attributes
    ----------
    imputer_: estimator
        The estimator by means of which missing values imputation is performed.

    viffed_cols_: list
        List of features from a given dataset that exceeded thresh.

    kept_cols_: list
        List of features that left after the vif procedure.

    References
    ----------
    [1] Ffisegydd, `sklearn multicollinearity class
    <https://www.kaggle.com/ffisegydd/sklearn-multicollinearity-class>`_, 2017

    See also
    --------
    paralytics.preprocessing.Imputer

    """

    def __init__(self, thresh=5.0, impute=False, impute_method="mean",
                 fit_intercept=True, verbose=0):
        self.thresh = thresh
        self.impute = impute
        self.impute_method = impute_method
        self.fit_intercept = fit_intercept
        self.verbose = verbose

    def fit(self, X, y=None):
        """Fits columns with a VIF value exceeding the threshold.

        If specified, fits the imputer on X.

        Parameters
        ----------
        X: DataFrame, shape = (n_samples, n_features)
            Input data, where n_samples is the number of samples and n_features
            is the number of features.l

        Returns
        -------
        self: object
            Returns the instance itself.

        """
        if isinstance(variance_inflation_factor, ImportError):
            raise ImportError(
                "`VIFSelector` requires extra requirements installed. "
                "Reinstall paralytics package with 'vif' extra "
                "specified or install the dependencies directly "
                "from the source."
            ).with_traceback(variance_inflation_factor.__traceback__)
        if self.impute:
            self.imputer_ = Imputer(numerical_method=self.impute_method)
            X = self.imputer_.fit_transform(X)

        self.viffed_cols_, self.kept_cols_ = self._viffing(X)

        return self

    def transform(self, X):
        """Apply feature selection based on Variance Inflation Factor.

        Until the maximum VIF in the given dataset does not exceed the declared
        threshold, in every iteration independent variables' VIF values are
        calculated and the variable with the highest VIF value is removed.

        Parameters
        ----------
        X: DataFrame, shape = (n_samples, n_features)
            Input data on which variables elimination will be applied.

        Returns
        -------
        X_new: DataFrame, shape = (n_samples, n_features_new)
            X data with variables remaining after applying feature elimination.

        """
        try:
            getattr(self, 'viffed_cols_')
            getattr(self, 'kept_cols_')
        except AttributeError:
            raise RuntimeError('Could not find the attribute.\n'
                               'Fitting is necessary before you do '
                               'the transformation.')
        X_new = X.copy()
        X_new.drop(self.viffed_cols_, axis=1, inplace=True)

        return X_new

    def _viffing(self, X):
        """In every iteration removes variable with the highest VIF value."""
        check_is_dataframe(X)
        assert ~(X.isnull().values).any(), (
            'DataFrame cannot contain any missing values, consider setting '
            '`impute` parameter to `True` first.'
        )
        assert all(is_numeric_dtype(X[col]) for col in X.columns), \
            'Only numeric dtypes are acceptable.'

        if self.fit_intercept:
            assert not "_constant" in X, (
                "When `fit_intercept == True` the DataFrame can not contain "
                "a column named `_constant`."
            )
            X_new = X.assign(_constant=1.)
        else:
            X_new = X.copy()

        viffed_cols = []

        keep_digging = True
        while keep_digging:
            keep_digging = False

            if self.fit_intercept:
                kept_cols = [col for col in X_new if col != "_constant"]
            else:
                kept_cols = X_new.columns.tolist()

            if len(kept_cols) == 1:
                print("Last variable survived, I'm stopping the procedure!")
                break

            vifs = [
                variance_inflation_factor(
                    X_new.values,
                    X_new.columns.get_loc(var)
                ) for var in kept_cols
            ]

            max_vif = max(vifs)
            if max_vif > self.thresh:
                col_out = kept_cols[vifs.index(max_vif)]
                if self.verbose:
                    print(
                        '{0} with vif={1:.2f} exceeds the threshold.'
                        .format(col_out, max_vif)
                    )
                X_new.drop(col_out, axis=1, inplace=True)
                viffed_cols.append(col_out)
                keep_digging = True

        return viffed_cols, kept_cols


class CorrelationReducer(BaseEstimator, TransformerMixin):
    """Removes correlated columns exceeding the thresh value.

    Parameters
    ----------
    method: string, optional (default='pearson')
        Compute pairwise correlation of columns, excluding NA/null values
        (based on pandas.DataFrame.corr).

        - `pearson`: Standard correlation coefficient.
        - `kendall`: Kendall Tau correlation coefficient.
        - `spearman`: Spearman rank correlation.

    thresh: float, optional (default=.8)
        Threshold value after which further rejection of variables is
        discontinued.

    Attributes
    ----------
    correlated_cols_: list
        List of correlated features from a given dataset that exceeded thresh.

    """
    def __init__(self, thresh=.8, method='pearson'):
        self.thresh = thresh
        self.method = method

    def fit(self, X, y=None):
        """Fits columns with a correlation coefficients exceeding the threshold.

        Parameters
        ----------
        X: DataFrame, shape = (n_samples, n_features)
            Input data, where n_samples is the number of samples and n_features
            is the number of features.

        y: Ignore

        Returns
        -------
        self: object
            Returns the instance itself.

        """
        check_is_dataframe(X)

        self.correlated_cols_ = self._reduce_corr(X, self.thresh, self.method)

        return self

    def transform(self, X):
        """Apply feature selection based on correlation coefficients.

        Removes correlated features with coefficient higher than the threshold
        value.

        Parameters
        ----------
        X: DataFrame, shape = (n_samples, n_features)
            Input data on which variables elimination will be applied.

        Returns
        -------
        X_new: DataFrame, shape = (n_samples, n_features_new)
            X data with variables remaining after applying feature elimination.

        """
        try:
            getattr(self, 'correlated_cols_')
        except AttributeError:
            raise RuntimeError('Could not find the attribute.\n'
                               'Fitting is necessary before you do '
                               'the transformation.')
        check_is_dataframe(X)

        X_new = X.drop(self.correlated_cols_, axis=1)

        return X_new

    @staticmethod
    def _reduce_corr(X, thresh, method):
        """Returns correlated columns exceeding the thresh value."""
        df = X.corr()

        # Create matrix of ones of the same size as the dataframe
        arr_one = np.ones(shape=df.shape, dtype=bool)

        # Set the value above the main diagonal to zero creating L-matrix
        L_arr_one = np.tril(arr_one)
        df.mask(L_arr_one, other=0., inplace=True)
        corr_cols = (df.abs() >= thresh).any()
        cols_out = corr_cols[corr_cols].index.tolist()

        return cols_out
