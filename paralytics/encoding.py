import numpy as np
import pandas as pd
import warnings

from inspect import currentframe, getargvalues
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold

from .exceptions import *


__all__ = [
    'TargetEncoder'
]


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Encodes categorical features with the corresponding target value.

    If cv param is specified, performs determination of mean values on the way
    of cross validation within inner cross validation. As a result for each of
    the outside folds received target aggregated values will be less biased.

    Parameters
    ----------
    columns: list, optional (default=None)
        List of DataFrame columns' names on which target encoding should be
        performed. If not specified all categorical columns are taken.

    nan_as_category: boolean, optional (default=True)
        If True includes NaNs as one of the categories and also applies
        target encoding for this subgroup.

    cv: int, optional (default=None)
        Number of cross-validation folds.

    inner_cv: int, optional (default=None)
        Number of inner cross-validation folds.

    shuffle: boolean, optional (default=True)
        Whether to shuffle the data before splitting into batches.

    alpha: int, optional (default=5)
        Regularization value (times of global mean added to the weighted mean
        of each category). The larger, the more conservative the algorithm
        will be. If you want to use the standard mean just set alpha to 0.

    random_state: int, optional (default=None)
        Random state for sklearn algorithms.

    Attributes
    ----------
    cat_aggval_: dict
        Dictionary of dictionaries of corresponding aggregated values to given
        subgroups. The key is the column name and the value is the dictionary
        in which the key is the subgroup name and the value is the fitted
        target aggregated value.

    Notes
    -----
    When setting cross-validation parameters remember that all categories must
    be sufficiently represented. If a category is sparse, because of the lack
    of representation in one of the k-folds, NaNs in this fold will be
    generated because there are no values ​​recorded from which the statistics
    are calculated. A simple solution is to apply the transformator:
    `preprocessing.CategoricalGrouper` that groups sparse categories into one
    category, before using the target encoding.

    See also
    --------
    paralytics.preprocessing.CategoricalGrouper

    """
    def __init__(self, columns=None, nan_as_category=True,
                 cv=None, inner_cv=None, shuffle=True,
                 alpha=5, random_state=None):

        icf = currentframe()
        args, _, _, values = getargvalues(icf)
        values.pop('self')

        for param, value in values.items():
            setattr(self, param, value)

    def fit(self, X, y):
        """Fits corresponding target aggregated values to categorical subgroups.

        Parameters
        ----------
        X: DataFrame, shape=(n_samples, n_features)
            Training data of independent categorical variables.

        y: array-like, shape=(n_samples, )
            Vector of target variable values corresponding to X data.

        Returns
        -------
        self: object
            Returns the instance itself.

        """
        assert isinstance(X, pd.DataFrame), \
            'Input must be an instance of pandas.DataFrame()!'
        assert len(X) == len(y), 'X and y must be the same length!'

        if self.columns is None:
            self.columns = X.select_dtypes('category').columns.tolist()
            if not self.columns:
                warnings.warn(
                    'No column selected. Make sure you have variables of '
                    '"category" type in your dataframe or explicitly provide '
                    'the column names you want to target encode.',
                    NothingSelectedWarning
                )

        self.cat_aggval_ = {}
        df = X.assign(target=y)

        for col in self.columns:
            agg_dict = df.groupby(col).target.agg(
                self._penalized_mean, y=y, alpha=self.alpha
            ).to_dict()
            if self.nan_as_category and df[col].isnull().sum() > 0:
                nan_val = df[df[col].isnull()].target.agg(
                    self._penalized_mean, y=y, alpha=self.alpha
                )
                agg_dict['NaNCategory'] = nan_val
            self.cat_aggval_[col] = agg_dict

        return self

    def transform(self, X, y=None):
        """Applies target encoding on X.

        X is target encoded with the aggregated values kept in the
        `cat_aggval_` and for the training data encoding is made with
        additional spread obtained in the cross-validation within
        cross-validation.

        Parameters
        ----------
        X: DataFrame, shape = (n_samples, n_features)
            New data with n_samples as its number of samples.

        y: array-like, shape = (n_samples, )
            Vector of target variable values corresponding to X data.

        Returns
        -------
        X_new: DataFrame, shape = (n_samples, n_features)
            X data with substituted values to their respective target
            aggregated values.

        """
        try:
            getattr(self, 'cat_aggval_')
        except AttributeError:
            raise RuntimeError(
                'Could not find the attribute.\nFitting is necessary before '
                'you do the transformation!'
            )
        assert isinstance(X, pd.DataFrame), \
            'Input must be an instance of pandas.DataFrame()!'

        X_new = X.copy()

        if self.nan_as_category:
            missing_col = X_new.columns[X_new.isnull().any()].tolist()
            col_to_fill = list(set(self.columns).intersection(missing_col))
            for col in col_to_fill:
                if X_new[col].dtype == 'category':
                    X_new[col] = X_new[col].cat.add_categories('NaNCategory')
            X_new[col_to_fill] = X_new[col_to_fill].fillna('NaNCategory')

        if y is None or self.cv is None:
            for col in self.columns:
                X_new[col] = X[col].map(self.cat_aggval_[col]).astype(float)

        else:
            assert self.inner_cv is not None, (
                'When cv param is specified you must assign a value to the '
                'inner_cv param as well!'
            )
            X_new[self.columns] = self._transform_train_cv(
                X_new[self.columns], y
            ).astype(float)
        return X_new

    def fit_transform(self, X, y=None):
        """Fit to data then transform it.

        Fits transformer to X and y and returns transformed version of X.

        Parameters
        ----------
        X: DataFrame, shape = (n_samples, n_features)
            Training data of independent categorical variables.

        y: array-like, shape = (n_samples, )
            Vector of target variable values corresponding to X data.

        Returns
        -------
        X_new: DataFrame, shape = (n_samples, n_features)
            X data with substituted values to their respective target
            aggregated values.

        """
        return self.fit(X, y).transform(X, y)

    def _transform_train_cv(self, X, y):
        """This method is only applied for a training set.

        By using only the part of the data (k-1 folds) it estimates the
        encoding value for the leftover fold. It performs those activities
        independently for each category. As a result we get synthetic values
        located more dense in a given space.

        """
        X_new = pd.DataFrame(index=X.index, columns=X.columns)
        df = X.assign(target=y)

        kf = KFold(
            n_splits=self.cv,
            shuffle=self.shuffle,
            random_state=self.random_state
        )

        for train_idx, encode_idx in kf.split(df):
            inner_kf = KFold(
                n_splits=self.inner_cv,
                shuffle=self.shuffle,
                random_state=self.random_state
            )

            # Declaring list of DataFrames that will store inner mean values.
            mean_df_list = [
                pd.DataFrame({f'{col}': X[col].unique()}) for col in X.columns
            ]
            inner_df = df.iloc[train_idx, :].copy()
            for loop_idx, (inner_train_idx, _) in (
                enumerate(inner_kf.split(inner_df))
            ):
                for idx, col in enumerate(X.columns):
                    agg_values = pd.DataFrame(
                        inner_df.iloc[inner_train_idx, :]
                        .groupby(col).target
                        .agg('mean')
                    ).reset_index()
                    mean_df_list[idx] = mean_df_list[idx].merge(
                        agg_values,
                        how='left',
                        on=f'{col}',
                        suffixes=(f'_{loop_idx-1}', f'_{loop_idx}')
                    )
            mean_df_list = [
                data.set_index(data.columns[0]).agg(
                    self._penalized_mean,
                    axis=1,
                    n_instances=X.iloc[train_idx, col_idx].count(),
                    y=y[train_idx],
                    alpha=self.alpha * (1 - 1 / self.cv)
                ).to_dict()
                for col_idx, data in enumerate(mean_df_list)
            ]

            for loop_idx, col in enumerate(X_new.columns):
                X_new[col].iloc[encode_idx] = X[col].map(
                    mean_df_list[loop_idx]
                )

        return X_new

    @staticmethod
    def _penalized_mean(series, y, alpha, n_instances=None):
        """Further regularization.

        Adds alpha value multiplied by the mean of the whole training
        data to the weighted average.

        Formula: (p_c * n_c + p_global * alpha) / (n_c + alpha), where:
            p_c: mean for a category
            n_c: number of instances in a category
            p_global: global target mean
            alpha: regularization parameter

        """
        if n_instances is None:
            n_instances = series.count()
        numerator = np.mean(series) * n_instances + np.mean(y) * alpha
        denominator = n_instances + alpha
        return numerator / denominator
