import numpy as np
import pandas as pd

from inspect import currentframe, getargvalues
from sklearn.base import BaseEstimator, TransformerMixin
from pandas.api.types import is_numeric_dtype


__all__ = [
    'CategoricalBinarizer',
    'CategoricalGrouper',
    'ColumnProjector',
    'ColumnSelector',
    'TypeSelector'
]


class CategoricalBinarizer(BaseEstimator, TransformerMixin):
    """Finds categorical columns with binary-like response and converts them.

    Searches throughout the categorical columns in the DataFrame and finds
    those which contain categories corresponding to the passed boolean values
    only.

    Parameters
    ----------
    keywords_{true, false}: list, optional (default=None)
        List of categories' names corresponding to {True, False} logical
        values.

    Attributes
    ----------
    columns_binarylike_: list
        List of column names that should be mapped to boolean.

    """
    def __init__(self, keywords_true=None, keywords_false=None):
        self.keywords_true = keywords_true
        self.keywords_false = keywords_false

    def fit(self, X, y=None):
        """Fits selection of binary-like columns.

        Parameters
        ----------
        X: pd.DataFrame, shape = (n_samples, n_features)
            Data with n_samples as its number of samples and n_features as its
            number of features.

        y: ignore

        Returns
        -------
        self: object
            Returns the instance itself.

        """
        self.columns_binarylike_ = []

        if self.keywords_true is None:
            self.keywords_true = ['yes', 'YES', 'Yes']
        if self.keywords_false is None:
            self.keywords_false = ['no', 'NO', 'No']

        keywords_binarylike = set(self.keywords_true + self.keywords_false)

        for col in X.columns:
            try:
                binarylike_only = \
                    set(X[col].cat.categories) <= keywords_binarylike
            except AttributeError as e:
                continue
            if binarylike_only:
                self.columns_binarylike_.append(col)

        return self

    def transform(self, X):
        """Applies boolean convertion to binary-like category columns.

        X columns that match the condition of containing only binary-like
        string values are mapped to boolean values corresponding to the
        passed strings expected to be interpreted as binary response.

        Parameters
        ----------
        X: pd.DataFrame, shape = (n_samples, n_features)
            Data with n_samples as its number of samples and n_features as its
            number of features.

        Returns
        -------
        X_new: pd.DataFrame, shape = (n_samples, n_features)
            X data with substituted binary-like category columns with its
            corresponding binary values.
        """
        try:
            getattr(self, 'columns_binarylike_')
        except AttributeError:
            raise RuntimeError(
                'Could not find the attribute.\nFitting is necessary before '
                'you do the transformation!'
            )
        assert isinstance(X, pd.DataFrame), \
            'Input must be an instance of pandas.DataFrame()!'

        X_new = X.copy()

        dict_true = dict.fromkeys(self.keywords_true, True)
        dict_false = dict.fromkeys(self.keywords_false, False)

        translator_binarylike = {**dict_true, **dict_false}

        for col in self.columns_binarylike_:
            X_new[col] = X[col].map(translator_binarylike)

        return X_new


class CategoricalGrouper(BaseEstimator, TransformerMixin):
    """Groups sparse observations in a categorical columns into one category.

    Parameters
    ----------
    method: string {'freq'}, optional (default='freq')
        The sparse categories grouping method:

        - `freq`:

          Counts the frequency against each category. Retains categories
          whose cumulative share (with respect to descending sort) in the
          total dataset is equal or higher than the percentile threshold.

    percentile_thresh: float, optional (default=.05)
        Defines the percentile threshold for 'freq' method.

    new_cat: string or int, optional (default='Other')
        Specifies the category name that will be imputed to the chosen sparse
        observations.

    include_cols: list, optional (default=None)
        Specifies column names that should be treated like categorical
        features. If None then estimator is executed only on the automatically
        selected categorical columns.

    exclude_cols: list, optional (default=None)
        Specifies categorical column names that should not be treated like
        categorical features. If None then no column is excluded from
        transformation.

    Attributes
    ----------
    cat_cols_: list
        List of categorical columns in a given dataset.

    imp_cats_: dict
        Dictionary that keeps track of replaced category names with the new
        category for every feature in the dataset.

    """
    def __init__(self, method='freq', percentile_thresh=.05, new_cat='Other',
                 include_cols=None, exclude_cols=None):
        icf = currentframe()
        args, _, _, values = getargvalues(icf)
        values.pop('self')

        for param, value in values.items():
            setattr(self, param, value)

    def fit(self, X, y=None):
        """Fits grouping with X by using given method.

        Parameters
        ----------
        X: pd.DataFrame, shape = (n_samples, n_features)
            Training data of independent variable values.

        y: ignore

        Returns
        -------
        self: object
            Returns the instance itself.

        """
        assert isinstance(X, pd.DataFrame), \
            'Input must be an instance of pandas.DataFrame()!'
        assert len(X) > 0, 'Input data can not be empty!'

        self.cat_cols_ = self._cat_cols_selection(
            X, self.include_cols, self.exclude_cols
        )

        self.imp_cats_ = {}
        if self.method == 'freq':
            for col in self.cat_cols_:
                tracker, i = 0, 0
                sorted_series = X[col].value_counts(normalize=True)
                while tracker < 1 - self.percentile_thresh:
                    tracker += sorted_series.iloc[i]
                    i += 1
                sparse_cats = sorted_series.index[i:].tolist()
                if len(sparse_cats) > 1:
                    self.imp_cats_[col] = sparse_cats
                else:
                    self.imp_cats_[col] = []

        return self

    def transform(self, X):
        """Apply grouping of sparse categories on X.

        Parameters
        ----------
        X: pd.DataFrame, shape = (n_samples, n_features)
            Data with n_samples as its number of samples.

        Returns
        -------
        X_new: pd.DataFrame, shape = (n_samples_new, n_features)
            X data with substituted sparse categories to new_cat.

        """
        try:
            getattr(self, 'imp_cats_')
            getattr(self, 'cat_cols_')
        except AttributeError:
            raise RuntimeError('Could not find the attribute.\n'
                               'Fitting is necessary before you do '
                               'the transformation.')
        assert isinstance(X, pd.DataFrame), \
            'Input must be an instance of pandas.DataFrame()'

        X_new = X.copy()
        for col in self.cat_cols_:
            row_indices = X_new[col].isin(self.imp_cats_[col])
            if X_new[col].dtype.name == 'category':
                try:
                    X_new[col].cat.add_categories(self.new_cat, inplace=True)
                except ValueError as e:
                    raise ValueError(
                        'You need to specify different "new_cat" value, '
                        'because the current one is already included in the '
                        'category names.'
                    ).with_traceback(e.__traceback__)
                cat_removals = list(
                    set(self.imp_cats_[col]).intersection(
                        X_new[col].cat.categories
                    )
                )
                X_new[col].cat.remove_categories(
                    cat_removals,
                    inplace=True
                )
            X_new.loc[row_indices, col] = self.new_cat

        return X_new

    @staticmethod
    def _cat_cols_selection(X, include, exclude):
        """Returns categorical columns including the user's corrections."""
        cat_cols = X.select_dtypes('category').columns.tolist()

        if include is not None:
            assert isinstance(include, list), \
                'Columns to include must be given as an instance of a list!'
            cat_cols = [
                col for col in X.columns
                if col in cat_cols or col in include
            ]

        if exclude is not None:
            assert isinstance(exclude, list), \
                'Columns to exclude must be given as an instance of a list!'
            cat_cols = [col for col in cat_cols if col not in exclude]

        return cat_cols


class ColumnProjector(BaseEstimator, TransformerMixin):
    """Projects variable types onto basic dtypes.

    If not specified projects numeric features onto float, boolean onto bool
    and categorical onto 'category' dtypes.

    Parameters
    ----------
    manual_projection: dictionary, optional (default=None)
        Dictionary where keys are dtype names onto which specified columns
        will be projected and values are lists containing names of variables to
        be projected onto given dtype. Example usage:

        >>> manual_projection = {
        >>>    float: ['foo', 'bar'],
        >>>    'category': ['baz'],
        >>>    int: ['qux'],
        >>>    bool: ['quux']
        >>> }

    num_to_float: boolean, optional (default=True)
        Specifies whether numerical variables should be projected onto float
        (if True) or onto int (if False).

    Attributes
    ----------
    automatic_projection_: dict
        Dictionary where key is the dtype name onto which specified columns
        will be projected chosen automatically (when manual_projection is
        specified then this manual assignment is decisive).
    """
    def __init__(self, manual_projection=None, num_to_float=True):
        self.manual_projection = manual_projection
        self.num_to_float = num_to_float

    def fit(self, X, y=None):
        """Fits corresponding dtypes to X.

        Parameters
        ----------
        X: pd.DataFrame, shape = (n_samples, n_features)
            Training data of independent variable values.

        y: ignore

        Returns
        -------
        self: object
            Returns the instance itself.

        """
        self.automatic_projection_ = {'category': [], bool: []}

        if self.num_to_float:
            self.automatic_projection_[float] = []
        else:
            self.automatic_projection_[int] = []

        for col in X.columns:
            if set(X[col]) <= {0, 1}:
                self.automatic_projection_[bool].append(col)
            elif self.num_to_float and is_numeric_dtype(X[col]):
                self.automatic_projection_[float].append(col)
            elif is_numeric_dtype(X[col]):
                self.automatic_projection_[int].append(col)
            else:
                self.automatic_projection_['category'].append(col)

        return self

    def transform(self, X):
        """Apply variable projection on X.

        Parameters
        ----------
        X: pd.DataFrame, shape = (n_samples, n_features)
            New data with n_samples as its number of samples.

        Returns
        -------
        X_new: pd.DataFrame, shape = (n_samples, n_features)
            X data with projected values onto specified dtype.

        """
        try:
            getattr(self, 'automatic_projection_')
        except AttributeError:
            raise RuntimeError(
                'Could not find the attribute.\nFitting is necessary before '
                'you do the transformation!'
            )
        assert isinstance(X, pd.DataFrame), \
            'Input must be an instance of pandas.DataFrame()'

        X_new, columns_projected = self._project(X, self.manual_projection)
        X_new, _ = self._project(
            X_new, self.automatic_projection_, skip_columns=columns_projected
        )

        return X_new

    @staticmethod
    def _project(X, projection_dict, skip_columns=None):
        """Projects X in accordance with the guidelines provided."""
        X_new = X.copy()
        columns_projected = []

        if skip_columns is None:
            skip_columns = []

        if projection_dict is not None:
            assert isinstance(projection_dict, dict), \
                'projection_dict must be an instance of the dictionary!'
            for col_type, col_names in projection_dict.items():
                assert isinstance(col_names, list), (
                    'Values of projection_dict must be an instance '
                    'of the list!'
                )
                cols_to_project = [
                    col for col in col_names if col not in skip_columns
                ]
                if cols_to_project:
                    try:
                        X_new[cols_to_project] = (
                            X_new[cols_to_project].astype(col_type)
                        )
                    except KeyError:
                        cols_error = list(
                            set(cols_to_project) - set(X_new.columns)
                        )
                        raise KeyError("C'mon, those columns ain't in "
                                       "the DataFrame: %s" % cols_error)
                    columns_projected.extend(cols_to_project)

        return X_new, columns_projected


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Limits the X to selected columns.

    Parameters
    ----------
    columns: list
        List of column names selected to be left.

    References
    ----------
    [1] J. Ramey, `Building Scikit-Learn Pipelines With Pandas DataFrame
    <https://ramhiser.com/post/2018-04-16-building-scikit-learn-pipeline-with-pandas-dataframe/>`_,
    April 16, 2018

    """
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        """Fits columns selection to X.

        Parameters
        ----------
        X: pd.DataFrame, shape = (n_samples, n_features)
            Training data of independent variable values.

        y: Ignore

        Returns
        -------
        self: object
            Returns the instance itself.

        """
        return self

    def transform(self, X):
        """Apply columns selection to X.

        Parameters
        ----------
        X: pd.DataFrame, shape = (n_samples, n_features)
            New data with n_samples as its number of samples.

        Returns
        -------
        X_new: pd.DataFrame, shape = (n_samples, n_features)
            X data limited to selected columns only.

        """
        assert isinstance(X, pd.DataFrame), \
            'Input must be an instance of pandas.DataFrame!'

        try:
            X_new = X[self.columns]
            return X_new
        except KeyError as e:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError(
                "Selected columns not found in the DataFrame: %s" % cols_error
            ).with_traceback(e.__traceback__)


class TypeSelector(BaseEstimator, TransformerMixin):
    """Limits the X to selected types.

    Parameters
    ----------
    col_type: string or list-like
        Names of types to be selected.

    References
    ----------
    [1] J. Ramey, `Building Scikit-Learn Pipelines With Pandas DataFrame
    <https://ramhiser.com/post/2018-04-16-building-scikit-learn-pipeline-with-pandas-dataframe/>`_,
    April 16, 2018

    """
    def __init__(self, col_type):
        self.col_type = col_type

    def fit(self, X, y=None):
        """Fits types selection to X.

        Parameters
        ----------
        X: pd.DataFrame, shape = (n_samples, n_features)
            Training data of independent variable values.

        y: ignore

        Returns
        -------
        self: object
            Returns the instance itself.

        """
        return self

    def transform(self, X):
        """Apply types selection to X.

        Parameters
        ----------
        X: pd.DataFrame, shape = (n_samples, n_features)
            New data with n_samples as its number of samples.

        Returns
        -------
        X_new: pd.DataFrame, shape = (n_samples, n_features)
            X data limited to selected types only.

        """
        assert isinstance(X, pd.DataFrame), \
            'Input must be an instance of pandas.DataFrame()'
        X_new = X.select_dtypes(include=[self.col_type])

        return X_new
