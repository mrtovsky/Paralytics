import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from ..utils import check_column_existence, check_is_dataframe, is_numeric


__all__ = [
    "ColumnProjector",
    "ColumnSelector",
    "ManualGrouper",
    "SparsityGrouper",
    "TypeSelector"
]


class ManualGrouper(BaseEstimator, TransformerMixin):
    """Groups columns in accordance with the guidelines provided.

    Parameters
    ----------
    replace : dict-like
        Dictionary-like object where key is the value that will be imputed
        to the selected values. Values' selection is defined in the values
        assigned to key, which should also be a dictionary containing two
        keys. First key: `regex` with regex sentence assigned as string,
        second key: `case` is the boolean value mentioning whether regex
        should be case-sensitive. Alternatively you can pass a string
        instead of dictionary-like object but then automatically regex will
        be case-insensitive.
    group_columns : string or list-like, optional (default=None)
        Columns for which not matched records will be grouped into a separate
        category. Left by default doesn't group any column. If:

        - `all`:

          Groups all of the categorical columns selected for grouping.

    other_name : string, optional (default="Other")
        Name of the group for unselected records. Valid only if
        `group_others` is set to `True`.

    Attributes
    ----------
    categorical_columns_ : list
        List of categorical columns in a given dataset that has been selected
        for sparsity grouping.
    group_columns_ : list
        List of categorical columns where no matching was found in the provided
        replace sentence that has been grouped into a separate category.

    """
    def __init__(self, replace, group_columns=None, other_name="Other"):
        self.replace = replace
        self.group_columns = group_columns
        self.other_name = other_name

    def fit(self, X, y=None):
        check_is_dataframe(X)

        _categorical_columns = list(self.replace.keys())
        check_column_existence(X, _categorical_columns)

        self.categorical_columns_ = _categorical_columns

        if self.group_columns is None:
            self.group_columns_ = []
        elif self.group_columns == "all":
            self.group_columns_ = self.categorical_columns_.copy()
        elif isinstance(self.group_columns, str):
            self.group_columns_ = [self.group_columns]
        else:
            assert set(self.group_columns) <= set(self.categorical_columns_), (
                "Columns to group must be a subset "
                "of selected categorical columns."
            )
            self.group_columns_ = list(self.group_columns)

        return self

    def transform(self, X):
        check_is_fitted(self, ["categorical_columns_", "group_columns_"])
        check_is_dataframe(X)
        check_column_existence(X, self.categorical_columns_)

        X_new = X.copy()
        for col, pair in self.replace.items():
            for name, params in pair.items():
                if isinstance(params, str):
                    params = {
                        "pat": params,
                        "case": False,
                        "regex": True
                    }
                self._find_and_replace(
                    df=X_new,
                    column=col,
                    replace_value=name,
                    contains_params=params,
                    inplace=True
                )
            if col in self.group_columns_:
                other_sentence = (
                    "^(?:"
                    + "|".join(list(pair.keys()))
                    + ")$"
                )
                X_new.loc[
                    ~X_new[col].str.contains(other_sentence, case=True), col
                ] = self.other_name

        return X_new

    @staticmethod
    def _find_and_replace(df, column, replace_value, contains_params=None,
                          inplace=False):
        if inplace:
            df_new = df
        else:
            df_new = df.copy()
        is_matched = df_new[column].str.contains(**contains_params)
        df_new.loc[is_matched, column] = replace_value

        return df_new if not inplace else None


class SparsityGrouper(BaseEstimator, TransformerMixin):
    """Groups sparse observations in a categorical columns into one category.

    Counts the frequency against each category. Retains categories whose
    cumulative share (with respect to descending sort) in the total dataset is
    equal or higher than the percentile threshold.

    Parameters
    ----------
    columns : list-like, optional (default=None)
        Columns to check for sparsity. If None then groups all columns of
        `object` or `category` dtypes.

    percentile_thresh : float, optional (default=.05)
        Defines the percentile threshold.

    new_category : string or int, optional (default='Other')
        Specifies the category name that will be imputed to the chosen sparse
        observations.

    Attributes
    ----------
    categorical_columns_ : list
        List of categorical columns in a given dataset that has been selected
        for sparsity grouping.

    imputed_categories_ : dict
        Dictionary that keeps track of replaced category names with the new
        category for every feature in the dataset.

    """
    def __init__(self, columns=None, percentile_thresh=.05, 
                 new_category="Other"):
        self.columns = columns
        self.percentile_thresh = percentile_thresh
        self.new_category = new_category

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
        check_is_dataframe(X)

        if self.columns is None:
            self.categorical_columns_ = X.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
        else:
            check_column_existence(X, self.columns)
            self.categorical_columns_ = list(self.columns)

        self.imputed_categories_ = {}
        for col in self.categorical_columns_:
            tracker, idx = 0, 0
            sorted_series = X[col].value_counts(normalize=True)
            while tracker < 1 - self.percentile_thresh:
                tracker += sorted_series.iloc[idx]
                idx += 1
            sparse_categories = sorted_series.index[idx:].tolist()
            if len(sparse_categories) > 1:
                self.imputed_categories_[col] = sparse_categories
            else:
                self.imputed_categories_[col] = []

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
        check_is_fitted(self, ["categorical_columns_", "imputed_categories_"])
        check_is_dataframe(X)
        check_column_existence(X, self.categorical_columns_)

        X_new = X.copy()
        for col in self.categorical_columns_:
            row_indices = X_new[col].isin(self.imputed_categories_[col])
            if X_new[col].dtype == "category":
                try:
                    X_new[col].cat.add_categories(
                        self.new_category, inplace=True
                    )
                except ValueError:
                    pass
                categories_to_remove = list(
                    set(self.imputed_categories_[col]).intersection(
                        X_new[col].cat.categories
                    )
                )
                X_new[col].cat.remove_categories(
                    categories_to_remove,
                    inplace=True
                )
            X_new.loc[row_indices, col] = self.new_category

        return X_new

class ColumnProjector(BaseEstimator, TransformerMixin):
    """Projects variable types onto basic dtypes.

    If not specified projects numeric features onto float, boolean onto bool
    and categorical onto 'category' dtypes.

    Parameters
    ----------
    manual_projection: dict-like, optional (default=None)
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
            if self.num_to_float and is_numeric(X[col]):
                self.automatic_projection_[float].append(col)
            elif is_numeric(X[col]):
                self.automatic_projection_[int].append(col)
            elif set(X[col]) <= {0, 1}:
                self.automatic_projection_[bool].append(col)
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
            for col_type, col_names in projection_dict.items():
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
    column_type: string or list-like
        Names of types to be selected.

    References
    ----------
    [1] J. Ramey, `Building Scikit-Learn Pipelines With Pandas DataFrame
    <https://ramhiser.com/post/2018-04-16-building-scikit-learn-pipeline-with-pandas-dataframe/>`_,
    April 16, 2018

    """
    def __init__(self, column_type):
        self.column_type = column_type

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
        X_new = X.select_dtypes(include=[self.column_type])

        return X_new
