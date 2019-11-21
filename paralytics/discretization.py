from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

from .utils.validation import check_is_dataframe, check_is_series


__all__ = [
    "ManualDiscretizer",
    "ClassificationTreeDiscretizer",
    "RegressionTreeDiscretizer",
    "discretize"
]


def discretize(series, cut_points=None, n_quantiles=4,
               labels=None, min_val=None, max_val=None):
    """Manually bins continuous variable into the declared intervals.

    If the cut-off points are not declared the split is made using
    quantiles.

    Parameters
    ----------
    series: array-like, shape = (n_samples, )
        Vector passed as an one-dimensional array-like object where
        n_samples in the number of samples.

    cut_points: array-like, optional (default=None)
        Increasing monotonic sequence generating right-closed intervals.
        Values not allocated to any of the categories will be assigned to
        the empty set. For example given: cut_points=[1, 5, 9] will
        generate intervals: [-inf, 1], (1, 5], (5, 9], (9, inf].
        If you want to specify lower and upper limitations, set parameters:
        "min_val", "max_val" to a specific value.

    n_quantiles: int, optional (default=4)
        When cut_points are not declared it sets the number of quantiles
        to which the variable will be splitted. For example setting
        n_quantiles = 4 will return quartiles of X values between min_val
        and max_val.

    labels: string: {'auto'} or list, optional (default=None)
        Specifies returned bucket names, needs to be the same length as the
        number of created buckets:

        - `auto`:

            Assigns default values to group names by numbering them.

    min_val: float, optional (default=None)
        Determines lower limit value. If not specified takes -np.inf.

    max_val: float, optional (default=None)
        Determines upper limit value. If not specified takes np.inf.

    Returns
    -------
    series_new: array, shape = (n_samples, )
        Input data with its original values ​​being substituted with their
        respective labels.

    """
    series = check_is_series(series)

    if min_val is None:
        min_val = -np.inf

    if max_val is None:
        max_val = np.inf

    # Default break_points in case of no declaration of cut_points.
    if cut_points is None:
        _series = series[~pd.isna(series)]
        _series = _series[(_series >= min_val) & (_series <= max_val)]
        break_points = np.quantile(
            _series.reset_index(drop=True), np.linspace(0, 1, n_quantiles + 1)
        )
    else:
        break_points = np.insert(
            cut_points,
            [0, len(cut_points)],
            [min_val, max_val]
        )
    break_points = np.unique(break_points)

    if labels == "auto":
        labels = range(len(break_points) - 1)

    series_new = pd.cut(
        series,
        bins=break_points,
        labels=labels,
        include_lowest=True
    )

    return series_new


class BaseDiscretizer(BaseEstimator, TransformerMixin, metaclass=ABCMeta):
    """Base class for all discretization classes.

    Warning: This class should not be used directly. Use derived classes
    instead.

    Parameters
    ----------
    columns: string {<column-name>, all} or list
        The collection of discretized column names. If string value is passed
        requires numeric column name or ``all`` value ordering the selection
        of all numeric columns.

    n_jobs: int
        The number of jobs to run in parallel for both ``fit`` and
        ``transform``.

    Attributes
    ----------
    columns_: list
        The collection of discretized columns with enforced type.

    cuts_: dictionary-like
        The collection of cut points for every discretized column, used to
        determine bins, where the key is the column name and the value is the
        collection of corresponding cut points.

    Notes
    -----
    Inheriting classes' fit method needs to define ``self.cuts_`` dictionary.

    """
    @abstractmethod
    def __init__(self, columns, n_jobs):
        self.columns = columns
        self.n_jobs = n_jobs

    @abstractmethod
    def fit(self, X, y=None):
        """Fit discretization with X by extracting cut points.

        Parameters
        ----------
        X: pandas.DataFrame, shape = (n_samples, n_features)
            Training data of independent variable values.

        y: array-like, shape = (n_samples, )
            Vector of target variable values corresponding to X data.

        Returns
        -------
        self: object
            Returns the instance itself.

        """
        check_is_dataframe(X)
        numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
        if self.columns == "all":
            self.columns_ = X.select_dtypes(include=np.number).columns.tolist()
        else:
            if isinstance(self.columns, str):
                columns = [self.columns]
            else:
                columns = list(self.columns)
            assert set(columns).issubset(numeric_columns), (
                "Columns chosen for discretization need to be of numeric "
                "dtype.\nThe following columns do not satisfy this condition:\n"
                "{}.\n\nMake sure column names are passed correctly."
                .format(
                    " ,".join(set(numeric_columns).difference(columns))
                )
            )
            self.columns_ = columns

        return self

    def transform(self, X, y=None):
        """Apply discretization on X.

        X is projected on the previously extracted cut points.

        Parameters
        ----------
        X: pd.DataFrame, shape = (n_samples, n_features)
            New data with n_samples as its number of samples.

        Returns
        -------
        X_new: pd.DataFrame, shape = (n_samples_new, n_features)
            X data with substituted values to their respective labels.

        """
        check_is_fitted(self, ["columns_", "cuts_"])
        check_is_dataframe(X)

        X_new = pd.concat(
            Parallel(n_jobs=self.n_jobs)(
                delayed(self._discretize)(X[column])
                for column in X.columns
            ),
            axis="columns"
        )
        return X_new

    def _discretize(self, series):
        """Discretize continuous feature, otherwise return the series itself."""
        name = series.name
        if name in self.columns_:
            return discretize(series, cut_points=self.cuts_[name])
        else:
            return series


class ManualDiscretizer(BaseDiscretizer):
    """Manually discretize continuous features.

    Parameters
    ----------
    columns: string {<column-name>, all} or list, optional (default="all")
        The collection of discretized column names. If string value is passed
        requires numeric column name or ``all`` value ordering the selection
        of all numeric columns.

    n_jobs: int, optional (default=None)
        The number of jobs to run in parallel for both ``fit`` and
        ``transform``.

    cut_points: string {quantiles} or array-like, optional (default="quantiles")
        The collection of cut points corresponding to column selected for
        discretization. Length must be equal to ``columns`` length.
        By default takes quantiles of corresponding columns as cut points.

    n_quantiles: int, optional(default=4)
        Number of quantiles taken as cut points if ``cut_points`` parameter
        is set to ``quantiles``. Left by default takes quartiles.

    Attributes
    ----------
    columns_: list
        The collection of discretized columns with enforced type.

    cut_points_: list
        The collection of cut points for corresponding columns.

    cuts_: dictionary-like
        The collection of cut points for every discretized column, used to
        determine bins, where the key is the column name and the value is the
        collection of corresponding cut points.

    """
    def __init__(self, columns="all", n_jobs=None, cut_points="quantiles",
                 n_quantiles=4):
        super().__init__(columns, n_jobs)
        self.cut_points = cut_points
        self.n_quantiles = n_quantiles

    def fit(self, X, y=None):
        """Fit manual discretization with X.

        Parameters
        ----------
        X: pandas.DataFrame, shape = (n_samples, n_features)
            Training data of independent variable values.

        y: array-like, shape = (n_samples, )
            Vector of target variable values corresponding to X data.

        Returns
        -------
        self: object
            Returns the instance itself.

        """
        super().fit(X)
        if self.cut_points == "quantiles":
            assert isinstance(self.n_quantiles, int) and self.n_quantiles > 0, (
                "Parameter ``n_quantiles`` takes natural numbers only."
            )
            cut_points = [
                np.quantile(
                    X[column][~pd.isna(X[column])],
                    np.delete(
                        np.linspace(0, 1, self.n_quantiles + 1),
                        [0, self.n_quantiles]
                    )
                )
                for column in self.columns_
            ]
        else:
            cut_points = list(np.atleast_2d(self.cut_points))
            assert len(self.columns_) == len(cut_points), (
                "Number of cut points collections passed to the ``cut_points`` "
                "parameter must be equal to number of columns selected for "
                "discretization.\n{} columns and {} cut points collections are "
                "currently specified."
                .format(len(self.columns_), len(cut_points))
            )
        self.cut_points_ = cut_points
        self.cuts_ = dict(zip(self.columns_, self.cut_points_))

        return self


class TreeDiscretizer(BaseDiscretizer, metaclass=ABCMeta):
    """Base class for decision tree discretizers.

    Warning: This class should not be used directly. Use derived classes
    instead.

    """
    @abstractmethod
    def __init__(self, columns, n_jobs, base_estimator, criterion, splitter,
                 max_depth, min_samples_split, min_samples_leaf,
                 min_weight_fraction_leaf, max_features, max_leaf_nodes,
                 min_impurity_decrease, min_impurity_split, class_weight=None,
                 presort=False, random_state=None, estimator_params=tuple(),
                 save_estimators=False):
        super().__init__(columns, n_jobs)
        self.base_estimator = base_estimator
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.class_weight = class_weight
        self.presort = presort
        self.random_state = random_state
        self.estimator_params = estimator_params
        self.save_estimators = save_estimators

    def fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted=None):
        """Fit tree discretization with X.

        Parameters
        ----------
        X: pandas.DataFrame, shape = (n_samples, n_features)
            Training data of independent variable values.

        y: array-like, shape = (n_samples, )
            Vector of target variable values corresponding to X data.

        **tree_params: **kwargs
            Fit method parameters of ``sklearn.tree.BaseDecisionTree``.

        Returns
        -------
        self: object
            Returns the instance itself.

        """
        super().fit(X)
        y = check_is_series(y)

        estimator = clone(self.base_estimator)
        estimator.set_params(**{
            param: getattr(self, param) for param in self.estimator_params
        })
        self.estimator_ = estimator

        if self.save_estimators:
            self.estimators_ = []

        self.cut_points_ = Parallel(n_jobs=self.n_jobs)(
            delayed(self._find_cut_points)(
                X[column], y, sample_weight, check_input, X_idx_sorted
            )
            for column in self.columns_
        )
        self.cuts_ = dict(zip(self.columns_, self.cut_points_))

        return self

    def _find_cut_points(self, series, y, sample_weight, check_input,
                         X_idx_sorted):
        """Find cut points for a single feature."""
        series = check_is_series(series)
        _series = series[~pd.isna(series)].reset_index(drop=True)
        _y = y[~pd.isna(series)].reset_index(drop=True)

        estimator = clone(self.estimator_)
        estimator.fit(
            _series.values.reshape(-1, 1), _y,
            sample_weight=sample_weight,
            check_input=check_input,
            X_idx_sorted=X_idx_sorted
        )
        if self.save_estimators:
            self.estimators_.append(estimator)

        # Excluding leaves.
        cut_points = estimator.tree_.threshold[estimator.tree_.feature != -2]

        min_val = _series.min()
        max_val = _series.max()

        cut_points = np.delete(
            cut_points,
            np.asarray(
                (min_val > cut_points)
                | (max_val < cut_points)
            ).nonzero()[0]
        )

        return cut_points


class ClassificationTreeDiscretizer(TreeDiscretizer):
    """Discretize continuous features with use of decision tree algorithm.

    Parameters
    ----------
    columns: string {<column-name>, all} or list, optional (default="all")
        The collection of discretized column names. If string value is passed
        requires numeric column name or ``all`` value ordering the selection
        of all numeric columns.

    n_jobs: int, optional (default=None)
        The number of jobs to run in parallel for both ``fit`` and
        ``transform``.

    **tree_params: **kwargs, optional
        Parameters passed to ``sklearn.tree.DecisionTreeClassifier``.

    save_estimators: boolean, optional (default=False)
        Determine whether each tree for a single continuous feature specified
        is to be saved into ``estimators_``.

    Attributes
    ----------
    columns_: list
        The collection of discretized columns with enforced type.

    estimator_: scikit-learn estimator
        Estimator used for acquiring single feature's cut points.

    estimators_: list
        List of fitted estimators for corresponding continuous column. Exists
        only when ``save_estimators`` was set to ``True``.

    cut_points_: list
        The collection of cut points for corresponding columns.

    cuts_: dictionary-like
        The collection of cut points for every discretized column, used to
        determine bins, where the key is the column name and the value is the
        collection of corresponding cut points.

    """
    def __init__(self, columns="all", n_jobs=None, criterion="gini",
                 splitter="best", max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0,
                 max_features=None, max_leaf_nodes=None,
                 min_impurity_decrease=0, min_impurity_split=None,
                 class_weight=None, presort=False, random_state=None,
                 save_estimators=False):
        super().__init__(
            columns=columns,
            n_jobs=n_jobs,
            base_estimator=DecisionTreeClassifier(),
            estimator_params=(
                "criterion", "splitter", "max_depth", "min_samples_split",
                "min_samples_leaf", "min_weight_fraction_leaf", "max_features",
                "max_leaf_nodes", "min_impurity_decrease", "min_impurity_split",
                "class_weight", "presort", "random_state"
            ),
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            presort=presort,
            random_state=random_state,
            save_estimators=save_estimators
        )


class RegressionTreeDiscretizer(TreeDiscretizer):
    """Discretize continuous features with use of decision tree algorithm.

    Parameters
    ----------
    columns: string {<column-name>, all} or list, optional (default="all")
        The collection of discretized column names. If string value is passed
        requires numeric column name or ``all`` value ordering the selection
        of all numeric columns.

    n_jobs: int, optional (default=None)
        The number of jobs to run in parallel for both ``fit`` and
        ``transform``.

    **tree_params: **kwargs, optional
        Parameters passed to ``sklearn.tree.DecisionTreeRegressor``.

    save_estimators: boolean, optional (default=False)
        Determine whether each tree for a single continuous feature specified
        is to be saved into ``estimators_``.

    Attributes
    ----------
    columns_: list
        The collection of discretized columns with enforced type.

    estimator_: scikit-learn estimator
        Estimator used for acquiring single feature's cut points.

    estimators_: list
        List of fitted estimators for corresponding continuous column. Exists
        only when ``save_estimators`` was set to ``True``.

    cut_points_: list
        The collection of cut points for corresponding columns.

    cuts_: dictionary-like
        The collection of cut points for every discretized column, used to
        determine bins, where the key is the column name and the value is the
        collection of corresponding cut points.

    """
    def __init__(self, columns="all", n_jobs=None, criterion="mse",
                 splitter="best", max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0,
                 max_features=None, max_leaf_nodes=None,
                 min_impurity_decrease=0, min_impurity_split=None,
                 presort=False, random_state=None, save_estimators=False):
        super().__init__(
            columns=columns,
            n_jobs=n_jobs,
            base_estimator=DecisionTreeRegressor(),
            estimator_params=(
                "criterion", "splitter", "max_depth", "min_samples_split",
                "min_samples_leaf", "min_weight_fraction_leaf", "max_features",
                "max_leaf_nodes", "min_impurity_decrease", "min_impurity_split",
                "presort", "random_state"
            ),
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            presort=presort,
            random_state=random_state,
            save_estimators=save_estimators
        )


class SpearmanDiscretizer(BaseDiscretizer):
    def __init__(self):
        import warnings

        from .exceptions import DevelopmentStageWarning


        warnings.warn(
            "This funcitonality is still in the development stage, "
            "you are using it on your own risk.",
            DevelopmentStageWarning
        )
