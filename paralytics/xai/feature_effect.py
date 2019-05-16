import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator

from .base import ExplainerMixin
from ..utils.validation import (
    check_column_existence, check_continuity, is_numeric
)


__all__ = [
    'FeatureEffectExplainer'
]


class FeatureEffectExplainer(BaseEstimator, ExplainerMixin):
    """Visualizes the effect of one or two features on the prediction.

    Parameters
    ----------
    estimator:

    features: str or list of length at most 2

    sample_size: int or float, optional (default=None)

    estimation_method: string: {'auto', 'span', 'observed'},
    optional (default='auto')
        Estimation method to determine the set of values of the explained
        variable to generate the synthetic data set containing all of the
        unique values of the obtained explained variable crossed with the
        rest of the observed values of independent variables excluding the
        grid features.

        - `auto`:

          Checks whether passed variable is truly continuous and based on the
          result of this test, selects the appropriate method from remaining.

        - `span`:

          Calculates the lower and the upper bound of the passed variable and
          spans the obtained range. Recommended for numeric features, because
          it also covers the values not included in the given data set.

        - `observed`:

          Takes all of the unique values of the explained features observed in
          the given data set and imputes into grid features. Performed by
          default for categorical variables.

    n_span_values: int, optional (default=1000)
        Declares number of unique values generated for grid feature when `span`
        estimation method is selected.

    pdplot: bool, optional (default=True)
        Defines if Partial Dependence Plot should be displayed. It visualizes
        marginal effect that grid features have on predictions with use of the
        Monte Carlo method.

    mplot: bool, optional (default=False)
        Defines if Marginal Plot should be displayed. It visualizes conditional
        effect that grid features have on predictions. Only possible for
        numeric features.

    neighborhoods: int or float or list, optional (default=.1)
        Neighborhood of the value to determine the interval
        `[current_value - neighborhood, current_value + neighborhood]`
        for which predictions will be averaged. Taken under consideration
        only when `mplot == True`. If:

        - `int`:

          Absolute value that will be deducted and added from the current
          value to determine the interval for synthetic data generation.

        - `float`:

          Fraction of the difference between biggest and smallest value in the
          variable to calculate the interval boundaries.

        Should be passed as the list of values mentioned above if two features
        are passed to explanation, in the same order in which the features are
        given.

    ice: bool, optional (default=True)
        Defines whether Individual Conditional Expectation plots should be
        displayed. Only possible if a single feature is explained.

    random_state: int, optional (default=None)
        Seed for the sample generator. Used when `sample_size` is not None.

    References
    ----------
    [1] C. Molnar, `Interpretable Machine Learning
    <https://christophm.github.io/interpretable-ml-book/pdp.html>`_, 2019

    """
    def __init__(self, estimator, features, sample_size=None,
                 estimation_method='auto', n_span_values=1000,
                 pdplot=True, mplot=False, neighborhoods=.1,
                 ice=False, random_state=None):
        self.estimator = estimator
        self.features = features
        self.sample_size = sample_size
        self.estimation_method = estimation_method
        self.n_span_values = n_span_values
        self.pdplot = pdplot
        self.mplot = mplot
        self.neighborhoods = neighborhoods
        self.ice = ice
        self.random_state = random_state

    def fit(self, X, y):
        """"""
        features = self._validate_input(X)
        X_sample = self._select_sample(X)
        estimation_methods = self._determine_estimation_methods(X, features)
        features_cartesian = self._generate_features_cartesian(
            X, features, estimation_methods
        )
        X_synthetic = self._prepare_synthetic_data(
            X_sample, features, features_cartesian
        )

        if self.ice:
            pass
        if self.pdplot:
            pass
        if self.mplot:
            pass

    def explain(self):
        fig, ax = plt.subplots()

    def _validate_input(self, X):
        """Validates whether input values were passed correctly.

        If everything is ok returns the features' names converted into a list.

        """
        if isinstance(self.features, str):
            features = [self.features]
        else:
            features = list(self.features)

        assert check_column_existence(X, features), \
            'Specified features do not exist in the given DataFrame.'

        assert len(features) <= 2, (
            'Too much features passed to explanation!\n'
            'More than two features require dimensions higher than 3 to '
            'visualize their joint effect on predictions hence the '
            'interpretability is very limited.'
        )

        if len(features) == 2:
            assert not self.ice, (
                'When two features are specified the `ice` parameter must be '
                'False!\nExplaining two features effect on predictions with '
                'use of Individual Conditional Expectation plots requires '
                'displaying single two-dimensional plane for every sample '
                'and even though it is possible it would give zero value due '
                'to lack of readability.'
            )

        if self.mplot:
            assert all([is_numeric(X[feature]) for feature in features]), (
                'Marginal plot can be drawn only for numerical features, '
                'because it requires to create a range of near values and to '
                'do so it has to calculate a distance, which is unclear for '
                'categorical features.\nConsider changing `mplot` to False or '
                'pass different features.'
            )
            if isinstance(self.neighborhoods, (int, float)):
                neighborhoods = [self.neighborhoods]
            else:
                neighborhoods = self.neighborhoods
            valid_neighborhoods = all([
                isinstance(neighborhood, (int, float))
                for neighborhood in neighborhoods
            ])
            assert valid_neighborhoods, (
                'Neighborhoods must be specified as int or float or a list '
                'consisting of a combination of those two types!'
            )
            assert len(neighborhoods) == len(features), (
                'The number of neighborhoods must be equal to the number of '
                'features!'
            )

        return features

    def _select_sample(self, X):
        """Selects sample data with `sample_size` number of samples."""
        if self.sample_size is not None:
            try:
                X_sample = X.sample(
                    n=self.sample_size, random_state=self.random_state
                )
            except ValueError:
                X_sample = X.sample(
                    frac=self.sample_size, random_state=self.random_state
                )
        else:
            X_sample = X.copy()

        return X_sample

    def _determine_estimation_methods(self, X, features):
        """Determines which method of generating unique values to use.

        Requires to pass the features as a list (even in case when only one
        feature is selected).

        """
        if self.estimation_method == 'auto':
            estimation_methods = [
                'span' if check_continuity(X[feature]) else 'observed'
                for feature in features
            ]
        elif self.estimation_method == 'span':
            estimation_methods = [
                'span' if is_numeric(X[feature]) else 'observed'
                for feature in features
            ]
        else:
            estimation_methods = ['observed' for _ in features]

        return estimation_methods

    def _generate_features_cartesian(self, X, features, estimation_methods):
        """Generates Cartesian product of unique features.

        Requires to pass the features as a list (even in case when only one
        feature is selected).
        Requires to pass the estimation_methods as list of strings declaring
        which estimation method to use for every feature).

        """
        features_unique_values = [
            np.linspace(
                start=X[feature].min(),
                stop=X[feature].max(),
                num=self.n_span_values
            ) if method == 'span' else X[feature].unique()
            for feature, method in zip(features, estimation_methods)
        ]

        if len(features) == 2:
            features_cartesian = np.array(
                np.meshgrid(*features_unique_values)
            ).T.reshape(-1, 2)
        else:
            features_cartesian = features_unique_values[0]

        return features_cartesian

    def _prepare_synthetic_data(self, X, features, features_values):
        """Prepares data by substituting grid features with passed values.

        For every row in the `features_values` array generates a dataframe
        containing this set of values across the whole grid features leaving
        the rest of the features unchanged.

        """
        # TODO: Consider leaving the original values of the X to mplot or
        # creating a special creation of X_synthetic when only mplot is
        # selected and not every of the prepared combination is useful.
        Xs_synthetic = [
            X.assign(**{
                'OriginalIndex': X.index,
                **dict(zip(features, features_value))
            })
            for features_value in features_values
        ]
        X_synthetic = pd.concat(Xs_synthetic, ignore_index=True, sort=False)

        return X_synthetic


