import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from itertools import product
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

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
    estimator: TODO

    features: str or list of length at most 2
        TODO

    sample_size: int or float, optional (default=None)
        TODO

    estimation_methods: string or list, optional (default=None)
        Estimation method to determine the set of values of the explained
        variable to generate the synthetic data set containing all of the
        unique values of the obtained explained variable crossed with the
        rest of the observed values of independent variables excluding the
        grid features. If left by default takes `auto` method for every feature.

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

        Should be passed as the list of strings mentioned above if two
        features are passed to explanation, in the same order in which the
        features are given.

    n_span_values: int, optional (default=1000)
        Declares number of unique values generated for grid feature when `span`
        estimation method is selected.

    random_state: int, optional (default=None)
        Seed for the sample generator. Used when `sample_size` is not None.

    Attributes
    ----------
    estimation_methods_: list
        Actual estimation methods used after features' type validation.

    X_synthetic_: pandas.DataFrame
        TODO

    References
    ----------
    [1] C. Molnar, `Interpretable Machine Learning
    <https://christophm.github.io/interpretable-ml-book/pdp.html>`_, 2019

    """
    CORRECT_ESTIMATION_METHODS = {'auto', 'span', 'observed'}

    def __init__(self, estimator, features, sample_size=None,
                 estimation_methods=None, n_span_values=1000,
                 random_state=None):
        self.estimator = estimator
        self.features = features
        self.sample_size = sample_size
        self.estimation_methods = estimation_methods
        self.n_span_values = n_span_values
        self.random_state = random_state

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, features):
        features = [features] if isinstance(features, str) else list(features)

        assert len(features) <= 2, (
            "Features must be passed as string or array-like object of strings "
            "containing no more than two columns' names."
        )

        self._features = features

    @property
    def estimation_methods(self):
        return self._estimation_methods

    @estimation_methods.setter
    def estimation_methods(self, methods):
        if isinstance(methods, str):
            methods = [methods]
        elif methods is not None:
            methods = list(methods)
        else:
            methods = ['auto' for _ in range(len(self.features))]

        assert len(methods) == len(self.features), \
            'Number of estimation methods must be equal to number of features.'

        assert not self.CORRECT_ESTIMATION_METHODS.difference(methods), (
            'Undefined estimation method is given.\n'
            'Available methods are: {}'.format(self.CORRECT_ESTIMATION_METHODS)
        )

        self._estimation_methods = methods

    def fit(self, X, y=None):
        """Fits creation of synthetic data to X.

        Parameters
        ----------
        X: pandas.DataFrame

        y: ignore

        Returns
        -------
        self: object
            Returns the instance itself.

        """
        assert check_column_existence(X, self.features), \
            'Specified features do not exist in the given DataFrame.'

        self.estimation_methods_ = self.determine_estimation_methods(X)
        X_sample = self.select_sample(X)
        X_synthetic, features_original = self._prepare_synthetic_data(X_sample)

        try:
            y_predicted = self.estimator.predict_proba(X_synthetic.drop(
                features_original + ['_BaseIndex'], axix=1
            ))[:, 1]
        except AttributeError:
            y_predicted = self.estimator.predict(X_synthetic.drop(
                features_original + ['_BaseIndex'], axix=1
            ))

        self.X_synthetic_ = X_synthetic[
            ['_BaseIndex', *features_original, *self.features]
        ].assign(Prediction=y_predicted).copy()

        return self

    def explain(self, pdplot=True, iceplot=False, mplot=False, aleplot=False,
                centers=None, iceplot_thresh=None, neighborhoods=.1,
                pdplot_params=None, iceplot_params=None, mplot_params=None,
                aleplot_params=None, ax=None):
        """Explains the features effect with use of the selected methods.

        pdplot: bool, optional (default=True)
            Defines if Partial Dependence Plot should be displayed. It
            visualizes marginal effect that grid features have on predictions
            with use of the Monte Carlo method.

        iceplot: bool, optional (default=False)
            Defines whether Individual Conditional Expectation plots should be
            displayed. Only possible if a single numeric feature is explained.

        mplot: bool, optional (default=False)
            Defines if Marginal Plot should be displayed. It visualizes
            conditional effect that grid features have on predictions. Only
            possible for numeric features.

        aleplot: bool, optional (default=False)
            Defines if Accumulated Local Effects Plot should be displayed. It
            visualizes accumulated differences between predictions based on the
            conditional distribution of the feature.

        centers: int or float or string or list, optional (default=None)
            Defines the center value that all of the predictions will be
            compared to and displayed as a difference in the prediction
            to this point. By default no centering is done. If:

            - `min`:

              Specifies that minimum of the grid features will be used for
              centering.

            Should be passed as the list of values mentioned above if two
            features are passed to explanation, in the same order in which the
            features are given.

        iceplot_thresh: int, optional (default=None)
            TODO

        neighborhoods: int or float or list, optional (default=.1)
            Neighborhood of the value to determine the interval
            `[current_value - neighborhood, current_value + neighborhood]`
            for which predictions will be averaged. Taken under consideration
            only when `mplot == True` or `aleplot == True`. If:

            - `int`:

              Absolute value that will be deducted and added from the current
              value to determine the interval for synthetic data generation.

            - `float`:

              Fraction of the difference between biggest and smallest value in
              the variable to calculate the interval boundaries.

            Should be passed as the list of values mentioned above if two
            features are passed to explanation, in the same order in which the
            features are given.

        {pd, ice, m, ale}plot_params: TODO

        ax: TODO

        """
        assert hasattr(self, 'X_synthetic_'), (
            'Could not find the attribute.\n'
            'Fitting is necessary before you do the transformation.'
        )

        if ax is None:
            ax = plt.gca()

        if centers is not None:
            if isinstance(centers, (int, float, str)):
                centers = [centers]
            else:
                centers = list(centers)

            assert len(centers) == len(self.features), (
                'The number of declared center values must be equal to the '
                'number of features specified to explanation.'
            )

            centers = [
                self.X_synthetic_['_Base' + self.features[0]].min()
                if center == 'min' else center
                for center in centers
            ]

        if iceplot:
            if iceplot_params is None:
                iceplot_params = {'color': '#ACDBD9'}
            self.plot_iceplot(
                center=centers[0],
                thresh=iceplot_thresh,
                ax=ax,
                **iceplot_params
            )

        if pdplot:
            if pdplot_params is None:
                pdplot_params = {'linewidth': 5, 'color': '#ECFF2A'}
            self.plot_pdplot(
                centers=centers,
                ax=ax,
                **pdplot_params
            )

        if mplot:
            if mplot_params is None:
                mplot_params = {'linewidth': 5, 'color': '#FF45F9'}
            self.plot_mplot(
                centers=centers,
                neighborhoods=neighborhoods,
                ax=ax,
                **mplot_params
            )

        return ax

    def determine_estimation_methods(self, X):
        """Determines estimation methods confronted with features' type."""
        features_methods_pairs = zip(self.features, self.estimation_methods)
        estimation_methods = []
        for feature, estimation_method in features_methods_pairs:
            span_condition = any([
                estimation_method == 'auto' and check_continuity(X[feature]),
                estimation_method == 'span' and is_numeric(X[feature])
            ])
            estimation_method = 'span' if span_condition else 'observed'
            estimation_methods.append(estimation_method)

        return estimation_methods

    def select_sample(self, X):
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

    def plot_iceplot(self, center, thresh, ax, **params):
        """Plots Individual Conditional Expectation."""
        random_state = check_random_state(self.random_state)

        indexes_ice = self.X_synthetic_['_BaseIndex'].unique()
        more_indexes_than_thresh = len(indexes_ice) > thresh

        # Select observations to plot Ceteris Paribus profiles for.
        if thresh is not None and more_indexes_than_thresh:
            indexes_ice = random_state.choice(
                indexes_ice, size=thresh, replace=False
            )

        for idx_ice in indexes_ice:
            X_observation = self.X_synthetic_[
                self.X_synthetic_['_BaseIndex'] == idx_ice,
                [self.features[0], 'Prediction']
            ]
            ax.plot(
                xdata=X_observation[self.features[0]].values,
                ydata=X_observation['Prediction'].values,
                **params
            )

    def plot_pdplot(self, centers, ax, **params):
        """Plots Partial Dependence Plot."""
        pass

    def plot_mplot(self, centers, neighborhoods, ax, **params):
        """Plots Marginal Plot."""
        pass

    def _prepare_synthetic_data(self, X):
        """Prepares data by substituting grid features with generated values.

        For every combination in the cartesian product of grid features
        generates a dataframe containing this set of values across the whole
        grid features leaving the rest of the features unchanged.

        Returns synthetic DataFrame and list of features names that store the
        information from the original data.

        """
        assert hasattr(self, 'estimation_methods_'), (
            'Could not find the attribute.\n'
            'Synthetic data preparation is only possible after estimation '
            'methods are matched to corresponding features.'
        )

        features_original = [
            '_Base' + feature for feature in self.features
        ]
        columns_original = features_original + ['_BaseIndex']
        columns_existing = set(columns_original).intersection(X.columns)

        assert not columns_existing, (
            "Temporary columns created in the synthetic data creation process "
            "are already existing in the passed data X. Common columns are: "
            "{}.\nC'mon, who names the columns this way? You can do better "
            "than that, start by changing those names!"
            .format(columns_existing)
        )

        features_unique_values = [
            np.linspace(
                start=X[feature].min(),
                stop=X[feature].max(),
                num=self.n_span_values
            ) if method == 'span' else X[feature].unique()
            for feature, method in zip(self.features, self.estimation_methods_)
        ]

        Xs_synthetic = [
            X.assign(**{
                '_BaseIndex': X.index.tolist(),
                **dict(zip(features_original, X[self.features].values.T)),
                **dict(zip(self.features, features_value))
            })
            for features_value in product(*features_unique_values)
        ]
        X_synthetic = pd.concat(Xs_synthetic, ignore_index=True, sort=False)

        return X_synthetic, features_original

    def _validate_explain_input(self, iceplot, iceplot_center, mplot,
                                neighborhoods):
        """TODO: Remove - add validation in every plot type at the beginning."""
        if iceplot:
            feature_is_numeric = is_numeric(
                self.X_synthetic_[self.features[0]]
            )
            assert len(self.features) == 1 and feature_is_numeric, (
                'When two features are specified or the feature is categorical '
                'the `ice` parameter must be False!\nExplaining two features '
                'effect on predictions with use of Individual Conditional '
                'Expectation plots requires displaying single two-dimensional '
                'plane for every sample and even though it is possible it '
                'would give zero value due to lack of readability.'
            )

            if iceplot_center is not None:
                # TODO: Check if given value is not lower than the minimum.
                pass

        if mplot:
            features_are_numeric = all([
                is_numeric(self.X_synthetic_[feature])
                for feature in self.features
            ])
            assert features_are_numeric, (
                'Marginal plot can be drawn only for numerical features, '
                'because it requires to create a range of near values and to '
                'do so it has to calculate a distance, which is unclear for '
                'categorical features.\nConsider changing `mplot` to False or '
                'pass different features and re-fit the explainer.'
            )

            if isinstance(neighborhoods, (int, float)):
                neighborhoods = [neighborhoods]
            else:
                neighborhoods = list(neighborhoods)

            valid_neighborhoods = all([
                isinstance(neighborhood, (int, float))
                for neighborhood in neighborhoods
            ])
            assert valid_neighborhoods, (
                'Neighborhoods must be specified as int or float or a list '
                'consisting of a combination of those two types!'
            )

            assert len(neighborhoods) == len(self.features), (
                'The number of neighborhoods must be equal to the number of '
                'features!'
            )
