import warnings

from functools import reduce
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from joblib import delayed, Parallel
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

from .base import ExplainerMixin
from ..utils import check_column_existence, is_numeric


__all__ = [
    'FeatureEffectExplainer'
]


class FeatureEffectExplainer(BaseEstimator, ExplainerMixin):
    """Visualizes the effect of one or two features on the prediction.

    Parameters
    ----------
    estimator: TODO

    features: str or list of length at most 2
        TODO: Grid features.

    dtypes: dict, optional (default=None)
        Types of the passed features. Possible values: 'numeric' or 'category'.
        Has to be passed as a dictionary where the key is the name of the
        feature for which data type is specified. Left by default it is
        determined automatically during `fit` method execution.

        Based on this parameter the appropriate explainers are selected.

    sample_size: int or float, optional (default=None)
        TODO

    estimation_values: int or dict, optional (default=100)
        Declares number of values to generate for the grid feature or explicitly
        specifies those values. When passed as:

        - `int`:

          Automatically detects whether the grid feature is numeric and if:

          - `True`:

            Generates the set of values from the lowest to the highest value
            recorded in the data set passed to the fit method with the
            interspace depending on the number of values to generate specified
            in the `n_estimated_values` parameter.

          - `False`:

            Takes all of the explained feature's unique values and imputes
            into grid feature. When you need to consider only a subset of the
            unique categories, pass them to the dictionary with a key being
            name of the feature.

          When two features are specified then takes the given value for both
          of them.

        - `dict`:

          Manually specify the values or pass separately for every feature how
          many values to generate. Dictionary specification:

          - `key`:

            Feature name passed to the `features` parameter.

          - `value`:

            Integer indicating how many values generate between the lowest and
            the highest value recorded in the data set or array of values with
            which grid feature will be imputed to make predictions for the
            synthetic data set.

    n_jobs: TODO

    random_state: int, optional (default=None)
        Seed for the sample generator. Used when `sample_size` is not None.

    Attributes
    ----------
    dtypes_: list
        Actual data types of grid features after evaluation if the automatic
        determination was specified. Otherwise is equal to dtypes but converted
        to a list where order ise the same as the order of passed features.

    estimation_values_: list
        Actual estimation values used to calculate dependency plots. The order
        of values is the same as the order of passed features.

    base_values_: np.array, shape = (n_samples, n_grid_features)
        TODO

    grid_values_: np.array, shape = (n_grid_values, n_grid_features)
        TODO

    y_grid_predictions_: np.array, shape = (n_samples, n_grid_values)
        Array of predictions for every grid values set where rows are
        predictions for consecutive observations.

    References
    ----------
    [1] C. Molnar, `Interpretable Machine Learning
    <https://christophm.github.io/interpretable-ml-book/pdp.html>`_, 2019

    """
    CORRECT_DTYPES = {'numeric', 'category'}

    def __init__(self, estimator, features, dtypes=None, sample_size=None,
                 estimation_values=100, n_jobs=None, random_state=None):
        self.estimator = estimator
        self.features = features
        self.dtypes = dtypes
        self.sample_size = sample_size
        self.estimation_values = estimation_values
        self.n_jobs = n_jobs
        self.random_state = random_state

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, features):
        if isinstance(features, str):
            features = [features]
        else:
            assert all([isinstance(feature, str) for feature in features]), (
                "Passing multiple grid features requires an array-like object "
                "of string values being the names of features."
            )
            features = list(features)
            assert len(features) <= 2, "Maximum two grid features are allowed."

        self._features = features

    @property
    def dtypes(self):
        return self._dtypes

    @dtypes.setter
    def dtypes(self, dtypes):
        if dtypes is not None:
            assert isinstance(dtypes, dict), (
                "When manually specifying the data types of grid features the "
                "dictionary is required where the key is the feature's name."
            )
            assert set(dtypes.keys()) == set(self.features), (
                "Manual data types specification of grid features requires "
                "passing appropriate values for every feature given in the "
                "`features` parameter. It should be a 'numeric' string when "
                "the feature is of numerical type and 'category' string "
                "otherwise."
            )
            assert set(dtypes.values()) <= self.CORRECT_DTYPES, (
                "Unavailable data type specified. Data types should be passed "
                "as strings from the given set: {}.".format(self.CORRECT_DTYPES)
            )

        self._dtypes = dtypes

    @property
    def estimation_values(self):
        return self._estimation_values

    @estimation_values.setter
    def estimation_values(self, values):
        if isinstance(values, int):
            values = {feature: values for feature in self.features}
        else:
            assert isinstance(values, dict), (
                "When manually specifying the estimation values for grid "
                "features separately the dictionary is required where the key "
                "is the feature's name."
            )
            assert set(values.keys()) == set(self.features), (
                "Manual values specification for grid features requires "
                "passing appropriate values for every feature given in the "
                "`features` parameter. It can be array-like object with "
                "explicitly declared values or integer indicating how many "
                "values shall be generated."
            )

        self._estimation_values = values

    def fit(self, X, y=None):
        """Fits creation of synthetic data to X.

        Parameters
        ----------
        X: pandas.DataFrame
            TODO

        y: ignore

        Returns
        -------
        self: object
            Returns the instance itself.

        """
        check_column_existence(X, self.features)

        if self.dtypes is None:
            self.dtypes_ = [
                'numeric' if is_numeric(X[feature]) else 'category'
                for feature in self.features
            ]
        else:
            self.dtypes_ = [self.dtypes[feature] for feature in self.features]

        self.estimation_values_ = self._determine_estimation_values(X)

        X_sample = self.select_sample(X)
        self.base_values_ = X_sample[self.features].values

        self.grid_values_, self.y_grid_predictions_ = \
            self.predict_grid_features(X_sample)

        return self

    def explain(self, pdplot=True, iceplot=False, mplot=False, aleplot=False,
                automatic_layout=True, centers=None, iceplot_thresh=None,
                neighborhoods=.1, pdline_params=None, iceline_params=None,
                mline_params=None, aleline_params=None, contour_params=None,
                contourf_params=None, bar_params=None, imshow_params=None,
                text_params=None, verbose=True, ax=None):
        """Explains the features effect with use of the selected methods.

        Parameters
        ----------
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

        automatic_layout: bool, optional (default=True)
            Specified whether format the plots in the automatic manner including
            ticks adjustment, axis signing, text formatting etc. or leave
            the plot in the raw state.

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

        iceplot_thresh: int or float, optional (default=None)
            Declares how many observations to take to visualize the ICE plots.
            If `int`, gives the exact number of observations, if `float`, gives
            a fraction of all observations to be taken.

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

        {pd, ice, m, ale}plot_params: dicts, optional (default=None)
            Keyword arguments for underlying plotting functions.

        verbose: TODO

        ax: TODO

        Returns
        -------
        TODO

        """
        assert hasattr(self, 'y_grid_predictions_'), (
            'Could not find the attribute.\n'
            'Fitting is necessary before you do the transformation.'
        )

        if len(self.features) == 2:
            assert sum([pdplot, mplot, aleplot]) <= 1, (
                'When explaining two features it is possible to plot only one '
                'of: `pdplot`, `mplot`, `aleplot`.'
            )

        features_are_numeric = [dtype == 'numeric' for dtype in self.dtypes_]

        if isinstance(neighborhoods, int):
            neighborhoods = [
                neighborhoods for feature_is_numeric in features_are_numeric
                if feature_is_numeric
            ]
        elif isinstance(neighborhoods, float):
            # TODO: Think of method of turning fraction to absolute value.
            pass
        else:
            neighborhoods = list(neighborhoods)
            assert len(neighborhoods) == sum(features_are_numeric), (
                "Neighborhoods can be declared for numeric features only "
                "and its length must be the same size as the number of "
                "specified grid features."
            )

        if centers is not None:
            assert all(features_are_numeric), (
                "Centering is only available for numerical features, because "
                "it needs to know the relations between feature's values to "
                "extract values higher or equal than the centering value.\n"
                "Consider not setting the `center` parameter or pass numerical "
                "features and re-fit the explainer."
            )

            if isinstance(centers, (int, float, str)):
                centers = [centers for _ in range(len(self.features))]
            else:
                centers = list(centers)
                assert len(centers) == len(self.features), (
                    'The number of declared center values must be equal to the '
                    'number of features specified to explanation.'
                )

            grid_values, y_grid = self._center_grid(centers)
        else:
            grid_values = self.grid_values_
            y_grid = self.y_grid_predictions_

        # Get current axis if none has been specified.
        if ax is None:
            ax = plt.gca()

        # Set default plots parameters.
        if contour_params is None:
            contour_params = {'linewidths': .5, 'colors': 'white'}

        if contourf_params is None:
            # TODO
            contourf_params = {}

        if bar_params is None:
            # TODO
            bar_params = {}

        if imshow_params is None:
            imshow_params = {'origin': 'lower', 'aspect': 'auto'}

        if text_params is None:
            text_params = {'ha': 'center', 'va': 'center', 'color': 'white'}

        if iceplot:
            one_feature = len(self.features) == 1
            assert one_feature and is_numeric(self.base_values_.flatten()), (
                'When two features are specified or the feature is categorical '
                'the `iceplot` parameter must be False!\nExplaining two '
                'features effect on predictions with use of Individual '
                'Conditional Expectation plots requires displaying single '
                'two-dimensional plane for every sample and even though it is '
                'possible it would give zero value due to lack of readability.'
            )

            if iceline_params is None:
                iceline_params = {'color': '#ACDBD9'}

            ax = self._plot_iceplot(
                grid_values=grid_values,
                predictions=y_grid,
                thresh=iceplot_thresh,
                line_params=iceline_params,
                ax=ax
            )

        if pdplot:
            if pdline_params is None:
                pdline_params = {'linewidth': 2, 'color': '#A6E22E'}

            ax = self._plot_pdplot(
                grid_values=grid_values,
                predictions=y_grid,
                features_are_numeric=features_are_numeric,
                automatic_layout=automatic_layout,
                line_params=pdline_params,
                contour_params=contour_params,
                contourf_params=contourf_params,
                bar_params=bar_params,
                imshow_params=imshow_params,
                text_params=text_params,
                ax=ax
            )

        if mplot:
            if mline_params is None:
                mline_params = {'linewidth': 2, 'color': '#FF45F9'}

            assert any(features_are_numeric), (
                "When plotting M-Plot at least one feature needs to be of "
                "numeric type. Otherwise, there is no point in calculating "
                "the unrealistic observations in the sense of the Euclidean "
                "distance for categorical features. If you want to visualize "
                "the effect of two features, ceteris paribus, just plot PDPlot "
                "instead."
            )

            ax = self._plot_mplot(
                grid_values=grid_values,
                predictions=y_grid,
                features_are_numeric=features_are_numeric,
                neighborhoods=neighborhoods,
                automatic_layout=automatic_layout,
                line_params=mline_params,
                contour_params=contour_params,
                contourf_params=contourf_params,
                imshow_params=imshow_params,
                verbose=verbose,
                ax=ax
            )

        return ax

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

    def predict_grid_features(self, X):
        """Predicts previously substituting grid features with generated values.

        For every combination in the cartesian product of unique grid features
        generates a temporary DataFrame containing this set of values across the
        whole grid features leaving the rest of the features unchanged. Then
        makes prediction for this synthetic DataFrame.

        Returns list of predictions for every synthetic DataFrame and list of
        grid values which replaced the original values across the grid features
        to create these DataFrames for prediction.

        """
        assert hasattr(self, 'estimation_values_'), (
            'Could not find the attribute.\n'
            'Synthetic data preparation is only possible after estimation '
            'values are generated.'
        )

        grid_values, y_grid_predictions = zip(*Parallel(n_jobs=self.n_jobs)(
            delayed(self._predict_single_grid_features)(X, grid_values)
            for grid_values in product(*self.estimation_values_)
        ))

        y_grid_predictions = np.stack(y_grid_predictions).T
        grid_values = np.stack(grid_values)

        return grid_values, y_grid_predictions

    def _determine_estimation_values(self, X):
        """Determines estimation values for grid features."""
        assert hasattr(self, "dtypes_"), (
            "Could not find the attribute.\n"
            "Dtypes evaluation is necessary before the values estimation."
        )

        estimation_values = []
        for feature, dtype in zip(self.features, self.dtypes_):
            values = self.estimation_values[feature]

            if isinstance(values, int) and dtype == "numeric":
                est_values = np.linspace(
                    start=X[feature].astype(np.number).min(),
                    stop=X[feature].astype(np.number).max(),
                    num=values
                )
            elif isinstance(values, int):
                est_values = np.unique(X[feature])
            else:
                est_values = np.sort(values)

            estimation_values.append(est_values)

        return estimation_values

    def _predict_single_grid_features(self, X, grid_values):
        """Makes prediction for single combination of grid features' values."""
        X_grid = X.assign(**dict(zip(self.features, grid_values)))
        try:
            y_grid_preds = self.estimator.predict_proba(X_grid)[:, 1]
        except AttributeError:
            y_grid_preds = self.estimator.predict(X_grid)

        return grid_values, y_grid_preds

    def _center_grid(self, centers):
        """Centers the grid values and predictions to the given list of values.

        Every set of values that is lower than the specified center values
        is removed from the grid and for the other values the central values
        are subtracted from grid values and prediction value for centers is
        subtracted from the grid predictions.

        """
        centers = [
            self.base_values_[:, idx].min()
            if center == 'min' else center
            for idx, center in enumerate(centers)
        ]

        above_central_values = np.all(self.grid_values_ >= centers, axis=1)
        grid_values = self.grid_values_[above_central_values, :]
        y_grid = self.y_grid_predictions_[:, above_central_values]
        y_grid = y_grid - y_grid[:, 0].reshape(-1, 1)

        return grid_values, y_grid

    def _plot_iceplot(self, grid_values, predictions, thresh, line_params, ax):
        """Plots Individual Conditional Expectation."""
        grid_values = np.array(grid_values)
        predictions = np.array(predictions)

        random_state = check_random_state(self.random_state)

        if isinstance(thresh, float):
            thresh = int(len(predictions) * thresh)
        try:
            more_indexes_than_thresh = len(predictions) > thresh
        except TypeError:
            more_indexes_than_thresh = False

        # Select observations to plot Ceteris Paribus profiles for.
        if more_indexes_than_thresh:
            predictions = predictions[random_state.choice(
                len(predictions),
                size=thresh,
                replace=False
            )]

        grid_values = grid_values.flatten()
        for prediction in predictions:
            ax.plot(
                grid_values,
                prediction,
                **line_params
            )

        return ax

    def _plot_pdplot(self, grid_values, predictions, features_are_numeric,
                     automatic_layout, line_params, contour_params,
                     contourf_params, bar_params, imshow_params, text_params,
                     ax):
        """Plots Partial Dependence Plot."""
        grid_values = np.array(grid_values)
        predictions = np.array(predictions)

        predictions_mean = np.mean(predictions, axis=0)

        if all(features_are_numeric):
            ax = self._plot_numerics(
                grid_values, predictions_mean, line_params,
                contour_params, contourf_params, ax
            )
        elif any(features_are_numeric):
            ax = self._plot_category_numeric(
                grid_values, predictions_mean, features_are_numeric,
                automatic_layout, imshow_params, ax
            )
        else:
            ax = self._plot_categories(
                grid_values, predictions_mean, automatic_layout,
                bar_params, imshow_params, text_params, ax
            )

        return ax

    def _plot_mplot(self, grid_values, predictions, features_are_numeric,
                    neighborhoods, automatic_layout, line_params,
                    contour_params, contourf_params, imshow_params, verbose,
                    ax):
        """Plots Marginal Plot."""
        grid_values = np.array(grid_values)
        predictions = np.array(predictions)

        predictions = self._replace_unreal_obs_with_nan(
            grid_values, predictions, features_are_numeric, neighborhoods
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            predictions_mean = np.nanmean(predictions, axis=0)
            try:
                warn_encountered = issubclass(w[-1].category, RuntimeWarning)
            except IndexError:
                warn_encountered = False
            if verbose and warn_encountered:
                print(
                    "With a given `neighborhoods` and `estimation_values`, "
                    "some values were not considered realistic for any "
                    "observation in the data set and therefore these places "
                    "will be presented as blank in the final plot.\n"
                    "In order to eliminate them, it is worth considering "
                    "changing the parameters mentioned above or getting rid of "
                    "outliers from the data set.\n"
                    "It is worth remembering that often empty spaces in the "
                    "plot will indicate truly unrealistic values, which will "
                    "make them desirable to keep illustrating.\n"
                    "To silence this message set `verbose` to False."
                )

        if all(features_are_numeric):
            ax = self._plot_numerics(
                grid_values, predictions_mean, line_params,
                contour_params, contourf_params, ax
            )
        else:
            ax = self._plot_category_numeric(
                grid_values, predictions_mean, features_are_numeric,
                automatic_layout, imshow_params, ax
            )

        return ax

    def _plot_numerics(self, grid_values, predictions_mean, line_params,
                       contour_params, contourf_params, ax):
        """Plots Partial Dependence Plot for numeric grid features only."""
        if len(self.features) == 1:
            ax.plot(
                grid_values.flatten(),
                predictions_mean,
                **line_params
            )
        else:
            x_shape = len(np.unique(grid_values[:, 0]))

            feature_x = grid_values[:, 0].reshape(x_shape, -1)
            feature_y = grid_values[:, 1].reshape(x_shape, -1)
            values = predictions_mean.reshape(x_shape, -1)

            contourf = ax.contourf(
                feature_x,
                feature_y,
                values,
                **contourf_params
            )
            ax.contour(
                feature_x,
                feature_y,
                values,
                **contour_params
            )

            ax.figure.colorbar(contourf, ax=ax)

            ax.set_xlabel('{}'.format(self.features[0]))
            ax.set_ylabel('{}'.format(self.features[1]))

        return ax

    def _plot_category_numeric(self, grid_values, predictions_mean,
                               features_are_numeric, automatic_layout,
                               imshow_params, ax):
        """Plots Partial Dependence Plot for category vs. numeric features."""
        idx_num, idx_cat = (0, 1) if features_are_numeric[0] else (1, 0)

        feature_num = grid_values[:, idx_num].astype(np.number)
        feature_cat = grid_values[:, idx_cat]

        feature_num_unique = np.unique(feature_num)
        feature_cat_unique = np.unique(feature_cat)

        if idx_cat:
            y_shape = len(feature_num_unique)
        else:
            y_shape = len(feature_cat_unique)

        values = predictions_mean.reshape((-1, y_shape), order='F')

        imshow = ax.imshow(
            values,
            **imshow_params
        )
        ax.figure.colorbar(imshow, ax=ax)

        if automatic_layout:
            span_range = abs(feature_num_unique[-1] - feature_num_unique[0])
            num_format = "{0:.0f}" if span_range > 10 else "{0:.2f}"
            num_labels = [
                num_format.format(value) for value in feature_num_unique
            ]
        else:
            num_labels = feature_num_unique

        # FIXME: More wet than DRY. Rewrite for less spaghetti code.
        if idx_cat:
            ax.set_xticks(np.arange(len(feature_num_unique)))
            ax.set_yticks(np.arange(len(feature_cat_unique)))

            ax.set_xticklabels(num_labels)
            ax.set_yticklabels(feature_cat_unique)

            if automatic_layout:
                tick_spacing = int(len(feature_num_unique) / 5)
                for idx, tick in enumerate(ax.get_xticklabels()):
                    if idx % tick_spacing:
                        tick.set_visible(False)

                ax.set_xlabel('{}'.format(self.features[idx_num]))
                ax.set_ylabel('{}'.format(self.features[idx_cat]))
        else:
            ax.set_xticks(np.arange(len(feature_cat_unique)))
            ax.set_yticks(np.arange(len(feature_num_unique)))

            ax.set_xticklabels(feature_cat_unique)
            ax.set_yticklabels(num_labels)

            if automatic_layout:
                tick_spacing = int(len(feature_num_unique) / 5)
                for idx, tick in enumerate(ax.get_yticklabels()):
                    if idx % tick_spacing:
                        tick.set_visible(False)

                ax.set_xlabel('{}'.format(self.features[idx_cat]))
                ax.set_ylabel('{}'.format(self.features[idx_num]))

        return ax

    def _plot_categories(self, grid_values, predictions_mean,
                         automatic_layout, bar_params, imshow_params,
                         text_params, ax):
        """Plots Partial Dependence Plot for categorical features only."""
        if len(self.features) == 1:
            ax.bar(
                x=grid_values.flatten(),
                height=predictions_mean,
                **bar_params
            )

            if automatic_layout:
                ax.set_xlabel("{}".format(self.features[0]))
                ax.set_ylabel("Average Prediction")
        else:
            # Preserves the order of occurrence.
            feature_x_unique = reduce(
                lambda l, x: l.append(x) or l if x not in l else l,
                grid_values[:, 0],
                []
            )
            n_feature_x_unique = len(feature_x_unique)
            n_feature_y_unique = len(np.unique(grid_values[:, 1]))
            feature_y_unique = grid_values[:n_feature_y_unique, 1].tolist()
            values = predictions_mean.reshape(
                (len(feature_y_unique), -1), order='F'
            )

            imshow = ax.imshow(
                values,
                **imshow_params
            )
            ax.figure.colorbar(imshow, ax=ax)

            ax.set_xticks(np.arange(n_feature_x_unique))
            ax.set_yticks(np.arange(n_feature_y_unique))

            ax.set_xticklabels(feature_x_unique)
            ax.set_yticklabels(feature_y_unique)

            ax.set_xlabel('{}'.format(self.features[0]))
            ax.set_ylabel('{}'.format(self.features[1]))

            if automatic_layout:
                plt.setp(
                    ax.get_xticklabels(), rotation=45,
                    ha="right", rotation_mode="anchor"
                )

                span_range = abs(np.max(values) - np.min(values))
                text_format = "{0:.0f}" if span_range > 10 else "{0:.2f}"
            else:
                text_format = "{0}"

            indexes_product = product(
                range(n_feature_x_unique), range(n_feature_y_unique)
            )

            for i, j in indexes_product:
                ax.text(
                    i, j, text_format.format(values[j, i]), **text_params
                )

        return ax

    def _replace_unreal_obs_with_nan(self, grid_values, predictions,
                                     features_are_numeric, neighborhoods):
        """Replaces unrealistic observations with NaN value.

        Prepares the predictions for plotting mplot by replacing predictions
        with nans for grid values which distance from the real values ​​on which
        they were generated is higher than defined by `neighborhoods`.

        """
        grid_values_num = grid_values[:, features_are_numeric].astype(np.number)

        for idx in range(len(self.base_values_)):
            base_values_num = self.base_values_[idx, features_are_numeric]

            borde_inferior = grid_values_num - neighborhoods <= base_values_num
            borde_superior = grid_values_num + neighborhoods >= base_values_num
            unrealistic_obs = np.any(~(borde_inferior & borde_superior), axis=1)

            predictions[idx, unrealistic_obs] = np.nan

        return predictions
