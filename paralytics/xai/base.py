"""Base classes for all estimators across paralytics.xai subpackage."""


from abc import ABCMeta, abstractmethod


__all__ = [
    'ExplainerMixin'
]


class ExplainerMixin(metaclass=ABCMeta):
    """Mixin class for all AI explainers in paralytics.xai subpackage."""
    _estimator_type = "explainer"

    @abstractmethod
    def fit(self, X, y):
        """Fit to data."""

    @abstractmethod
    def explain(self):
        """Explain the black box returning object that visualizes properties.

        Notes
        -----
        It should ultimately return the plot showing previously obtained
        properties in the fitting phase. Sometimes it is insufficient or
        impossible, hence it is acceptable to return a concise value or the
        description itself.

        """

    def fit_explain(self, X, y=None, **fit_params):
        """Fit to data, then explain properties."""
        if y is None:
            return self.fit(X, **fit_params).explain(X)
        else:
            return self.fit(X, y, **fit_params).explain(X, y)
