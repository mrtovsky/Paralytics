"""
This module includes all custom warnings and error classes used across
paralytics.
"""
__all__ = [
    'UniqueValuesError',
    'NothingSelectedWarning'
]


class UniqueValuesError(ValueError):
    """Exception class to raise if number of unique values is not matching
    the required threshold."""


class NothingSelectedWarning(UserWarning):
    """Warning used to notify user that nothing has been selected.

    This warning notifies the user that after executing the script nothing
    has been selected where it is expected not to receive an empty object.

    For example we can thus emphasize that the user forgot to implement the
    appropriate step in the pipeline.
    """


class DevelopmentStageWarning(UserWarning):
    """Warning used to notify that functionality is in the development stage.

    Warning raised when functionality has not yet been fully implemented and
    tested, and thus may cause unexpected errors.
    """
