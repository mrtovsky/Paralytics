"""
This module includes all custom warnings and error classes used across 
scitkit-learn.
"""

__all__ = ['UniqueValuesError']


class UniqueValuesError(ValueError):
    """Exception class to raise if number of unique values is not matching
    the required threshold.
    """
