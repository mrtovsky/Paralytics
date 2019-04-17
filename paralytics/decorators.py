"""
This module includes all custom decorators used across paralytics.
"""
__all__ = [
    'force_context_manager'
]


def force_context_manager(cls):
    """Prevents user from calling objects of decorated classes without using
    the `with` statement."""
    class Wrapper(object):
        original = cls

        def __init__(self, *args, **kwargs):
            check_exit = getattr(cls, '__exit__', None)
            if check_exit is None:
                raise NotImplementedError(
                    'Magic method: `__exit__` must be implemented in the '
                    'decorated class!'
                )
            self.args = args
            self.kwargs = kwargs

        def __enter__(self):
            self._wrapped_obj = cls(*self.args, **self.kwargs)
            return self._wrapped_obj

        def __exit__(self, *args, **kwargs):
            self._wrapped_obj.__exit__(*args, **kwargs)

        def __getattr__(self, attr):
            raise RuntimeError(
                'Object of the {0} should only be initialized with the '
                '`with` statement. Otherwise, the {0} methods will not '
                'be available.'.format(self.original.__name__)
            )
    Wrapper.__doc__ = cls.__doc__
    Wrapper.__name__ = cls.__name__

    return Wrapper
