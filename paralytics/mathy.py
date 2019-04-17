from math import ceil, sqrt


__all__ = [
    'lcm',
    'custom_round',
    'check_prime'
]


def lcm(x, y):
    """Calculates least common multiple of two integers x and y."""
    assert isinstance(x + y, int), 'Values must be integers!'

    lcm_value = x if x > y else y
    while lcm_value % x or lcm_value % y:
        lcm_value += 1
    return lcm_value


def custom_round(value_list, n):
    """Rounds n to the closest value from the given value_list."""
    return min(value_list, key=lambda x: abs(x - n))


def check_prime(n):
    """Checks whether the given value is a prime number."""
    assert (isinstance(n, int) and n >= 0), \
        'Expected integer greater than or equal to 1 as parameter!'

    if n in (0, 1):
        return False
    elif not n % 2 and n > 2:
        return False
    else:
        for divider in range(3, ceil(sqrt(n)) + 1, 2):
            if not n % divider:
                return False
        return True
