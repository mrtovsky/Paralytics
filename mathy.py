def lcm(x, y):
    """Calculates least common multiple of two integers x and y.
    """
    assert isinstance(x+y, int), 'Values must be integers!'

    lcm_value = x if x > y else y
    while lcm_value%x or lcm_value%y:
        lcm_value += 1
    return lcm_value


