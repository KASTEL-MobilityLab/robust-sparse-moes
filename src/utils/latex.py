def bold_formatter(x, value, num_decimals=2):
    """Format a number in bold when (almost) identical to a given value.

    Args:
        x: Input number.

        value: Value to compare x with.

        num_decimals: Number of decimals to use for output format.

    Returns:
        String converted output.
    """
    # Consider values equal, when rounded results are equal
    # otherwise, it may look surprising in the table where they seem identical
    if round(x, num_decimals) >= round(value, num_decimals):
        return f"{{\\bfseries{x:.{num_decimals}f}}}"
    else:
        return f"{x:.{num_decimals}f}"
