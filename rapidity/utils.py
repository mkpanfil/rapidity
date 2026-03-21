"""
Utility functions for the rapidity package.
"""

import numpy as np
from rapidity.core import Grid1D, Field


def make_kernel(f: callable, grid: Grid1D) -> Field:
    """Construct a 2D convolution kernel Field from a difference kernel function.

    Parameters
    ----------
    f : callable
        A function of one variable representing K(theta - theta').
    grid : Grid1D
        The quadrature grid on which the kernel is defined.

    Returns
    -------
    Field
        A 2D Field representing K(theta, theta') = f(theta - theta').

    Examples
    --------
    >>> grid = Grid1D.gauss_legendre(-10, 10, 200, "theta")
    >>> kernel = make_kernel(lambda x: 1 / (2 * np.pi) / (1 + x**2), grid)
    """
    return Field.from_function(lambda t1, t2: f(t1 - t2), [grid, grid])


def sine_kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The sine kernel.

    .. math::

        K(x, y) = \\frac{\\sin(\\pi(x-y))}{\\pi(x-y)}

    The diagonal singularity at x = y is handled analytically via the
    limiting value K(x, x) = 1.

    Parameters
    ----------
    x : np.ndarray
        First argument.
    y : np.ndarray
        Second argument.

    Returns
    -------
    np.ndarray
        The sine kernel evaluated at (x, y).
    """
    # np.sinc is defined as sin(π*x)/(π*x), so the π factors are included
    # and the singularity at x = y is handled internally
    return np.sinc(x - y)
