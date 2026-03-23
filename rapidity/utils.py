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


def sine_kernel(grid: Grid1D) -> Field:
    """The sine kernel as a Field on a product grid.

    .. math::

        K(x, y) = \\frac{\\sin(\\pi(x-y))}{\\pi(x-y)}

    The diagonal singularity at x = y is handled analytically via
    numpy.sinc, which is defined as sin(πx)/(πx) with sinc(0) = 1.

    Parameters
    ----------
    grid : Grid1D
        The quadrature grid on which the kernel is defined.

    Returns
    -------
    Field
        A 2D Field representing the sine kernel on the product grid.

    Examples
    --------
    >>> grid = Grid1D.gauss_legendre(0.0, 1.0, 50, "x")
    >>> kernel = sine_kernel(grid)
    >>> det = fredholm_det(kernel)
    """
    # np.sinc is defined as sin(π*x)/(π*x), so the π factors are included
    # and the singularity at x = y is handled internally
    return Field.from_function(lambda x, y: np.sinc(x - y), [grid, grid])
