"""
Utility functions for the rapidity package.
"""

import numpy as np
import matplotlib.pyplot as plt
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


def plot(
    field: Field,
    ax: plt.Axes = None,
    value: str = "real",
    **kwargs,
):
    """Plot a 1D or 2D Field using matplotlib.

    Parameters
    ----------
    field : Field
        The Field to plot.
    ax : plt.Axes, optional
        The axes on which to plot. If None, the current axes are used.
    value : str, optional
        The type of the field to plot. Can be "real", "imag", or "abs".
    **kwargs
        Additional keyword arguments passed to plt.plot (for 1D) or plt.imshow (for 2D).

    Raises
    ------
    ValueError
        If the Field is not 1D or 2D.
        If the value argument is not one of "real", "imag", or "abs".

    Examples
    --------
    >>> grid = Grid1D.uniform(-10, 10, 200, "theta")
    >>> f = Field.from_function(lambda t: np.exp(-t**2), [grid])
    >>> plot(f)
    >>> plt.show()
    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    if value == "real":
        z = field.values.real
    elif value == "imag":
        z = field.values.imag
    elif value == "abs":
        z = np.abs(field.values)
    else:
        raise ValueError("value must be 'real', 'imag', or 'abs'")

    if len(field.grids) == 1:
        x = field.grids[0].points

        ax.plot(x, z, **kwargs)
        ax.set_xlabel(field.grids[0].label)
        ax.set_ylabel("Field value")
        ax.set_title("1D Field Plot")
        ax.grid()
    elif len(field.grids) == 2:
        x = field.grids[0].points
        y = field.grids[1].points
        X, Y = np.meshgrid(x, y, indexing="ij")
        ax.imshow(
            z,
            extent=(y[0], y[-1], x[0], x[-1]),
            origin="lower",
            aspect="auto",
            **kwargs,
        )

        ax.set_xlabel(field.grids[1].label)
        ax.set_ylabel(field.grids[0].label)
        ax.set_title("2D Field Plot")
        # ax.setcolorbar(label="Field value")
    else:
        raise ValueError("Only 1D and 2D Fields can be plotted.")
