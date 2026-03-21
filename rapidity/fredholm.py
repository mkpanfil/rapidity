"""
Numerical evaluation of Fredholm determinants.

This module provides tools for computing Fredholm determinants of the form

.. math::

    \\det(I - wK)

where :math:`K` is an integral operator with kernel :math:`K(x, x')` and
:math:`w(x)` is an optional weight function.

The main entry point is :func:`fredholm_det`, which accepts a kernel and
an optional weight function as :class:`~rapidity.core.Field` objects defined
on a :class:`~rapidity.core.Grid1D` quadrature grid.

References
----------
Bornemann, F. (2010). On the numerical evaluation of Fredholm determinants.
*Mathematics of Computation*, 79(270), 871-915.
"""

import numpy as np
from rapidity.core import Field


def fredholm_det(kernel: Field, weight: Field | None = None) -> float:
    """Compute the Fredholm determinant det(I - wK) using the Nyström method.

    The Fredholm determinant is approximated as a finite matrix determinant:

    .. math::

        \\det(I - wK) \\approx \\det(\\delta_{ij} - w(x_i) \\sqrt{w_i} K(x_i, x_j) \\sqrt{w_j})

    where :math:`w_i` are the quadrature weights and :math:`w(x)` is the
    optional weight function.

    Parameters
    ----------
    kernel : Field
        A 2D field representing the kernel K(x, x'). Must have exactly
        two identical grids.
    weight : Field, optional
        A 1D field representing the weight function w(x). If None,
        w(x) = 1 is assumed, reducing to det(I - K).

    Returns
    -------
    float
        The Fredholm determinant det(I - wK).

    Raises
    ------
    ValueError
        If the kernel is not a 2D field, if its two grids are not identical,
        or if the weight is not a 1D field on the same grid as the kernel.

    Examples
    --------
    >>> grid = Grid1D.gauss_legendre(0.0, 1.0, 50, "x")
    >>> kernel = Field.from_function(lambda x, y: np.sin(x - y), [grid, grid])
    >>> fredholm_det(kernel)
    """
    if len(kernel.grids) != 2:
        raise ValueError(f"kernel must be a 2D field, got {len(kernel.grids)}D")
    if kernel.grids[0] != kernel.grids[1]:
        raise ValueError("both grids of the kernel must be the same")

    if weight is not None:
        if len(weight.grids) != 1:
            raise ValueError(f"weight must be a 1D field, got {len(weight.grids)}D")
        if weight.grids[0] != kernel.grids[0]:
            raise ValueError("weight grid must match the kernel grid")

    grid = kernel.grids[0]
    sqrt_w = np.sqrt(grid.weights)
    K = sqrt_w[:, None] * kernel.values * sqrt_w[None, :]

    if weight is not None:
        K = weight.values[:, None] * K

    return np.linalg.det(np.eye(len(grid.points)) - K)
