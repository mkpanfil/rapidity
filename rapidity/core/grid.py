"""
One-dimensional quadrature grids.

This module defines :class:`Grid1D`, the fundamental discretization object
in `rapidity`. A grid stores quadrature points, weights, and a dimension
label. It is a pure data container with no knowledge of physical quantities.

Alternative constructors are provided for common quadrature rules:

- :meth:`Grid1D.gauss_legendre` — for integrals on finite intervals
- :meth:`Grid1D.uniform` — trapezoid rule on a uniform grid
- :meth:`Grid1D.gauss_hermite` — for rapidly decaying functions on the
  full real line
"""

import numpy as np
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Grid1D
# ---------------------------------------------------------------------------


@dataclass
class Grid1D:
    points: np.ndarray
    weights: np.ndarray
    label: str

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Grid1D):
            return NotImplemented
        return (
            self.label == other.label
            and np.array_equal(self.points, other.points)
            and np.array_equal(self.weights, other.weights)
        )

    def __hash__(self):
        return hash((self.label, self.points.tobytes(), self.weights.tobytes()))

    @classmethod
    def uniform(cls, a: float, b: float, n: int, label: str) -> "Grid1D":
        """Constructs a uniform quadrature grid on[a, b] of n points

        Parameters
        ----------
        a, b : float
            The grid boundaries.
        n : int
            Number of quadrature points.
        label : str
            Dimension label

        Returns
        -------
        Grid1D
            A uniform quadrature grid.
        """

        points = np.linspace(a, b, n)
        weights = np.full(n, (b - a) / (n - 1))
        weights[0] /= 2
        weights[-1] /= 2
        return cls(points, weights, label)

    @classmethod
    def gauss_legendre(cls, a: float, b: float, n: int, label: str) -> "Grid1D":
        """Constructs a Gauss-Legendre quadrature grid on [a, b] of n points

        Parameters
        ----------
        a, b : float
            The grid boundaries.
        n : int
            Number of quadrature points.
        label : str
            Dimension label

        Returns
        -------
        Grid1D
            A Gauss-Legendre quadrature grid.
        """
        points, weights = np.polynomial.legendre.leggauss(n)
        points = 0.5 * (b - a) * points + 0.5 * (b + a)
        weights = 0.5 * (b - a) * weights
        return cls(points, weights, label)

    @classmethod
    def gauss_hermite(cls, n: int, label: str) -> "Grid1D":
        """Construct a Gauss-Hermite quadrature grid on (-inf, inf).

        Approximates integrals of rapidly decaying functions f(x) on the
        full real line, without assuming any explicit Gaussian factor in
        the integrand.

        Parameters
        ----------
        n : int
            Number of quadrature points.
        label : str
            Dimension label.

        Returns
        -------
        Grid1D
            A Gauss-Hermite quadrature grid.
        """
        points, weights = np.polynomial.hermite.hermgauss(n)
        weights = weights * np.exp(points**2)
        return cls(points, weights, label)
