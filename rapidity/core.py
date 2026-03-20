import numpy as np
from dataclasses import dataclass


@dataclass
class Grid1D:
    points: np.ndarray
    weights: np.ndarray
    label: str

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


@dataclass
class Field:
    values: np.ndarray
    grids: list[Grid1D]

    @classmethod
    def from_function(cls, f: callable, grids: list[Grid1D]) -> "Field":
        """Construct a field by evaluating a function on a product grid."""
        # for a single grid
        if len(grids) == 1:
            values = f(grids[0].points)
        # for multiple grids, meshgrid
        else:
            arrays = np.meshgrid(*[g.points for g in grids], indexing="ij")
            values = f(*arrays)
        return cls(values, grids)

    def _get_axis(self, dim: str | None = None) -> tuple[int, Grid1D]:
        if dim is None:
            if len(self.grids) != 1:
                raise ValueError("Must specify dim for multi-dimensional fields")
            return 0, self.grids[0]
        for i, g in enumerate(self.grids):
            if g.label == dim:
                return i, g
        raise ValueError(
            f"Dimension '{dim}' not found. Available dimensions: {[g.label for g in self.grids]}"
        )

    def integrate(self, dim: str | None = None) -> "Field":
        axis, grid = self._get_axis(dim)
        new_values = self.values.swapaxes(axis, -1) @ grid.weights
        new_grids = [g for g in self.grids if g != grid]
        return Field(new_values, new_grids)

    def convolve(self, kernel: "Field", dim: str | None = None) -> "Field":
        axis, grid = self._get_axis(dim)
        ...
