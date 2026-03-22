"""
Core data structures for the rapidity package.

This module defines the two fundamental building blocks:

- Grid1D: a 1D quadrature grid with points, weights, and a label
- Field: a multi-dimensional array of values defined on a product of Grid1D grids

All other modules in the package build on these two classes.
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


# ---------------------------------------------------------------------------
# Field
# ---------------------------------------------------------------------------


@dataclass
class Field:
    values: np.ndarray
    grids: list[Grid1D]

    @classmethod
    def from_function(cls, f: callable, grids: list[Grid1D]) -> "Field":
        """Construct a Field by evaluating a function on a product grid.

        The function is evaluated at all combinations of grid points,
        producing a multi-dimensional array of values. The shape of the
        resulting array matches the sizes of the grids in the order they
        are provided.

        Parameters
        ----------
        f : callable
            A function to evaluate on the grid points. For a single grid,
            f should accept a 1D array. For multiple grids, f should accept
            one array per grid, as produced by np.meshgrid with indexing='ij'.
        grids : list[Grid1D]
            The grids defining the domain. The order determines the axis
            ordering of the resulting values array.

        Returns
        -------
        Field
            A Field with values obtained by evaluating f on the product grid.

        Examples
        --------
        >>> grid = Grid1D.gauss_legendre(-10, 10, 200, "theta")
        >>> field = Field.from_function(lambda theta: np.exp(-theta ** 2), [grid])

        >>> grid_x = Grid1D.uniform(0.0, 1.0, 10, "x")
        >>> grid_t = Grid1D.uniform(0.0, 1.0, 10, "t")
        >>> field = Field.from_function(lambda x, t: x + t, [grid_x, grid_t])
        """
        # for a single grid
        if len(grids) == 1:
            values = f(grids[0].points)
        # for multiple grids, meshgrid
        else:
            arrays = np.meshgrid(*[g.points for g in grids], indexing="ij")
            values = f(*arrays)
        return cls(values, grids)

    def _get_axis(self, dim: str | None = None) -> tuple[int, Grid1D]:
        """Find the axis index and grid corresponding to a dimension label.

        Parameters
        ----------
        dim : str, optional
            Dimension label to look up. If None and the field is 1D,
            returns the only axis. If None and the field is multi-dimensional,
            raises a ValueError.

        Returns
        -------
        tuple[int, Grid1D]
            The axis index and corresponding Grid1D object.

        Raises
        ------
        ValueError
            If dim is None and the field is multi-dimensional, or if the
            requested dimension label is not found.
        """
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

    def derivative(self, dim: str | None = None, order: int = 1) -> "Field":
        """Compute the derivative of the field along a dimension.

        Uses second-order accurate finite differences via numpy.gradient.

        Parameters
        ----------
        dim : str, optional
            Dimension to differentiate along. Can be omitted for 1D fields.
        order : int, optional
            Order of the derivative, either 1 or 2. Default is 1.

        Returns
        -------
        Field
            A new Field with the same grids containing the derivative values.

        Raises
        ------
        ValueError
            If order is not 1 or 2.
        """
        if order not in (1, 2):
            raise ValueError(f"order must be 1 or 2, got {order}")

        axis, grid = self._get_axis(dim)
        # edge_order=2 ensures second-order accuracy at boundaries, consistent with
        # the second-order accurate central differences used in the interior
        result = np.gradient(self.values, grid.points, axis=axis, edge_order=2)
        if order == 2:
            result = np.gradient(result, grid.points, axis=axis, edge_order=2)
        return Field(result, self.grids)

    def integrate(self, dim: str | None = None) -> "Field":
        """Integrate the field along a dimension using quadrature weights.

        Parameters
        ----------
        dim : str, optional
            Dimension to integrate over. Can be omitted for 1D fields.

        Returns
        -------
        Field
            A new Field with the specified dimension removed.

        Raises
        ------
        ValueError
            If dim is not specified for a multi-dimensional field, or if
            the requested dimension is not found.
        """
        axis, grid = self._get_axis(dim)
        new_values = self.values.swapaxes(axis, -1) @ grid.weights
        new_grids = [g for g in self.grids if g != grid]
        return Field(new_values, new_grids)

    def convolve(self, kernel: "Field", dim: str | None = None) -> "Field":
        """Convolve field with a general kernel along a dimension.

        Parameters
        ----------
        kernel : Field
            A 2D field representing the kernel K(theta, theta').
        dim : str, optional
            Dimension to convolve along.
        """
        axis, grid = self._get_axis(dim)
        # kernel.values is (n, n), contract with weights
        new_values = kernel.values @ (grid.weights * self.values)
        new_grids = [g for g in self.grids if g != grid]
        return Field(new_values, new_grids)

    def interpolate(self, new_grid: Grid1D) -> "Field":
        """Interpolate the field onto a new grid.

        Currently supported for 1D fields only.

        Parameters
        ----------
        new_grid : Grid1D
            The grid to interpolate onto.

        Returns
        -------
        Field
            A new Field with values interpolated onto new_grid.

        Raises
        ------
        NotImplementedError
            If the field is not 1D.
        """
        if len(self.grids) != 1:
            raise NotImplementedError(
                "interpolation is currently only supported for 1D fields"
            )

        from scipy.interpolate import interp1d

        f = interp1d(self.grids[0].points, self.values, kind="cubic")
        return Field(f(new_grid.points), [new_grid])
