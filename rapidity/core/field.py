"""
Multi-dimensional fields on product grids.

This module defines :class:`Field`, which represents a physical quantity
defined on a product of :class:`~rapidity.core.grid.Grid1D` grids. A field
always carries its grids alongside its values, ensuring that the
discretization structure is never lost.

Key operations provided:

- Construction via :meth:`Field.from_function` or :meth:`Field.load`
- Calculus: :meth:`Field.integrate`, :meth:`Field.convolve`,
  :meth:`Field.derivative`, :meth:`Field.interpolate`
- Arithmetic: standard operators ``+``, ``-``, ``*``, ``/``, ``**``
  between fields and scalars, with broadcasting along matching dimensions
- I/O: :meth:`Field.save` and :meth:`Field.load` using HDF5
"""

import numpy as np
import h5py
from dataclasses import dataclass
from rapidity.core.grid import Grid1D

_SCALAR_TYPES = (int, float, complex, np.floating, np.integer, np.complexfloating)

# ---------------------------------------------------------------------------
# Field
# ---------------------------------------------------------------------------


@dataclass
class Field:
    values: np.ndarray
    grids: list[Grid1D]

    # -----------------------------------------------------------------------
    # Arithmetic operations
    # -----------------------------------------------------------------------

    def __add__(self, other: "Field | complex") -> "Field":
        """Add two fields or a field and a scalar.

        Two fields are added pointwise. If they have different dimensions,
        the smaller field is broadcast along the missing dimensions, provided
        the shared dimensions have identical grids.

        Parameters
        ----------
        other : Field or scalar
            The field or scalar to add. Scalars must be int, float, or complex.

        Returns
        -------
        Field
            The sum, defined on the union of both fields' grids.

        Raises
        ------
        ValueError
            If both fields share a dimension but with incompatible grids.
        TypeError
            If other is not a Field or a supported scalar type.
        """
        if isinstance(other, Field):
            a, b, grids = _align(self, other)
            return Field(a + b, grids)
        if not isinstance(other, _SCALAR_TYPES):
            raise TypeError(
                f"unsupported operand type for +: 'Field' and '{type(other).__name__}'"
            )
        return Field(self.values + other, self.grids)

    def __sub__(self, other: "Field | complex") -> "Field":
        """Subtract two fields or a scalar from a field."""
        if isinstance(other, Field):
            a, b, grids = _align(self, other)
            return Field(a - b, grids)
        if not isinstance(other, _SCALAR_TYPES):
            raise TypeError(
                f"unsupported operand type for -: 'Field' and '{type(other).__name__}'"
            )
        return Field(self.values - other, self.grids)

    def __mul__(self, other: "Field | complex") -> "Field":
        """Multiply two fields or a field and a scalar."""
        if isinstance(other, Field):
            a, b, grids = _align(self, other)
            return Field(a * b, grids)
        if not isinstance(other, _SCALAR_TYPES):
            raise TypeError(
                f"unsupported operand type for *: 'Field' and '{type(other).__name__}'"
            )
        return Field(self.values * other, self.grids)

    def __truediv__(self, other: "Field | complex") -> "Field":
        """Divide two fields or a field by a scalar."""
        if isinstance(other, Field):
            a, b, grids = _align(self, other)
            return Field(a / b, grids)
        if not isinstance(other, _SCALAR_TYPES):
            raise TypeError(
                f"unsupported operand type for /: 'Field' and '{type(other).__name__}'"
            )
        return Field(self.values / other, self.grids)

    def __neg__(self) -> "Field":
        """Negate a field."""
        return Field(-self.values, self.grids)

    def __rmul__(self, other: complex) -> "Field":
        """Right multiply a field by a scalar."""
        if not isinstance(other, _SCALAR_TYPES):
            raise TypeError(
                f"unsupported operand type for *: '{type(other).__name__}' and 'Field'"
            )
        return Field(other * self.values, self.grids)

    def __radd__(self, other: complex) -> "Field":
        """Right add a scalar to a field."""
        if not isinstance(other, _SCALAR_TYPES):
            raise TypeError(
                f"unsupported operand type for +: '{type(other).__name__}' and 'Field'"
            )
        return Field(other + self.values, self.grids)

    def __rsub__(self, other: complex) -> "Field":
        """Right subtract a field from a scalar."""
        if not isinstance(other, _SCALAR_TYPES):
            raise TypeError(
                f"unsupported operand type for -: '{type(other).__name__}' and 'Field'"
            )
        return Field(other - self.values, self.grids)

    def __rtruediv__(self, other: complex) -> "Field":
        """Right divide a scalar by a field."""
        if not isinstance(other, _SCALAR_TYPES):
            raise TypeError(
                f"unsupported operand type for /: '{type(other).__name__}' and 'Field'"
            )
        return Field(other / self.values, self.grids)

    def __abs__(self) -> "Field":
        """Absolute value of a field."""
        return Field(np.abs(self.values), self.grids)

    def __pow__(self, other: complex) -> "Field":
        """Raise a field to a power."""
        if not isinstance(other, _SCALAR_TYPES):
            raise TypeError(
                f"unsupported operand type for **: 'Field' and '{type(other).__name__}'"
            )
        return Field(self.values**other, self.grids)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # Constructors
    # -----------------------------------------------------------------------

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
            A function to evaluate on the grid points. Must be vectorized,
            i.e. able to accept numpy arrays as arguments and return a
            numpy array. For a single grid, f should accept a 1D array.
            For multiple grids, f should accept one array per grid,
            as produced by np.meshgrid with indexing='ij'.
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

    @classmethod
    def from_function(cls, f: callable, grids: list["Grid1D"]) -> "Field":
        if len(grids) == 1:
            values = np.asarray(f(grids[0].points))
            # broadcast_to handles the case where f returns a scalar or
            # an array of wrong shape, e.g. lambda t: 1.0
            # copy() makes the result writeable since broadcast_to returns
            # a read-only view
            values = np.broadcast_to(values, grids[0].points.shape).copy()
        else:
            arrays = np.meshgrid(*[g.points for g in grids], indexing="ij")
            expected_shape = tuple(g.points.size for g in grids)
            values = np.broadcast_to(np.asarray(f(*arrays)), expected_shape).copy()
        return cls(values, grids)

    @classmethod
    def from_values(cls, values: np.ndarray, grids: list["Grid1D"]) -> "Field":
        """Construct a Field from an array of values and a list of grids.

        The shape of values must be consistent with the grids — axis i
        must have the same size as grids[i].

        Parameters
        ----------
        values : np.ndarray
            Array of values. Shape must match the sizes of the grids.
        grids : list[Grid1D]
            Grids defining the domain. The order determines the axis
            ordering of the values array.

        Returns
        -------
        Field
            A Field with the given values on the given grids.

        Raises
        ------
        ValueError
            If the shape of values is inconsistent with the grids.

        Examples
        --------
        >>> grid = Grid1D.gauss_legendre(-5.0, 5.0, 50, "theta")
        >>> values = np.exp(-grid.points**2)
        >>> field = Field.from_values(values, [grid])
        """
        expected_shape = tuple(g.points.size for g in grids)
        if values.shape != expected_shape:
            raise ValueError(
                f"Shape of values {values.shape} is inconsistent with "
                f"grids {expected_shape}"
            )
        return cls(values, grids)

    @classmethod
    def load(cls, path: str) -> "Field":
        """Load a field from an HDF5 file.

        Parameters
        ----------
        path : str
            Path to the HDF5 file to load.

        Returns
        -------
        Field
            The field stored in the file.
        """
        with h5py.File(path, "r") as f:
            values = f["values"][:]
            grids = []
            for i in range(len(f["grids"])):
                g = f["grids"][str(i)]
                grids.append(
                    Grid1D(
                        points=g["points"][:],
                        weights=g["weights"][:],
                        label=g.attrs["label"],
                    )
                )
        return cls(values, grids)

    # -----------------------------------------------------------------------
    # Calculus
    # -----------------------------------------------------------------------

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

        Returns
        -------
        Field
            A new Field with the same grids as self, with convolution
            performed along the specified dimension.

        Raises
        ------
        ValueError
            If kernel is not a 2D field defined on the same grid in both
            dimensions.
        """
        axis, grid = self._get_axis(dim)

        if len(kernel.grids) != 2:
            raise ValueError("kernel must be a 2D Field")
        if kernel.grids[0] != grid or kernel.grids[1] != grid:
            raise ValueError(
                "kernel must be defined on the same grid for both dimensions"
            )

        # Move convolution axis to last position for vectorized contraction
        data = np.moveaxis(self.values, axis, -1)
        weighted = data * grid.weights

        # K[i,j] * f[...,j] summed over j => result[...,i]
        conv = np.tensordot(kernel.values, weighted, axes=([1], [-1]))
        conv = np.moveaxis(conv, 0, -1)

        # Restore original axis ordering
        new_values = np.moveaxis(conv, -1, axis)

        return Field(new_values, self.grids)

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

    # -----------------------------------------------------------------------
    # I/O
    # -----------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the field to an HDF5 file.

        Parameters
        ----------
        path : str
            Path to the HDF5 file to create.
        """
        with h5py.File(path, "w") as f:
            f.create_dataset("values", data=self.values)
            grids_group = f.create_group("grids")
            for i, grid in enumerate(self.grids):
                g = grids_group.create_group(str(i))
                g.create_dataset("points", data=grid.points)
                g.create_dataset("weights", data=grid.weights)
                g.attrs["label"] = grid.label


def _align(
    field1: "Field", field2: "Field"
) -> tuple[np.ndarray, np.ndarray, list[Grid1D]]:
    """Align two fields for broadcasting along matching dimensions."""
    # check shared grids are compatible
    field1_labels = [g.label for g in field1.grids]
    field2_labels = [g.label for g in field2.grids]

    for label in field1_labels:
        if label in field2_labels:
            field1_grid = field1.grids[field1_labels.index(label)]
            field2_grid = field2.grids[field2_labels.index(label)]
            if field1_grid != field2_grid:
                raise ValueError(
                    f"Dimension '{label}' is present in both fields "
                    f"but grids are incompatible"
                )

    # build output grids — union of both grids in a consistent order
    all_labels = field1_labels + [
        label for label in field2_labels if label not in field1_labels
    ]
    all_grids = []
    for label in all_labels:
        if label in field1_labels:
            all_grids.append(field1.grids[field1_labels.index(label)])
        else:
            all_grids.append(field2.grids[field2_labels.index(label)])

    # reshape self.values and other.values for broadcasting
    field1_shape = [
        field1.grids[field1_labels.index(label)].points.size
        if label in field1_labels
        else 1
        for label in all_labels
    ]
    field2_shape = [
        field2.grids[field2_labels.index(label)].points.size
        if label in field2_labels
        else 1
        for label in all_labels
    ]

    return (
        field1.values.reshape(field1_shape),
        field2.values.reshape(field2_shape),
        all_grids,
    )
