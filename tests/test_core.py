"""
Tests for rapidity.core — Grid1D and Field.

Covers:
- Grid1D constructors: uniform, gauss_legendre, gauss_hermite
- Field.from_function: 1D and 2D cases
- Field.integrate: correctness and error handling
- Field.convolve: correctness
- Field.interpolate: recovery, finer grid, error handling
"""

import numpy as np
import pytest
import tempfile
import os
from rapidity.core import Grid1D, Field


# ---------------------------------------------------------------------------
# Grid1D
# ---------------------------------------------------------------------------


def test_uniform_grid_is_uniform():
    """Checks if the grid is uniform and with equal weights up to the boundaries"""
    a, b, n = 0.0, 1.0, 10
    grid = Grid1D.uniform(a, b, n, "x")

    # points are evenly spaced
    diffs = np.diff(grid.points)
    assert np.allclose(diffs, diffs[0])

    # bulk weights are all equal
    assert np.allclose(grid.weights[1:-1], grid.weights[1])

    # boundary weights are half of the bulk weights
    assert np.allclose(grid.weights[0], grid.weights[1] / 2)
    assert np.allclose(grid.weights[-1], grid.weights[1] / 2)


def test_uniform_grid_integrates_constant():
    """Constant function integrates to sum of the grid weights"""
    a, b, n = 0.0, 1.0, 10
    grid = Grid1D.uniform(a, b, n, "x")

    # integral of f(x) = 1 over [0, 1] should be 1
    f = np.ones(n)
    assert np.isclose(f @ grid.weights, b - a)


def test_uniform_grid_integrates_linear_exactly():
    """The uniform grid integrates a linear function exactly"""
    a, b, n = 0.0, 1.0, 10
    grid = Grid1D.uniform(a, b, n, "x")
    f = grid.points  # f(x) = x, integral over [0,1] = 0.5
    assert np.isclose(f @ grid.weights, 0.5)


def test_gauss_legendre_integrates_polynomials_exactly():
    """Gauss-Legendre with n points integrates polynomials of degree 2n-1 exactly."""
    a, b, n = 0.0, 1.0, 10
    grid = Grid1D.gauss_legendre(a, b, n, "theta")

    # f(x) = x^(2n-1), integral over [a, b] = 1/(2n) * (b^(2n) - a^(2n))
    f = grid.points ** (2 * n - 1)
    expected = (b ** (2 * n) - a ** (2 * n)) / (2 * n)
    assert np.isclose(f @ grid.weights, expected)


def test_gauss_hermite_integrates_gaussian_exactly():
    """Gauss-Hermite integrates a plain Gaussian on the real line exactly."""
    n = 10
    grid = Grid1D.gauss_hermite(n, "theta")

    # integral of exp(-x^2) over (-inf, inf) = sqrt(pi)
    f = np.exp(-(grid.points**2))
    assert np.isclose(f @ grid.weights, np.sqrt(np.pi))


# ---------------------------------------------------------------------------
# Field
# ---------------------------------------------------------------------------


def test_field_from_function_1d():
    """Field.from_function evaluates a 1D function correctly on a single grid."""
    grid = Grid1D.gauss_legendre(-10, 10, 200, "theta")
    field = Field.from_function(lambda theta: np.exp(-(theta**2)), [grid])

    assert field.values.shape == (200,)
    assert np.allclose(field.values, np.exp(-(grid.points**2)))


def test_field_from_function_2d():
    """Field.from_function evaluates a 2D function correctly on a product grid."""
    grid_x = Grid1D.uniform(0.0, 1.0, 10, "x")
    grid_t = Grid1D.uniform(0.0, 1.0, 10, "t")
    field = Field.from_function(lambda x, t: x + t, [grid_x, grid_t])

    assert field.values.shape == (10, 10)
    assert np.allclose(field.values[0, :], grid_t.points)  # x=0: f = t
    assert np.allclose(field.values[:, 0], grid_x.points)  # t=0: f = x


def test_field_from_function_constant():
    """Field.from_function with a constant function returns an array not a scalar."""
    grid = Grid1D.gauss_legendre(-5.0, 5.0, 50, "theta")
    field = Field.from_function(lambda t: 1.0, [grid])
    assert isinstance(field.values, np.ndarray)
    assert field.values.shape == (50,)
    assert np.allclose(field.values, 1.0)


def test_field_from_values():
    """Field.from_values constructs a field with the given values."""
    grid = Grid1D.gauss_legendre(-5.0, 5.0, 50, "theta")
    values = np.exp(-(grid.points**2))
    field = Field.from_values(values, [grid])
    assert np.allclose(field.values, values)


def test_field_from_values_raises_for_wrong_shape():
    """Field.from_values raises ValueError for inconsistent shape."""
    grid = Grid1D.gauss_legendre(-5.0, 5.0, 50, "theta")
    values = np.ones(30)  # wrong size
    with pytest.raises(ValueError):
        Field.from_values(values, [grid])


def test_integrate_raises_for_missing_dim():
    """integrate raises ValueError when the requested dimension does not exist."""
    grid = Grid1D.gauss_legendre(-10, 10, 50, "theta")
    field = Field.from_function(lambda theta: np.exp(-(theta**2)), [grid])

    with pytest.raises(ValueError):
        field.integrate("x")


def test_integrate_raises_without_dim_for_multidimensional_field():
    """integrate raises ValueError when dim is not specified for a multi-dimensional field."""
    grid_x = Grid1D.uniform(0.0, 1.0, 10, "x")
    grid_t = Grid1D.uniform(0.0, 1.0, 10, "t")
    field = Field.from_function(lambda x, t: x + t, [grid_x, grid_t])

    with pytest.raises(ValueError):
        field.integrate()


def test_convolve_gaussian_with_gaussian():
    """Convolution of two Gaussians gives a Gaussian with known width."""
    grid = Grid1D.gauss_hermite(100, "theta")

    f = Field.from_function(lambda theta: np.exp(-(theta**2)), [grid])
    kernel = Field.from_function(lambda t1, t2: np.exp(-((t1 - t2) ** 2)), [grid, grid])

    result = f.convolve(kernel)
    expected = np.sqrt(np.pi / 2) * np.exp(-(grid.points**2) / 2)
    assert np.allclose(result.values, expected, atol=1e-6)


def test_convolve_along_axis_2d():
    """Convolution along a specified axis in a 2D field retains other dimensions."""
    grid_x = Grid1D.uniform(0.0, 1.0, 4, "x")
    grid_t = Grid1D.uniform(0.0, 1.0, 5, "t")

    f = Field.from_function(lambda x, t: x + t, [grid_x, grid_t])
    kernel = Field.from_function(
        lambda t1, t2: np.exp(-((t1 - t2) ** 2)), [grid_t, grid_t]
    )

    result = f.convolve(kernel, dim="t")

    expected = np.zeros_like(f.values)
    for ix in range(len(grid_x.points)):
        for it in range(len(grid_t.points)):
            expected[ix, it] = sum(
                kernel.values[it, jt] * f.values[ix, jt] * grid_t.weights[jt]
                for jt in range(len(grid_t.points))
            )

    assert np.allclose(result.values, expected, atol=1e-12)
    assert result.values.shape == f.values.shape
    assert [g.label for g in result.grids] == ["x", "t"]


def test_interpolate_exact_recovery():
    """Interpolating onto the same grid returns the same values."""
    grid = Grid1D.gauss_legendre(-5.0, 5.0, 50, "x")
    field = Field.from_function(lambda x: np.exp(-(x**2)), [grid])
    result = field.interpolate(grid)
    assert np.allclose(result.values, field.values)


def test_interpolate_smooth_function():
    """Interpolation of a smooth function onto a finer grid is accurate."""
    coarse_grid = Grid1D.uniform(-2.0, 2.0, 20, "x")
    fine_grid = Grid1D.uniform(-2.0, 2.0, 100, "x")
    field = Field.from_function(lambda x: np.exp(-(x**2)), [coarse_grid])
    result = field.interpolate(fine_grid)
    expected = np.exp(-(fine_grid.points**2))
    assert np.allclose(result.values, expected, atol=1e-4)


def test_interpolate_raises_for_multidimensional():
    """interpolate raises NotImplementedError for multi-dimensional fields."""
    grid_x = Grid1D.uniform(0.0, 1.0, 10, "x")
    grid_t = Grid1D.uniform(0.0, 1.0, 10, "t")
    field = Field.from_function(lambda x, t: x + t, [grid_x, grid_t])
    new_grid = Grid1D.uniform(0.0, 1.0, 20, "x")
    with pytest.raises(NotImplementedError):
        field.interpolate(new_grid)


def test_derivative_first_order_quadratic():
    """First derivative of a quadratic function is exact."""
    grid = Grid1D.uniform(-5.0, 5.0, 100, "x")
    field = Field.from_function(lambda x: x**2, [grid])
    result = field.derivative()
    expected = 2 * grid.points
    assert np.allclose(result.values, expected, atol=1e-10)


def test_derivative_second_order_quadratic():
    """Second derivative of a quadratic function is exact."""
    grid = Grid1D.uniform(-5.0, 5.0, 100, "x")
    field = Field.from_function(lambda x: x**2, [grid])
    result = field.derivative(order=2)
    expected = np.full_like(grid.points, 2.0)
    assert np.allclose(result.values, expected, atol=1e-10)


def test_derivative_along_axis():
    """Derivative along a specific axis of a 2D field is correct."""
    grid_x = Grid1D.uniform(-5.0, 5.0, 100, "x")
    grid_t = Grid1D.uniform(0.0, 1.0, 50, "t")
    field = Field.from_function(lambda x, t: x**2 + t, [grid_x, grid_t])
    result = field.derivative("x")
    # df/dx = 2x, independent of t
    expected = 2 * grid_x.points[:, None] * np.ones((100, 50))
    assert np.allclose(result.values, expected, atol=1e-10)


def test_field_save_load_1d():
    """Saving and loading a 1D field recovers the original field exactly."""
    grid = Grid1D.gauss_legendre(-5.0, 5.0, 50, "theta")
    field = Field.from_function(lambda x: np.exp(-(x**2)), [grid])

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        path = f.name
    try:
        field.save(path)
        loaded = Field.load(path)
        assert np.allclose(loaded.values, field.values)
        assert np.allclose(loaded.grids[0].points, grid.points)
        assert np.allclose(loaded.grids[0].weights, grid.weights)
        assert loaded.grids[0].label == grid.label
    finally:
        os.remove(path)


def test_field_save_load_2d():
    """Saving and loading a 2D field recovers the original field exactly."""
    grid_x = Grid1D.uniform(0.0, 1.0, 20, "x")
    grid_t = Grid1D.uniform(0.0, 1.0, 20, "t")
    field = Field.from_function(lambda x, t: x**2 + t, [grid_x, grid_t])

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        path = f.name
    try:
        field.save(path)
        loaded = Field.load(path)
        assert np.allclose(loaded.values, field.values)
        assert len(loaded.grids) == 2
        assert loaded.grids[0].label == "x"
        assert loaded.grids[1].label == "t"
    finally:
        os.remove(path)


def test_add_fields_same_grid():
    """Adding two fields on the same grid gives correct values."""
    grid = Grid1D.gauss_legendre(-5.0, 5.0, 50, "x")
    f = Field.from_function(lambda x: x**2, [grid])
    g = Field.from_function(lambda x: x**3, [grid])
    result = f + g
    assert np.allclose(result.values, grid.points**2 + grid.points**3)
    assert result.grids == [grid]


def test_add_fields_broadcasting():
    """Adding a 1D and 2D field broadcasts correctly."""
    grid_x = Grid1D.uniform(0.0, 1.0, 10, "x")
    grid_y = Grid1D.uniform(0.0, 1.0, 10, "y")
    f = Field.from_function(lambda x: x**2, [grid_x])
    g = Field.from_function(lambda x, y: x + y, [grid_x, grid_y])
    result = f + g
    assert result.values.shape == (10, 10)
    assert result.grids == [grid_x, grid_y]
    expected = (
        grid_x.points[:, None] ** 2 + grid_x.points[:, None] + grid_y.points[None, :]
    )
    assert np.allclose(result.values, expected)


def test_add_field_scalar():
    """Adding a scalar to a field gives correct values."""
    grid = Grid1D.gauss_legendre(-5.0, 5.0, 50, "x")
    f = Field.from_function(lambda x: x**2, [grid])
    result = f + 2.0
    assert np.allclose(result.values, grid.points**2 + 2.0)


def test_add_field_incompatible_grids_raises():
    """Adding two fields with incompatible grids raises ValueError."""
    grid1 = Grid1D.gauss_legendre(-5.0, 5.0, 50, "x")
    grid2 = Grid1D.gauss_legendre(-3.0, 3.0, 50, "x")
    f = Field.from_function(lambda x: x**2, [grid1])
    g = Field.from_function(lambda x: x**3, [grid2])
    with pytest.raises(ValueError):
        f + g


def test_add_field_invalid_type_raises():
    """Adding a non-scalar to a field raises TypeError."""
    grid = Grid1D.gauss_legendre(-5.0, 5.0, 50, "x")
    f = Field.from_function(lambda x: x**2, [grid])
    with pytest.raises(TypeError):
        f + "string"


def test_sub_fields():
    """Subtracting two fields gives correct values."""
    grid = Grid1D.gauss_legendre(-5.0, 5.0, 50, "x")
    f = Field.from_function(lambda x: x**2, [grid])
    g = Field.from_function(lambda x: x**3, [grid])
    result = f - g
    assert np.allclose(result.values, grid.points**2 - grid.points**3)


def test_mul_fields():
    """Multiplying two fields gives correct values."""
    grid = Grid1D.gauss_legendre(-5.0, 5.0, 50, "x")
    f = Field.from_function(lambda x: x**2, [grid])
    g = Field.from_function(lambda x: x**3, [grid])
    result = f * g
    assert np.allclose(result.values, grid.points**5)


def test_div_fields():
    """Dividing two fields gives correct values."""
    grid = Grid1D.gauss_legendre(-5.0, 5.0, 50, "x")
    f = Field.from_function(lambda x: x**2, [grid])
    g = Field.from_function(lambda x: x + 10.0, [grid])
    result = f / g
    assert np.allclose(result.values, grid.points**2 / (grid.points + 10.0))


def test_neg_field():
    """Negating a field gives correct values."""
    grid = Grid1D.gauss_legendre(-5.0, 5.0, 50, "x")
    f = Field.from_function(lambda x: x**2, [grid])
    result = -f
    assert np.allclose(result.values, -(grid.points**2))


def test_abs_field():
    """Absolute value of a field gives correct values."""
    grid = Grid1D.gauss_legendre(-5.0, 5.0, 50, "x")
    f = Field.from_function(lambda x: x, [grid])
    result = abs(f)
    assert np.allclose(result.values, np.abs(grid.points))


def test_pow_field():
    """Raising a field to a power gives correct values."""
    grid = Grid1D.gauss_legendre(0.0, 5.0, 50, "x")
    f = Field.from_function(lambda x: x, [grid])
    result = f**2
    assert np.allclose(result.values, grid.points**2)


def test_rmul_field():
    """Right multiplying a field by a scalar gives correct values."""
    grid = Grid1D.gauss_legendre(-5.0, 5.0, 50, "x")
    f = Field.from_function(lambda x: x**2, [grid])
    result = 2.0 * f
    assert np.allclose(result.values, 2.0 * grid.points**2)


def test_radd_field():
    """Right adding a scalar to a field gives correct values."""
    grid = Grid1D.gauss_legendre(-5.0, 5.0, 50, "x")
    f = Field.from_function(lambda x: x**2, [grid])
    result = 2.0 + f
    assert np.allclose(result.values, 2.0 + grid.points**2)


def test_rsub_field():
    """Right subtracting a field from a scalar gives correct values."""
    grid = Grid1D.gauss_legendre(-5.0, 5.0, 50, "x")
    f = Field.from_function(lambda x: x**2, [grid])
    result = 2.0 - f
    assert np.allclose(result.values, 2.0 - grid.points**2)


def test_rtruediv_field():
    """Right dividing a scalar by a field gives correct values."""
    grid = Grid1D.gauss_legendre(1.0, 5.0, 50, "x")
    f = Field.from_function(lambda x: x, [grid])
    result = 1.0 / f
    assert np.allclose(result.values, 1.0 / grid.points)


def test_mul_complex_scalar():
    """Multiplying a field by a complex scalar gives correct values."""
    grid = Grid1D.gauss_legendre(-5.0, 5.0, 50, "x")
    f = Field.from_function(lambda x: x**2, [grid])
    result = f * (1 + 2j)
    assert np.allclose(result.values, grid.points**2 * (1 + 2j))
