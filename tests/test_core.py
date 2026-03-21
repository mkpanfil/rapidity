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
