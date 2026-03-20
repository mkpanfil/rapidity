import numpy as np
from rapidity.core import Grid1D, Field


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
