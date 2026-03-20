import numpy as np
from rapidity.core import Grid1D


def test_uniform_grid_is_uniform():
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
    a, b, n = 0.0, 1.0, 10
    grid = Grid1D.uniform(a, b, n, "x")

    # integral of f(x) = 1 over [0, 1] should be 1
    f = np.ones(n)
    assert np.isclose(f @ grid.weights, b - a)


def test_uniform_grid_integrates_linear():
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
