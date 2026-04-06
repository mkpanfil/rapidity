import pytest
import numpy as np
from rapidity.core import Grid1D, Field
from rapidity.models import Model, LiebLiniger, StringModel, XXXSpinChain


# ---------------------------------------------------------------------------
# Model protocol
# ---------------------------------------------------------------------------


def test_lieb_liniger_satisfies_model_protocol():
    """LiebLiniger satisfies the Model protocol."""
    model = LiebLiniger(c=1.0)
    assert isinstance(model, Model)


def test_xxx_satisfies_string_model_protocol():
    """XXXSpinChain satisfies the StringModel protocol."""
    model = XXXSpinChain(S=0.5, n_max=5)
    assert isinstance(model, StringModel)


# ---------------------------------------------------------------------------
# Lieb-Liniger
# ---------------------------------------------------------------------------


def test_lieb_liniger_raises_for_nonpositive_c():
    """LiebLiniger raises ValueError for non-positive coupling constant."""
    with pytest.raises(ValueError):
        LiebLiniger(c=0.0)
    with pytest.raises(ValueError):
        LiebLiniger(c=-1.0)


def test_lieb_liniger_bare_state_density_is_derivative_of_momentum():
    """Bare state density equals the derivative of the momentum charge divided by 2pi."""
    model = LiebLiniger(c=1.0)
    grid = Grid1D.uniform(-10.0, 10.0, 500, "theta")

    a = model.bare_state_density(grid)
    q1_derivative = model.charge(1, grid).derivative() / (2 * np.pi)

    assert np.allclose(a.values, q1_derivative.values, atol=1e-10)


# ---------------------------------------------------------------------------
# XXX spin chain
# ---------------------------------------------------------------------------


def test_xxx_raises_for_nonpositive_spin():
    """XXXSpinChain raises ValueError for non-positive spin."""
    with pytest.raises(ValueError):
        XXXSpinChain(S=0.0)
    with pytest.raises(ValueError):
        XXXSpinChain(S=-0.5)


def test_xxx_raises_for_non_half_integer_spin():
    """XXXSpinChain raises ValueError for non-half-integer spin."""
    with pytest.raises(ValueError):
        XXXSpinChain(S=0.3)
    with pytest.raises(ValueError):
        XXXSpinChain(S=1.7)


def test_xxx_raises_for_invalid_n_max():
    """XXXSpinChain raises ValueError for n_max < 1."""
    with pytest.raises(ValueError):
        XXXSpinChain(n_max=0)


def test_xxx_bare_state_density_is_derivative_of_momentum():
    """Bare state density equals derivative of momentum charge divided by 2pi."""
    model = XXXSpinChain(S=0.5, n_max=5)
    grid = Grid1D.uniform(-1.0, 1.0, 400, "theta")

    for n in range(1, 4):
        a = model.bare_state_density(n, grid)
        q1_derivative = model.charge(1, n, grid).derivative() / (2 * np.pi)
        assert np.allclose(a.values[50:-50], q1_derivative.values[50:-50], atol=1e-4), (
            f"Failed for string species n={n}"
        )


def test_xxx_kernel_a_n_is_2d():
    """kernel_a_n returns a 2D Field."""
    model = XXXSpinChain(S=0.5, n_max=5)
    grid = Grid1D.uniform(-5.0, 5.0, 50, "theta")
    kernel = model.kernel_a_n(1, grid)
    assert len(kernel.grids) == 2
    assert kernel.values.shape == (50, 50)


def test_xxx_a_n_is_1d():
    """a_n returns a 1D Field."""
    model = XXXSpinChain(S=0.5, n_max=5)
    grid = Grid1D.uniform(-5.0, 5.0, 50, "theta")
    a = model.a_n(1, grid)
    assert len(a.grids) == 1
    assert a.values.shape == (50,)


def test_xxx_a_n_is_lorentzian():
    """a_n is a Lorentzian with width n/2."""
    model = XXXSpinChain(S=0.5, n_max=5)
    grid = Grid1D.uniform(-5.0, 5.0, 100, "theta")
    for n in range(1, 4):
        a = model.a_n(n, grid)
        expected = n / (2 * np.pi * ((n / 2) ** 2 + grid.points**2))
        assert np.allclose(a.values, expected, atol=1e-10), (
            f"Failed for string species n={n}"
        )


def test_xxx_convolve_a1_constant():
    """Convolving a constant with a1 gives the same constant."""
    model = XXXSpinChain(S=0.5, n_max=3)
    grid = Grid1D.uniform(-20, 20, 1000, "theta")
    f = Field.from_function(lambda t: np.full_like(t, np.log(2)), [grid])
    result = model.convolve_a1(f)
    # convolution of constant c with a1 should give c * integral(a1) = c
    assert np.allclose(result.values[100:-100], np.log(2), atol=1e-4)


def test_xxx_charge_order_1_is_arctan():
    """Momentum charge for XXX is 2*arctan(2*theta/n)."""
    model = XXXSpinChain(S=0.5, n_max=5)
    grid = Grid1D.uniform(-5.0, 5.0, 100, "theta")
    for n in range(1, 4):
        q1 = model.charge(1, n, grid)
        expected = 2 * np.arctan(2 * grid.points / n)
        assert np.allclose(q1.values, expected, atol=1e-10), (
            f"Failed for string species n={n}"
        )
