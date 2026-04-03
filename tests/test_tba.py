import numpy as np
import pytest
from rapidity.core import Grid1D, Field
from rapidity.models import LiebLiniger
from rapidity.tba import TBAState


# ---------------------------------------------------------------------------
# TBAState construction
# ---------------------------------------------------------------------------


def test_tbastate_raises_for_wrong_grid_label():
    """TBAState raises ValueError if grid label does not match model."""
    model = LiebLiniger(c=1.0)
    grid = Grid1D.gauss_hermite(100, "x")  # wrong label
    with pytest.raises(ValueError):
        TBAState.from_betas(model, grid, betas={2: 1.0, 0: -0.5})


def test_tbastate_from_filling_roundtrip():
    """Constructing from filling and retrieving it gives the same field."""
    model = LiebLiniger(c=1.0)
    grid = Grid1D.gauss_hermite(100, "theta")
    n = Field.from_function(lambda t: 1 / (1 + np.exp(t**2)), [grid])
    state = TBAState.from_filling(model, grid, n)
    assert np.allclose(state.filling.values, n.values)


# ---------------------------------------------------------------------------
# Zero temperature
# ---------------------------------------------------------------------------


def test_zero_temperature_filling_is_one():
    """Zero temperature filling function is 1 everywhere."""
    model = LiebLiniger(c=1.0)
    state = TBAState.zero_temperature(model, theta_f=1.0)
    assert np.allclose(state.filling.values, 1.0)


def test_zero_temperature_tonks_girardeau_density():
    """In Tonks-Girardeau limit rho_p = 1/(2pi) inside Fermi sea."""
    model = LiebLiniger(c=1e6)  # approximate Tonks-Girardeau
    theta_f = 1.0
    state = TBAState.zero_temperature(model, theta_f=theta_f)
    expected = 1 / (2 * np.pi)
    assert np.allclose(state.rho_p().values, expected, atol=1e-4)


def test_zero_temperature_tonks_girardeau_total_density():
    """In Tonks-Girardeau limit total density is theta_f / pi."""
    model = LiebLiniger(c=1e6)
    theta_f = 1.0
    state = TBAState.zero_temperature(model, theta_f=theta_f)
    expected = theta_f / np.pi
    assert np.isclose(state.rho_p().integrate().values, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Free fermion limit
# ---------------------------------------------------------------------------


def test_free_fermion_limit():
    """In the free fermion limit (c -> inf) filling reduces to Fermi-Dirac."""
    model = LiebLiniger(c=1e6)  # approximate free fermion
    grid = Grid1D.gauss_hermite(200, "theta")
    beta, mu = 1.0, 0.5
    state = TBAState.from_betas(model, grid, betas={2: beta, 0: -mu})

    # expected free Fermi-Dirac filling
    expected = Field.from_function(lambda t: 1 / (1 + np.exp(beta * t**2 - mu)), [grid])
    assert np.allclose(state.filling.values, expected.values, atol=1e-4)


# ---------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------


def test_filling_symmetric_for_zero_momentum():
    """Filling function is symmetric for zero momentum chemical potential."""
    model = LiebLiniger(c=1.0)
    grid = Grid1D.gauss_hermite(200, "theta")
    state = TBAState.from_betas(model, grid, betas={2: 1.0, 0: -0.5})
    assert np.allclose(state.filling.values, state.filling.values[::-1], atol=1e-8)


# ---------------------------------------------------------------------------
# Yang-Yang relation
# ---------------------------------------------------------------------------


def test_yang_yang_relation():
    """State density satisfies rho_s = a + K * rho_p."""
    model = LiebLiniger(c=1.0)
    grid = Grid1D.gauss_hermite(200, "theta")
    state = TBAState.from_betas(model, grid, betas={2: 1.0, 0: -0.5})

    rho_s = state.rho_s()
    rho_p = state.rho_p()
    a = model.bare_state_density(grid)
    kernel = model.kernel(grid)

    rho_s_check = a + rho_p.convolve(kernel)
    assert np.allclose(rho_s.values, rho_s_check.values, atol=1e-8)


# ---------------------------------------------------------------------------
# Dressing consistency
# ---------------------------------------------------------------------------


def test_dressing_consistency():
    """Dressing the bare state density gives rho_s."""
    model = LiebLiniger(c=1.0)
    grid = Grid1D.gauss_hermite(200, "theta")
    state = TBAState.from_betas(model, grid, betas={2: 1.0, 0: -0.5})

    rho_s = state.rho_s()
    a_dr = state.dress(model.bare_state_density(grid))
    assert np.allclose(rho_s.values, a_dr.values, atol=1e-8)


# ---------------------------------------------------------------------------
# Physical sanity checks
# ---------------------------------------------------------------------------


def test_filling_between_zero_and_one():
    """Filling function values are between 0 and 1."""
    model = LiebLiniger(c=1.0)
    grid = Grid1D.gauss_hermite(200, "theta")
    state = TBAState.from_betas(model, grid, betas={2: 1.0, 0: -0.5})
    assert np.all(state.filling.values >= 0)
    assert np.all(state.filling.values <= 1)


def test_rho_p_positive():
    """Particle density is non-negative."""
    model = LiebLiniger(c=1.0)
    grid = Grid1D.gauss_hermite(200, "theta")
    state = TBAState.from_betas(model, grid, betas={2: 1.0, 0: -0.5})
    assert np.all(state.rho_p().values >= 0)


def test_rho_s_positive():
    """State density is non-negative."""
    model = LiebLiniger(c=1.0)
    grid = Grid1D.gauss_hermite(200, "theta")
    state = TBAState.from_betas(model, grid, betas={2: 1.0, 0: -0.5})
    assert np.all(state.rho_s().values >= 0)


def test_from_density_roundtrip():
    """Constructing from rho_p and computing rho_p gives the same result."""
    model = LiebLiniger(c=1.0)
    grid = Grid1D.gauss_hermite(200, "theta")
    state = TBAState.from_betas(model, grid, betas={2: 1.0, 0: -0.5})
    rho_p = state.rho_p()
    state2 = TBAState.from_density(model, grid, rho_p)
    assert np.allclose(state2.rho_p().values, rho_p.values, atol=1e-8)
