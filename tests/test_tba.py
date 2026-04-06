import numpy as np
import pytest
from rapidity.core import Grid1D, Field
from rapidity.models import LiebLiniger, XXXSpinChain
from rapidity.tba import TBAState, StringTBAState


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
# StringTBAState construction
# ---------------------------------------------------------------------------


def test_string_tbastate_raises_for_wrong_grid_label():
    """StringTBAState raises ValueError if grid label does not match model."""
    model = XXXSpinChain(S=0.5, n_max=3)
    grid = Grid1D.gauss_legendre(-10, 10, 100, "x")  # wrong label
    with pytest.raises(ValueError):
        StringTBAState.from_betas(model, grid, betas={1: 0.1})


def test_string_tbastate_has_correct_number_of_fillings():
    """StringTBAState has one filling per string species."""
    model = XXXSpinChain(S=0.5, n_max=5)
    grid = Grid1D.gauss_legendre(-10, 10, 100, "theta")
    state = StringTBAState.from_betas(model, grid, betas={1: 1, 0: 0.5})
    assert len(state.filling) == model.n_max


def test_string_tbastate_from_filling_roundtrip():
    """Constructing from filling and retrieving it gives the same fields."""
    model = XXXSpinChain(S=0.5, n_max=3)
    grid = Grid1D.gauss_legendre(-10, 10, 100, "theta")
    filling = [
        Field.from_function(lambda t: np.full_like(t, 0.5), [grid])
        for _ in range(model.n_max)
    ]
    state = StringTBAState.from_filling(model, grid, filling)
    for n in range(model.n_max):
        assert np.allclose(state.filling[n].values, filling[n].values)


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
# High temperature
# ---------------------------------------------------------------------------


def test_string_tba_high_temperature_filling():
    """At high temperature fillings approach the finite n_max fixed point."""
    n_max = 5
    model = XXXSpinChain(S=0.5, n_max=n_max)
    grid = Grid1D.uniform(-20, 20, 1000, "theta")
    T = 1000.0
    state = StringTBAState.from_betas(model, grid, betas={2: 1 / T, 0: 0.0})

    # compute expected Y values by solving fixed point equations
    Y = np.array([(n + 1) * (n + 3) for n in range(n_max)], dtype=float)
    for _ in range(100000):
        Y_new = np.zeros_like(Y)
        for n in range(n_max):
            left = 1 + 1 / Y[n - 1] if n > 0 else 1.0
            right = 1 + 1 / Y[n + 1] if n < n_max - 1 else 1.0
            Y_new[n] = left * right
        if np.max(np.abs(Y_new - Y)) < 1e-12:
            break
        Y = Y_new

    expected_filling = 1 / (1 + Y)

    for n, filling in enumerate(state.filling):
        assert np.allclose(filling.values[100:-100], expected_filling[n], atol=1e-2), (
            f"Failed for species n={n + 1}"
        )


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


def test_string_tba_filling_symmetric_for_zero_field():
    """Filling functions are symmetric for zero magnetic field."""
    model = XXXSpinChain(S=0.5, n_max=5)
    grid = Grid1D.gauss_legendre(-10, 10, 200, "theta")
    state = StringTBAState.from_betas(model, grid, betas={2: 1.0})
    for n, filling in enumerate(state.filling):
        assert np.allclose(filling.values, filling.values[::-1], atol=1e-8), (
            f"Failed for string species n={n + 1}"
        )


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


def test_string_yang_yang_relation():
    """State density satisfies the simplified Yang-Yang relation.

    Uses the neighbouring coupling form:
    rho_s^n = a_n + a_1 * (rho_p^{n-1} + rho_p^{n+1})
    """
    model = XXXSpinChain(S=0.5, n_max=3)
    grid = Grid1D.uniform(-20, 20, 1000, "theta")
    state = StringTBAState.from_betas(model, grid, betas={2: 1.0, 0: 0.1})

    rho_s = state.rho_s()
    rho_p = state.rho_p()
    zero = Field.from_function(lambda t: np.zeros_like(t), [grid])

    for n in range(model.n_max):
        a_n = model.bare_state_density(n + 1, grid)
        left = model.convolve_a1(rho_p[n - 1]) if n > 0 else zero
        right = model.convolve_a1(rho_p[n + 1]) if n < model.n_max - 1 else zero
        rho_s_check = a_n + left + right
        assert np.allclose(
            rho_s[n].values[100:-100], rho_s_check.values[100:-100], atol=1e-8
        ), f"Yang-Yang relation failed for species n={n + 1}"


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


def test_string_dressing_consistency():
    """Dressing bare state densities gives rho_s."""
    model = XXXSpinChain(S=0.5, n_max=5)
    grid = Grid1D.uniform(-20, 20, 1000, "theta")
    state = StringTBAState.from_betas(model, grid, betas={2: 1.0, 0: -0.5})

    rho_s = state.rho_s()
    a = [model.bare_state_density(n + 1, grid) for n in range(model.n_max)]
    a_dr = state.dress(a)

    for n in range(model.n_max):
        assert np.allclose(rho_s[n].values, a_dr[n].values, atol=1e-8), (
            f"Dressing consistency failed for species n={n + 1}"
        )


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


def test_string_tba_filling_between_zero_and_one():
    """All filling functions are between 0 and 1."""
    model = XXXSpinChain(S=0.5, n_max=5)
    grid = Grid1D.gauss_legendre(-10, 10, 100, "theta")
    state = StringTBAState.from_betas(model, grid, betas={2: 1.0, 0: -0.5})
    for n, filling in enumerate(state.filling):
        assert np.all(filling.values >= 0), f"Negative filling for species n={n + 1}"
        assert np.all(filling.values <= 1), f"Filling > 1 for species n={n + 1}"


def test_string_tba_rho_p_positive():
    """All particle densities are non-negative."""
    model = XXXSpinChain(S=0.5, n_max=5)
    grid = Grid1D.uniform(-20, 20, 1000, "theta")
    state = StringTBAState.from_betas(model, grid, betas={2: 1.0, 0: -0.5})
    for n, rho_p in enumerate(state.rho_p()):
        assert np.all(rho_p.values >= 0), f"Negative rho_p for species n={n + 1}"


def test_string_tba_rho_s_positive():
    """All state densities are non-negative."""
    model = XXXSpinChain(S=0.5, n_max=5)
    grid = Grid1D.uniform(-20, 20, 1000, "theta")
    state = StringTBAState.from_betas(model, grid, betas={2: 1.0, 0: -0.5})
    for n, rho_s in enumerate(state.rho_s()):
        assert np.all(rho_s.values >= 0), f"Negative rho_s for species n={n + 1}"
