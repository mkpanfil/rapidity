import pytest
import numpy as np
from rapidity.core import Grid1D
from rapidity.models import Model, LiebLiniger
from rapidity.thermodynamics import find_mu
from rapidity.tba import TBAState


def test_find_mu_reproduces_density():
    """find_mu finds chemical potential that reproduces target density."""
    model = LiebLiniger(c=1.0)
    grid = Grid1D.gauss_hermite(200, "theta")
    target_density = 1.0
    T = 0.5
    mu = find_mu(model, grid, target_density, T)
    state = TBAState.from_betas(model, grid, betas={2: 1 / T, 0: -mu / T})
    assert np.isclose(state.rho_p().integrate().values, target_density, atol=1e-8)


def test_find_mu_increases_with_density():
    """Chemical potential increases with density at fixed temperature."""
    model = LiebLiniger(c=1.0)
    grid = Grid1D.gauss_hermite(200, "theta")
    T = 0.5
    mu1 = find_mu(model, grid, density=0.5, T=T)
    mu2 = find_mu(model, grid, density=1.0, T=T)
    assert mu2 > mu1


def test_find_mu_raises_for_unreachable_density():
    """find_mu raises ValueError if density cannot be achieved within bounds."""
    model = LiebLiniger(c=1.0)
    grid = Grid1D.gauss_hermite(200, "theta")
    with pytest.raises(ValueError):
        find_mu(model, grid, density=1000.0, T=0.5, mu_bounds=(-10.0, 10.0))
