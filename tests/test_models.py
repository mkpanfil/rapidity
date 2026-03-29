import pytest
import numpy as np
from rapidity.core import Grid1D
from rapidity.models import LiebLiniger


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
