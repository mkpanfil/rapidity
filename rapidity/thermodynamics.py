"""
Thermodynamic observables for integrable models.

This module provides functions for computing derived thermodynamic
quantities from a :class:`~rapidity.tba.TBAState`:

    - :func:`sound_velocity`
    - :func:`heat_capacity`
    - :func:`luttinger_parameter`
    - :func:`susceptibility`

"""

import numpy as np
from scipy.optimize import brentq
from rapidity.core import Grid1D, Field
from rapidity.tba import TBAState, StringTBAState
from rapidity.models import Model, StringModel


def free_energy(state: TBAState) -> float:
    """Compute the free energy density.

    For a single particle species, the free energy density is given by
    .. math::

        f = -\\int \\frac{d\\theta}{2\\pi}
        \\log(1 + e^{-\\epsilon(\\theta)})

    For string models, the free energy density is given by
    .. math::

        f = -\\sum_n \\int d\\theta\\,
        a_n(\\theta) \\log(1 + e^{-\\epsilon_n(\\theta)})

    Returns
    -------
    float
        The free energy density.
    """
    # epsilon = Field(
    #     np.log((1 - self.filling.values) / self.filling.values), [self.grid]
    # )
    # the code is not protected against dividing by 0.
    if isinstance(state.model, Model):
        epsilon = state.filling.apply(lambda n: np.log((1 - n) / n))
        log_term = Field(np.log(1 + np.exp(-epsilon.values)), [state.grid])
        return -(log_term / (2 * np.pi)).integrate().values
    elif isinstance(state.model, StringModel):
        total = 0.0
        for n, filling in enumerate(state.filling):
            a_n = state.model.bare_state_density(n + 1, state.grid)
            epsilon = filling.apply(lambda x: np.log((1 - x) / x))
            log_term = epsilon.apply(lambda x: np.log(1 + np.exp(-x)))
            total += -(a_n * log_term).integrate().values
        return total


def find_mu(
    model: Model,
    grid: Grid1D,
    density: float,
    T: float,
    mu_bounds: tuple[float, float] = (-10.0, 20.0),
    tol: float = 1e-10,
) -> float:
    """Find the chemical potential that reproduces a target density.

    Solves:

    .. math::

        \\int d\\theta\\, \\rho_p(\\theta) = N/L

    for the chemical potential :math:`\\mu`.

    Parameters
    ----------
    model : Model
        The integrable model.
    grid : Grid1D
        The rapidity grid.
    density : float
        Target particle density N/L.
    T : float
        Temperature.
    mu_bounds : tuple[float, float], optional
        Bounds for the chemical potential search. Default is (-10.0, 20.0).
    tol : float, optional
        Tolerance for the density matching. Default is 1e-10.

    Returns
    -------
    float
        The chemical potential.

    Raises
    ------
    ValueError
        If the target density cannot be achieved within the given bounds.
    """

    def residual(mu: float) -> float:
        state = TBAState.from_betas(model, grid, betas={2: 1 / T, 0: -mu / T})
        return state.rho_p().integrate().values - density

    try:
        return brentq(residual, *mu_bounds, xtol=tol)
    except ValueError:
        raise ValueError(
            f"Could not find chemical potential for density {density} "
            f"within bounds {mu_bounds}. Try widening mu_bounds."
        )


def sound_velocity(state: TBAState) -> float:
    """Compute the sound velocity.

    .. math::

        v_s = \\sqrt{\\frac{\\partial P}{\\partial \\rho}}

    Parameters
    ----------
    state : TBAState
        The thermodynamic state.

    Returns
    -------
    float
        The sound velocity.
    """
    ...


def magnetization(state: StringTBAState) -> float:
    """Compute the total magnetization per site.

    .. math::

        m = S - \\sum_n n \\int d\\theta\\, \\rho_p^n(\\theta)

    Parameters
    ----------
    state : StringTBAState
        The thermodynamic state.

    Returns
    -------
    float
        The magnetization per site.
    """
    rho_p = state.rho_p()
    total = sum((n + 1) * rho_p[n].integrate().values for n in range(state.model.n_max))
    return state.model.S - total
