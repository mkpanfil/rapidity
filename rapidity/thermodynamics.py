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
from rapidity.tba import TBAState, Model


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
