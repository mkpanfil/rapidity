"""
Physical models for the rapidity package.

This module defines classes representing specific integrable models.
Each model encodes the scattering kernel and single-particle charges
needed for TBA and GHD calculations. The thermodynamic state is
represented separately by :class:`~rapidity.tba.TBAState`.

Currently implemented:

- :class:`LiebLiniger` — the Lieb-Liniger model of bosons with delta
  interaction
"""

import numpy as np
from dataclasses import dataclass
from rapidity.core import Grid1D, Field
from rapidity.utils import make_kernel


@dataclass
class LiebLiniger:
    """The Lieb-Liniger model of bosons with repulsive delta interaction.

    The model is parameterized by the coupling constant c only. The
    thermodynamic state is encoded separately in :class:`~rapidity.tba.TBAState`.
    The scattering kernel includes the 1/(2π) factor by convention, so
    the TBA equation reads:

    .. math::

        \\epsilon(\\theta) = \\epsilon_0(\\theta) -
        \\int d\\theta'\\, K(\\theta - \\theta')
        \\log(1 + e^{-\\epsilon(\\theta')})

    Parameters
    ----------
    c : float
        Coupling constant. Must be positive for repulsive interactions.

    Examples
    --------
    >>> model = LiebLiniger(c=1.0)
    >>> grid = Grid1D.gauss_hermite(200, "theta")
    >>> kernel = model.kernel(grid)
    """

    c: float

    def __post_init__(self):
        if self.c <= 0:
            raise ValueError(f"Coupling constant c must be positive, got {self.c}")

    def charge(self, order: int, grid: Grid1D) -> Field:
        """Single-particle eigenvalue of the conserved charge of given order.

        .. math::

            q_s(\\theta) = \\theta^s

        Parameters
        ----------
        order : int
            Order of the charge.
        grid : Grid1D
            The rapidity grid.

        Returns
        -------
        Field
            The charge eigenvalue as a 1D Field.
        """
        return Field.from_function(lambda t: t**order, [grid])

    def driving(self, grid: Grid1D, betas: dict[int, float]) -> Field:
        """Driving term as a linear combination of charge eigenvalues.

        .. math::

            \\epsilon_0(\\theta) = \\sum_s \\beta_s \\theta^s

        Parameters
        ----------
        grid : Grid1D
            The rapidity grid.
        betas : dict[int, float]
            Chemical potentials keyed by charge order. For example
            ``{2: 1/T, 0: -mu/T}`` gives a thermal state at temperature
            T with chemical potential mu.

        Returns
        -------
        Field
            The driving term as a 1D Field.
        """
        return sum(beta * self.charge(s, grid) for s, beta in betas.items())

    def bare_state_density(self, grid: Grid1D) -> Field:
        """Bare density of states: a(theta) = 1/(2pi)."""
        return Field.from_function(lambda t: np.ones_like(t) / (2 * np.pi), [grid])

    def kernel(self, grid: Grid1D) -> Field:
        """Lieb-Liniger scattering kernel including the 1/(2π) factor.

        .. math::

            K(\\theta) = \\frac{c}{\\pi(c^2 + \\theta^2)}

        Parameters
        ----------
        grid : Grid1D
            The rapidity grid.

        Returns
        -------
        Field
            The scattering kernel as a 2D Field.
        """
        return make_kernel(lambda t: self.c / (np.pi * (self.c**2 + t**2)), grid)
