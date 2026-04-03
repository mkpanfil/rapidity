"""
Physical models for the rapidity package.

This module defines the :class:`Model` protocol, which specifies the
interface that all integrable models must implement, and concrete
implementations of specific models.

Currently implemented:

- :class:`LiebLiniger` — the Lieb-Liniger model of bosons with delta
  interaction

To implement a new model, create a dataclass that satisfies the
:class:`Model` protocol by implementing all required methods.
"""

import numpy as np
from typing import Protocol, runtime_checkable
from dataclasses import dataclass
from rapidity.core import Grid1D, Field
from rapidity.utils import make_kernel


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@runtime_checkable
class Model(Protocol):
    """Protocol for integrable models.

    Any class implementing these methods can be used as a model
    in :class:`TBAState`.
    """

    rapidity_label: str

    def charge(self, order: int, grid: Grid1D) -> Field:
        """Single-particle eigenvalue of the conserved charge of given order."""
        ...

    def driving(self, grid: Grid1D, betas: dict[int, float]) -> Field:
        """Driving term as a linear combination of charge eigenvalues."""
        ...

    def bare_state_density(self, grid: Grid1D) -> Field:
        """Bare density of states a(theta)."""
        ...

    def kernel(self, grid: Grid1D) -> Field:
        """Scattering kernel including the 1/(2π) factor."""
        ...


# ---------------------------------------------------------------------------
# LiebLiniger
# ---------------------------------------------------------------------------


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
    rapidity_label: str = "theta"

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


# ---------------------------------------------------------------------------
# QHR
# ---------------------------------------------------------------------------


@dataclass
class QHR:
    """The Quantum hard rods model.

    The model is parameterized by the rod length a only. The
    thermodynamic state is encoded separately in :class:`~rapidity.tba.TBAState`.
    The scattering kernel includes the 1/(2π) factor by convention, so
    the TBA equation reads:

    .. math::

        \\epsilon(\\theta) = \\epsilon_0(\\theta) -
        \\int d\\theta'\\, K(\\theta - \\theta')
        \\log(1 + e^{-\\epsilon(\\theta')})

    Parameters
    ----------
    a : float
        Rod length. Must be positive.

    Examples
    --------
    >>> model = QHR(a=1.0)
    >>> grid = Grid1D.gauss_hermite(200, "theta")
    >>> kernel = model.kernel(grid)
    """

    a: float
    rapidity_label: str = "theta"

    # def __post_init__(self):
    #     if self.a <= 0:
    #         raise ValueError(f"Rod length a must be positive, got {self.a}")

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
        """QHR scattering kernel including the 1/(2π) factor.

        .. math::

            K(\\theta) = -a

        Parameters
        ----------
        grid : Grid1D
            The rapidity grid.

        Returns
        -------
        Field
            The scattering kernel as a 2D Field.
        """
        return make_kernel(lambda t: -self.a / (2 * np.pi), grid)
