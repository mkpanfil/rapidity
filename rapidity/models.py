"""
Physical models for the rapidity package.

This module defines the :class:`Model` and :class:`StringModel` protocols,
which specify the interfaces that integrable models must implement, and
concrete implementations of specific models.

Currently implemented:

- :class:`LiebLiniger` — satisfies :class:`Model`
- :class:`HardRods` — satisfies :class:`Model`
- :class:`XXXSpinChain` — satisfies :class:`StringModel`

To implement a new single-species model, create a dataclass that satisfies
the :class:`Model` protocol. For a model with string hypothesis, satisfy
the :class:`StringModel` protocol instead.
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


@runtime_checkable
class StringModel(Protocol):
    """Protocol for integrable models with string hypothesis."""

    n_max: int
    rapidity_label: str

    def charge(self, order: int, species: int, grid: Grid1D) -> Field:
        """Single-particle eigenvalue of the conserved charge of given order
        for string of length species."""
        ...

    def bare_state_density(self, species: int, grid: Grid1D) -> Field:
        """Bare density of states a_n(theta) for string of length species."""
        ...

    def a_n(self, n: int, grid: Grid1D) -> Field:
        """Basic kernel a_n(theta) as a 1D Field."""
        ...

    def kernel_a_n(self, n: int, grid: Grid1D) -> Field:
        """Basic kernel a_n(theta) as a 2D Field for convolution."""
        ...

    def convolve_a1(self, f: Field) -> Field: ...

    def driving(self, grid: Grid1D, betas: dict[int, float]) -> list[Field]:
        """Driving terms for all string species."""
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


# ---------------------------------------------------------------------------
# XXX spin chain
# ---------------------------------------------------------------------------


@dataclass
class XXXSpinChain:
    """XXX Heisenberg spin chain with string hypothesis.

    The model uses the simplified TBA equations that couple only
    neighbouring string species, requiring only the basic kernel
    a_1. The full kernel T_{nm} is also provided for reference.

    Parameters
    ----------
    S : float
        Spin. Must be a positive half-integer. Default is 0.5.
    n_max : int
        Maximum string length. Default is 10.
    rapidity_label : str
        Label for the rapidity dimension. Default is 'theta'.

    Examples
    --------
    >>> model = XXXSpinChain(S=0.5, n_max=10)
    >>> grid = Grid1D.gauss_legendre(-10, 10, 200, "theta")
    >>> a1 = model.kernel_a_n(1, grid)
    """

    S: float = 0.5
    n_max: int = 10
    rapidity_label: str = "theta"

    def __post_init__(self):
        if self.S <= 0:
            raise ValueError(f"Spin S must be positive, got {self.S}")
        if not (self.S * 2).is_integer():
            raise ValueError(f"Spin S must be a half-integer, got {self.S}")
        if self.n_max < 1:
            raise ValueError(
                f"Maximum string length n_max must be at least 1, got {self.n_max}"
            )
        if self.S >= 1:
            raise ValueError(
                f"Only spin S=1/2 properly implemented for XXX chain, got {self.S}"
            )

    def a_n(self, n: int, grid: Grid1D) -> Field:
        """Basic kernel a_n(theta) as a 1D Field.

        .. math::

            a_n(\\theta) = \\frac{1}{2\\pi}
            \\frac{n}{(n/2)^2 + \\theta^2}

        Parameters
        ----------
        n : int
            String length.
        grid : Grid1D
            The rapidity grid.

        Returns
        -------
        Field
            The basic kernel as a 1D Field.
        """
        return Field.from_function(
            lambda t: n / (2 * np.pi * ((n / 2) ** 2 + t**2)), [grid]
        )

    def kernel_a_n(self, n: int, grid: Grid1D) -> Field:
        """Basic kernel a_n(theta) as a 2D Field for convolution.

        Parameters
        ----------
        n : int
            String length.
        grid : Grid1D
            The rapidity grid.

        Returns
        -------
        Field
            The basic kernel as a 2D Field.
        """
        return make_kernel(lambda t: n / (2 * np.pi * ((n / 2) ** 2 + t**2)), grid)

    def bare_state_density(self, species: int, grid: Grid1D) -> Field:
        """Bare state density for string of length species.

        .. math::

            a_n(\\theta) = \\frac{1}{2\\pi}
            \\frac{n}{(n/2)^2 + \\theta^2}

        Parameters
        ----------
        species : int
            String length n, starting from 1.
        grid : Grid1D
            The rapidity grid.

        Returns
        -------
        Field
            The bare state density as a 1D Field.
        """
        return self.a_n(species, grid)

    def charge(self, order: int, species: int, grid: Grid1D) -> Field:
        """Single-particle eigenvalue of the conserved charge of given order
        for string of length species.

        For order=1 the momentum charge is:

        .. math::

            q_1^{(n)}(\\theta) = 2\\arctan\\left(\\frac{2\\theta}{n}\\right)

        For higher orders the charges involve the digamma function:

        .. math::

            q_r^{(n)}(\\theta) = \\frac{i}{r-1}\\left(
            \\psi\\left(\\frac{n}{2} + i\\theta + \\frac{r-1}{2}\\right) -
            \\psi\\left(\\frac{n}{2} - i\\theta + \\frac{r-1}{2}\\right)
            \\right)

        Parameters
        ----------
        order : int
            Order of the charge, starting from 1.
        species : int
            String length n, starting from 1.
        grid : Grid1D
            The rapidity grid.

        Returns
        -------
        Field
            The charge eigenvalue as a 1D Field.
        """
        from scipy.special import digamma

        if order == 1:
            return Field.from_function(lambda t: 2 * np.arctan(2 * t / species), [grid])

        def charge_values(t: np.ndarray) -> np.ndarray:
            n = species
            r = order
            return (
                1j
                / (r - 1)
                * (
                    digamma(n / 2 + 1j * t + (r - 1) / 2)
                    - digamma(n / 2 - 1j * t + (r - 1) / 2)
                )
            ).real

        return Field.from_function(charge_values, [grid])

    def kernel(self, n: int, m: int, grid: Grid1D) -> Field:
        """Full scattering kernel T_{nm} as a 2D Field.

        For the simplified TBA only kernel_a_n(1, grid) is needed.
        This method provides the full kernel for reference.

        .. math::

            T_{nm}(\\theta) = (1-\\delta_{nm}) a_{|n-m|} +
            2a_{|n-m|+2} + \\ldots + 2a_{n+m-2} + a_{n+m}

        Parameters
        ----------
        n : int
            First string length.
        m : int
            Second string length.
        grid : Grid1D
            The rapidity grid.

        Returns
        -------
        Field
            The scattering kernel as a 2D Field.
        """

        def T_nm(t: np.ndarray) -> np.ndarray:
            result = np.zeros_like(t)
            for k in range(abs(n - m), n + m + 1, 2):
                if k == 0:
                    continue
                prefactor = 1 if (k == abs(n - m) or k == n + m) else 2
                result += prefactor * self.a_n(k, grid).values
            return result

        return make_kernel(T_nm, grid)

    def driving(self, grid: Grid1D, betas: dict[int, float]) -> list[Field]:
        """Driving terms for all string species with J=1.

        .. math::

            \\epsilon_n^0(\\theta) = \\delta_{n,1} \\beta_2 \\cdot 2\\pi a_1(\\theta)
            - n \\beta_0

        where :math:`\\beta_2 = 1/T` is the inverse temperature and
        :math:`\\beta_0 = h/T` is the reduced magnetic field.

        Parameters
        ----------
        grid : Grid1D
            The rapidity grid.
        betas : dict[int, float]
            Chemical potentials keyed by charge order:
            - key 0: reduced magnetic field h/T
            - key 2: inverse temperature 1/T

        Returns
        -------
        list[Field]
            Driving terms for each string species n=1,...,n_max.
        """
        beta_2 = betas.get(2, 0.0)  # inverse temperature
        beta_0 = betas.get(0, 0.0)  # reduced magnetic field

        a1 = self.a_n(1, grid)

        driving = []
        for n in range(1, self.n_max + 1):
            if n == 1:
                d = a1 * (2 * np.pi * beta_2) - Field.from_function(
                    lambda t: np.full_like(t, beta_0), [grid]
                )
            else:
                d = Field.from_function(lambda t: np.full_like(t, -n * beta_0), [grid])
            driving.append(d)
        return driving

    def convolve_a1(self, f: Field) -> Field:
        """Convolve f with the basic kernel a_1 using FFT.

        Uses the analytical Fourier transform of a_1:

        .. math::

            \\hat{a}_1(k) = e^{-|k|/2}

        which avoids discretization issues with the Lorentzian kernel.

        Parameters
        ----------
        f : Field
            The field to convolve. Must be defined on a uniform grid.

        Returns
        -------
        Field
            The convolution result on the same grid.
        """
        grid = f.grids[0]
        dx = grid.points[1] - grid.points[0]
        n = len(grid.points)

        k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
        a1_hat = np.exp(-np.abs(k) / 2)
        f_hat = np.fft.fft(f.values)
        result = np.fft.ifft(a1_hat * f_hat).real

        return Field(result, f.grids)
