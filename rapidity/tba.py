"""
Thermodynamic Bethe Ansatz solver.

This module provides the :class:`TBAState` class representing the
thermodynamic state of an integrable model, and the tools to compute
thermodynamic quantities from it.

The TBA equation for a single-species model reads:

.. math::

    \\epsilon(\\theta) = \\epsilon_0(\\theta) -
    \\int d\\theta'\\, K(\\theta - \\theta')
    \\log(1 + e^{-\\epsilon(\\theta')})

where :math:`\\epsilon_0` is the driving term, :math:`K` is the
scattering kernel, and the filling function is:

.. math::

    n(\\theta) = \\frac{1}{1 + e^{\\epsilon(\\theta)}}

The state can be constructed from chemical potentials, a filling
function, or a particle density.

"""

import numpy as np
from dataclasses import dataclass
from rapidity.core import Grid1D, Field
from rapidity.models import Model


def _solve_tba(
    driving: Field, kernel: Field, tol: float = 1e-10, max_iter: int = 1000
) -> Field:
    """Solve the TBA equation by fixed-point iteration.

    Parameters
    ----------
    driving : Field
        The driving term epsilon_0(theta).
    kernel : Field
        The scattering kernel K(theta, theta') as a 2D Field.
    tol : float, optional
        Convergence tolerance. Default is 1e-10.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.

    Returns
    -------
    Field
        The pseudoenergy epsilon(theta).

    Raises
    ------
    RuntimeError
        If the iteration does not converge within max_iter iterations.
    """
    epsilon = Field(driving.values.copy(), driving.grids)

    for _ in range(max_iter):
        # log_term = Field(np.log(1 + np.exp(-epsilon.values)), epsilon.grids)
        log_term = epsilon.apply(lambda x: np.log(1 + np.exp(-x)))
        epsilon_new = driving - log_term.convolve(kernel, dim="theta")
        if np.max(np.abs((epsilon_new - epsilon).values)) < tol:
            return epsilon_new
        epsilon = epsilon_new

    raise RuntimeError(
        f"TBA iteration did not converge after {max_iter} iterations. "
        f"Try increasing max_iter or using a finer grid."
    )


def _dress(
    h: Field, filling: Field, kernel: Field, tol: float = 1e-10, max_iter: int = 1000
) -> Field:
    """Solve the dressing equation by fixed-point iteration.

    Parameters
    ----------
    h : Field
        The bare quantity to dress.
    filling : Field
        The filling function n(theta).
    kernel : Field
        The scattering kernel as a 2D Field.
    tol : float, optional
        Convergence tolerance. Default is 1e-10.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.

    Returns
    -------
    Field
        The dressed quantity h^dr.

    Raises
    ------
    RuntimeError
        If the iteration does not converge within max_iter iterations.
    """
    h_dr = Field(h.values.copy(), h.grids)

    for _ in range(max_iter):
        h_dr_new = h + (filling * h_dr).convolve(kernel, dim=h.grids[0].label)
        if np.max(np.abs((h_dr_new - h_dr).values)) < tol:
            return h_dr_new
        h_dr = h_dr_new

    raise RuntimeError(
        f"Dressing iteration did not converge after {max_iter} iterations."
    )


def _check_grid(model: Model, grid: Grid1D) -> None:
    """Check that the grid label matches the model's rapidity label."""
    if grid.label != model.rapidity_label:
        raise ValueError(
            f"Grid label '{grid.label}' does not match model's rapidity "
            f"label '{model.rapidity_label}'. "
            f"Create your grid with Grid1D.gauss_hermite(n, '{model.rapidity_label}')."
        )


@dataclass
class TBAState:
    """Thermodynamic state of an integrable model.

    The state is characterized by the filling function n(theta),
    which encodes the occupation of rapidity modes. It always
    carries the model and grid alongside the filling function.

    Parameters
    ----------
    model : Model
        The integrable model.
    grid : Grid1D
        The rapidity grid.
    filling : Field
        The filling function n(theta).

    Examples
    --------
    >>> from rapidity.models import LiebLiniger
    >>> from rapidity.core import Grid1D
    >>> model = LiebLiniger(c=1.0)
    >>> grid = Grid1D.gauss_hermite(200, "theta")
    >>> state = TBAState.from_betas(model, grid, betas={2: 1.0, 0: -0.5})
    """

    model: Model
    grid: Grid1D
    filling: Field

    @classmethod
    def from_betas(
        cls,
        model: Model,
        grid: Grid1D,
        betas: dict[int, float],
        tol: float = 1e-10,
        max_iter: int = 1000,
    ) -> "TBAState":
        """Construct state by solving the TBA equation from chemical potentials.

        Parameters
        ----------
        model : Model
            The integrable model.
        grid : Grid1D
            The rapidity grid.
        betas : dict[int, float]
            Chemical potentials keyed by charge order.
        tol : float, optional
            Convergence tolerance for the TBA iteration. Default is 1e-10.
        max_iter : int, optional
            Maximum number of iterations. Default is 1000.

        Returns
        -------
        TBAState
            The thermodynamic state.
        """
        _check_grid(model, grid)
        driving = model.driving(grid, betas)
        kernel = model.kernel(grid)
        epsilon = _solve_tba(driving, kernel, tol, max_iter)
        #filling = Field(1 / (1 + np.exp(epsilon.values)), [grid])
        filling = epsilon.apply(lambda x: 1 / (1 + np.exp(x)))
        return cls(model, grid, filling)

    @classmethod
    def from_filling(cls, model: Model, grid: Grid1D, filling: Field) -> "TBAState":
        """Construct state directly from a filling function.

        Parameters
        ----------
        model : Model
            The integrable model.
        grid : Grid1D
            The rapidity grid.
        filling : Field
            The filling function n(theta).

        Returns
        -------
        TBAState
            The thermodynamic state.
        """
        _check_grid(model, grid)
        return cls(model, grid, filling)

    @classmethod
    def from_density(cls, model: Model, grid: Grid1D, rho_p: Field) -> "TBAState":
        """Construct state from particle density rho_p(theta).

        Parameters
        ----------
        model : Model
            The integrable model.
        grid : Grid1D
            The rapidity grid.
        rho_p : Field
            The particle density.

        Returns
        -------
        TBAState
            The thermodynamic state.
        """
        _check_grid(model, grid)
        kernel = model.kernel(grid)
        a = model.bare_state_density(grid)
        rho_s = a + rho_p.convolve(kernel)
        filling = rho_p / rho_s
        return cls(model, grid, filling)

    @classmethod
    def zero_temperature(
        cls, model: Model, theta_f: float, n_points: int = 200
    ) -> "TBAState":
        """Construct zero temperature ground state.

        At zero temperature all states within the Fermi sea are filled.
        The particle density satisfies the linear integral equation:

        .. math::

            \\rho_p(\\theta) = a(\\theta) + \\int_{-\\theta_F}^{\\theta_F}
            K(\\theta - \\theta') \\rho_p(\\theta') d\\theta'

        which is equivalent to dressing with uniform filling n=1.

        Parameters
        ----------
        model : Model
            The integrable model.
        theta_f : float
            The Fermi rapidity.
        n_points : int, optional
            Number of Gauss-Legendre quadrature points. Default is 200.

        Returns
        -------
        TBAState
            The zero temperature ground state with filling n=1 everywhere.
        """
        label = model.rapidity_label
        grid = Grid1D.gauss_legendre(-theta_f, theta_f, n_points, label)
        filling = Field.from_function(lambda t: np.ones_like(t), [grid])
        return cls.from_filling(model, grid, filling)

    def bare_state_density(self) -> Field:
        """Bare density of states a(theta).

        Delegates to the model's implementation of bare_state_density,
        which satisfies:

        .. math::

            a(\\theta) = \\frac{1}{2\\pi} \\partial_\\theta q_1(\\theta)

        Returns
        -------
        Field
            The bare density of states as a 1D Field.
        """
        return self.model.bare_state_density(self.grid)

    def dress(self, h: Field) -> Field:
        """Compute the dressed quantity h^dr via the dressing equation.

        .. math::

            h^{dr}(\\theta) = h(\\theta) + \\int d\\theta'\\,
            K(\\theta - \\theta') n(\\theta') h^{dr}(\\theta')

        Parameters
        ----------
        h : Field
            The bare quantity to dress.

        Returns
        -------
        Field
            The dressed quantity.
        """
        kernel = self.model.kernel(self.grid)
        return _dress(h, self.filling, kernel)

    def rho_s(self) -> Field:
        """Compute the density of states rho_s(theta).

        .. math::

            \\rho_s(\\theta) = a^{dr}(\\theta)

        where :math:`a(\\theta)` is the bare state density.

        Returns
        -------
        Field
            The density of states as a 1D Field.
        """
        return self.dress(self.bare_state_density())

    def rho_p(self) -> Field:
        """Compute the particle density rho_p(theta).

        .. math::

            \\rho_p(\\theta) = n(\\theta) \\rho_s(\\theta)

        Returns
        -------
        Field
            The particle density as a 1D Field.
        """
        return self.filling * self.rho_s()

    def v_eff(self) -> Field:
        """Compute the effective velocity v^eff(theta).

        .. math::

            v^{eff}(\\theta) = \\frac{(\\partial_\\theta q_2)^{dr}}
            {(\\partial_\\theta q_1)^{dr}}

        where :math:`q_1` and :math:`q_2` are the momentum and energy
        charges respectively.

        Returns
        -------
        Field
            The effective velocity as a 1D Field.
        """
        momentum = self.model.charge(1, self.grid)
        energy = self.model.charge(2, self.grid)
        return self.dress(energy.derivative()) / self.dress(momentum.derivative())

    def free_energy(self) -> float:
        """Compute the free energy density.

        .. math::

            f = -\\int \\frac{d\\theta}{2\\pi}
            \\log(1 + e^{-\\epsilon(\\theta)})

        Returns
        -------
        float
            The free energy density.
        """
        # epsilon = Field(
        #     np.log((1 - self.filling.values) / self.filling.values), [self.grid]
        # )
        # the code is not protected against dividing by 0.
        epsilon = self.filling.apply(lambda n: np.log((1 - n) / n))
        log_term = Field(np.log(1 + np.exp(-epsilon.values)), [self.grid])
        return -(log_term / (2 * np.pi)).integrate().values
