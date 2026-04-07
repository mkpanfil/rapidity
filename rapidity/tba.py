"""
Thermodynamic Bethe Ansatz solver.

This module provides :class:`TBAState` for single-species models and
:class:`StringTBAState` for models with string hypothesis. Both classes
provide methods for computing thermodynamic quantities from the filling
function.

The TBA equation for a single-species model reads:

.. math::

    \\epsilon(\\theta) = \\epsilon_0(\\theta) -
    \\int d\\theta'\\, K(\\theta - \\theta')
    \\log(1 + e^{-\\epsilon(\\theta')})

For string models the simplified equations couple only neighbouring
string species:

.. math::

    \\epsilon_n(\\theta) = \\epsilon_n^0(\\theta) -
    a_1 * \\log\\left[(1 + e^{-\\epsilon_{n-1}(\\theta)})
    (1 + e^{-\\epsilon_{n+1}(\\theta)})\\right]
"""

import numpy as np
from dataclasses import dataclass
from rapidity.core import Grid1D, Field
from rapidity.models import Model, StringModel


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


def _solve_string_tba(
    driving: list[Field],
    convolve_a1: callable,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> list[Field]:
    """Solve the simplified string TBA equations by fixed-point iteration.

    The equations couple only neighbouring string species:

    .. math::

        \\epsilon_n = \\epsilon_n^0 + a_1 * \\log
        \\left[(1 + e^{-\\epsilon_{n-1}})(1 + e^{-\\epsilon_{n+1}})\\right]

    Note the plus sign, in contrast to the single-species TBA where
    the convolution term enters with a minus sign.

    Boundary conditions: :math:`\\epsilon_0 = \\epsilon_{N_{max}+1} = +\\infty`,
    so the boundary log terms vanish.

    Parameters
    ----------
    driving : list[Field]
        Driving terms for each string species.
    convolve_a1 : callable
        Function that convolves a Field with the basic kernel a_1.
    tol : float, optional
        Convergence tolerance. Default is 1e-10.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.

    Returns
    -------
    list[Field]
        The pseudoenergies for each string species.

    Raises
    ------
    RuntimeError
        If the iteration does not converge within max_iter iterations.
    """
    n_max = len(driving)
    zero = Field.from_function(lambda t: np.zeros_like(t), driving[0].grids)
    epsilon = [Field(d.values.copy(), d.grids) for d in driving]

    for i in range(max_iter):
        epsilon_new = []
        for n in range(n_max):
            log_left = (
                epsilon[n - 1].apply(lambda x: np.logaddexp(0, -x)) if n > 0 else zero
            )
            log_right = (
                epsilon[n + 1].apply(lambda x: np.logaddexp(0, -x))
                if n < n_max - 1
                else zero
            )
            log_sum = log_left + log_right
            # note: plus sign here, contrast with single-species TBA which has minus
            epsilon_new.append(driving[n] + convolve_a1(log_sum))

        errors = [
            np.max(np.abs((epsilon_new[n] - epsilon[n]).values)) for n in range(n_max)
        ]
        if max(errors) < tol:
            return epsilon_new
        epsilon = epsilon_new

    raise RuntimeError(
        f"String TBA iteration did not converge after {max_iter} iterations."
    )


def _dress_string(
    h: list[Field],
    filling: list[Field],
    convolve_a1: callable,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> list[Field]:
    """Solve the string dressing equations by fixed-point iteration.

    The dressing equation couples only neighbouring string species:

    .. math::

        h_n^{dr} = h_n + a_1 * (n_{n-1} h_{n-1}^{dr} + n_{n+1} h_{n+1}^{dr})

    Note the plus sign, consistent with the string TBA equations.

    Parameters
    ----------
    h : list[Field]
        Bare quantities to dress, one per string species.
    filling : list[Field]
        Filling functions, one per string species.
    convolve_a1 : callable
        Function that convolves a Field with the basic kernel a_1.
    tol : float, optional
        Convergence tolerance. Default is 1e-10.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.

    Returns
    -------
    list[Field]
        Dressed quantities, one per string species.

    Raises
    ------
    RuntimeError
        If the iteration does not converge within max_iter iterations.
    """
    n_max = len(h)
    zero = Field.from_function(lambda t: np.zeros_like(t), h[0].grids)
    h_dr = [Field(hi.values.copy(), hi.grids) for hi in h]

    for _ in range(max_iter):
        h_dr_new = []
        for n in range(n_max):
            # note: plus sign, consistent with string TBA equations
            left = convolve_a1(filling[n - 1] * h_dr[n - 1]) if n > 0 else zero
            right = convolve_a1(filling[n + 1] * h_dr[n + 1]) if n < n_max - 1 else zero
            h_dr_new.append(h[n] + left + right)

        errors = [np.max(np.abs((h_dr_new[n] - h_dr[n]).values)) for n in range(n_max)]
        if max(errors) < tol:
            return h_dr_new
        h_dr = h_dr_new

    raise RuntimeError(
        f"String dressing iteration did not converge after {max_iter} iterations."
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
        # filling = Field(1 / (1 + np.exp(epsilon.values)), [grid])
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


@dataclass
class StringTBAState:
    """Thermodynamic state for integrable models with string hypothesis.

    The state is characterized by a list of filling functions n_s(theta),
    one per string species. It always carries the model and grid alongside
    the filling functions.

    Parameters
    ----------
    model : StringModel
        The integrable model with string hypothesis.
    grid : Grid1D
        The shared rapidity grid for all string species.
    filling : list[Field]
        Filling functions n_s(theta), one per string species.

    Examples
    --------
    >>> from rapidity.models import XXXSpinChain
    >>> from rapidity.core import Grid1D
    >>> model = XXXSpinChain(S=0.5, n_max=10)
    >>> grid = Grid1D.gauss_legendre(-10, 10, 200, "theta")
    >>> state = StringTBAState.from_betas(model, grid, betas={1: 0.1, 0: -0.5})
    """

    model: StringModel
    grid: Grid1D
    filling: list[Field]

    @classmethod
    def from_betas(
        cls,
        model: StringModel,
        grid: Grid1D,
        betas: dict[int, float],
        tol: float = 1e-10,
        max_iter: int = 1000,
    ) -> "StringTBAState":
        """Construct state by solving the string TBA equations.

        Parameters
        ----------
        model : StringModel
            The integrable model.
        grid : Grid1D
            The rapidity grid.
        betas : dict[int, float]
            Chemical potentials keyed by charge order.
        tol : float, optional
            Convergence tolerance. Default is 1e-10.
        max_iter : int, optional
            Maximum number of iterations. Default is 1000.

        Returns
        -------
        StringTBAState
            The thermodynamic state.
        """
        _check_grid(model, grid)
        driving = model.driving(grid, betas)
        # a1_kernel = model.kernel_a_n(1, grid)
        epsilon = _solve_string_tba(driving, model.convolve_a1, tol, max_iter)
        filling = [e.apply(lambda x: 1 / (1 + np.exp(x))) for e in epsilon]
        return cls(model, grid, filling)

    @classmethod
    def from_filling(
        cls, model: StringModel, grid: Grid1D, filling: list[Field]
    ) -> "StringTBAState":
        """Construct state directly from filling functions.

        Parameters
        ----------
        model : StringModel
            The integrable model.
        grid : Grid1D
            The rapidity grid.
        filling : list[Field]
            Filling functions, one per string species.

        Returns
        -------
        StringTBAState
            The thermodynamic state.
        """
        _check_grid(model, grid)
        return cls(model, grid, filling)

    def dress(self, h: list[Field]) -> list[Field]:
        """Compute dressed quantities using the simplified dressing equation.

        The dressing equation couples only neighbouring string species:

        .. math::

            h_n^{dr} = h_n + a_1 * (n_{n-1} h_{n-1}^{dr} +
            n_{n+1} h_{n+1}^{dr})

        Parameters
        ----------
        h : list[Field]
            Bare quantities to dress, one per string species.

        Returns
        -------
        list[Field]
            Dressed quantities, one per string species.
        """
        # a1_kernel = self.model.kernel_a_n(1, self.grid)
        return _dress_string(h, self.filling, self.model.convolve_a1)

    def rho_s(self) -> list[Field]:
        """Compute state densities rho_s^n(theta) for all string species.

        Returns
        -------
        list[Field]
            State densities, one per string species.
        """
        a = [
            self.model.bare_state_density(n + 1, self.grid)
            for n in range(self.model.n_max)
        ]
        return self.dress(a)

    def rho_p(self) -> list[Field]:
        """Compute particle densities rho_p^n(theta) for all string species.

        Returns
        -------
        list[Field]
            Particle densities, one per string species.
        """
        return [n * rho_s for n, rho_s in zip(self.filling, self.rho_s())]

    def v_eff(self) -> list[Field]:
        """Compute effective velocities v^eff_n(theta) for all string species.

        Returns
        -------
        list[Field]
            Effective velocities, one per string species.
        """
        label = self.model.rapidity_label
        momentum = [
            self.model.charge(1, n + 1, self.grid) for n in range(self.model.n_max)
        ]
        energy = [
            self.model.charge(2, n + 1, self.grid) for n in range(self.model.n_max)
        ]
        de_dr = self.dress([e.derivative(label) for e in energy])
        dp_dr = self.dress([p.derivative(label) for p in momentum])
        return [de / dp for de, dp in zip(de_dr, dp_dr)]
