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
from typing import Protocol
from rapidity.core import Grid1D, Field
