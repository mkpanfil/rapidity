Getting Started
===============

Installation
------------

.. code-block:: bash

    pip install rapidity

Or for development:

.. code-block:: bash

    git clone https://github.com/mkpanfil/rapidity.git
    cd rapidity
    pip install -e .

Quick Example
-------------

Solve the TBA for the Lieb-Liniger model at finite temperature and
compute the particle density:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt 
    from rapidity.core import Grid1D
    from rapidity.models import LiebLiniger
    from rapidity.tba import TBAState
    from rapidity.utils import plot

    # define model and grid
    model = LiebLiniger(c=1.0)
    grid = Grid1D.uniform(-5, 5, 200, "theta")

    # solve TBA at temperature T=0.5 and chemical potential mu=0.5
    T, mu = 0.5, 0.5
    state = TBAState.from_betas(model, grid, betas={2: 1/T, 0: -mu/T})

    # compute particle density
    rho_p = state.rho_p()
    N_L = rho_p.integrate().values
    print(f"Total density N/L = {N_L:.6f}")

    # plot particle density
    plot(rho_p)
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\rho_p(\theta)$")
    plt.title(f"Lieb-Liniger model at T={T}, mu={mu}")
    plt.show()