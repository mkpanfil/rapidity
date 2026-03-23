import numpy as np
from rapidity.core import Grid1D, Field
from rapidity.fredholm import fredholm_det
from rapidity.utils import sine_kernel


def test_fredholm_det_sine_kernel():
    """Fredholm determinant of the sine kernel matches Bornemann (2010).

    E2(0; 0.1) = det(I - K_sine) on [0, 0.1] = 0.9000272717982592
    Reference: Bornemann (2010), p. 5.
    """
    grid = Grid1D.gauss_legendre(0.0, 0.1, 50, "x")
    kernel = sine_kernel(grid)
    assert np.isclose(fredholm_det(kernel), 0.9000272717982592, atol=1e-10)
