"""
Core data structures for the rapidity package.

This subpackage defines the two fundamental building blocks:

- :class:`~rapidity.core.grid.Grid1D` — a 1D quadrature grid
- :class:`~rapidity.core.field.Field` — a multi-dimensional field on a product of grids

Both are re-exported at this level for convenience:

.. code-block:: python

    from rapidity.core import Grid1D, Field
"""

from rapidity.core.grid import Grid1D
from rapidity.core.field import Field

__all__ = ["Grid1D", "Field"]
