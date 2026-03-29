import pytest
from rapidity.models import LiebLiniger


def test_lieb_liniger_raises_for_nonpositive_c():
    """LiebLiniger raises ValueError for non-positive coupling constant."""
    with pytest.raises(ValueError):
        LiebLiniger(c=0.0)
    with pytest.raises(ValueError):
        LiebLiniger(c=-1.0)
