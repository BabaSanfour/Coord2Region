import numpy as np
import pytest

from coord2region.coord2region import _get_numeric_hemi, AtlasMapper


@pytest.mark.unit
def test_get_numeric_hemi_variants():
    assert _get_numeric_hemi("L") == 0
    assert _get_numeric_hemi("left") == 0
    assert _get_numeric_hemi("R") == 1
    assert _get_numeric_hemi("right") == 1
    assert _get_numeric_hemi(0) == 0
    assert _get_numeric_hemi(1) == 1
    assert _get_numeric_hemi(None) is None
    with pytest.raises(ValueError):
        _get_numeric_hemi("center")


@pytest.mark.unit
def test_atlasmapper_invalid_hdr_shape():
    vol = np.zeros((2, 2, 2))
    bad_hdr = np.eye(3)
    with pytest.raises(ValueError):
        AtlasMapper("bad", vol, bad_hdr)
