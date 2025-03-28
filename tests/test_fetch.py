import numpy as np
import pandas as pd
import pytest
from coord2region import AtlasFetcher
import warnings
# List of nilearn atlases to test
NILEARN_ATLASES = ["yeo", "harvard-oxford", "juelich", "schaefer", "brodmann", "aal",
                   'destrieux', 'pauli', 'basc']

@pytest.mark.parametrize("atlas_name", NILEARN_ATLASES)
def test_fetch_nilearn_atlases(atlas_name):
    """Test fetching of nilearn atlases using AtlasFetcher."""
    af = AtlasFetcher()
    if atlas_name in ["brodmann", "aal"]:
        warnings.warn(
            f"Atlas '{atlas_name}' is not available in the current version of nilearn. Skipping test."
        )
        return 
    atlas = af.fetch_atlas(atlas_name)


    for key in ["vol", "hdr", "labels"]:
        assert atlas[key] is not None, f"Key '{key}' missing in atlas '{atlas_name}' output."


    assert isinstance(atlas["vol"], np.ndarray), (
        f"'vol' should be a numpy array for atlas '{atlas_name}'."
    )
    assert atlas["vol"].size > 0, f"'vol' is empty for atlas '{atlas_name}'."
    assert atlas["vol"].ndim == 3, f"'vol' should be a 3D numpy array for atlas '{atlas_name}'."


    if atlas["hdr"] is not None:
        assert isinstance(atlas["hdr"], np.ndarray), (
            f"'hdr' should be a numpy array for atlas '{atlas_name}' if provided."
        )
        assert atlas["hdr"].shape == (4, 4), (
            f"'hdr' should be a 4x4 numpy array for atlas '{atlas_name}' if provided."
        )
    else: 
        Warning(
            f"'hdr' is None for atlas '{atlas_name}'."
        )


    assert isinstance(atlas["labels"], list) and len(atlas["labels"]) > 0, (
        f"Labels should be a non-empty list for atlas '{atlas_name}'."
    )

# List of nilearn coord based atlases to test
NILEARN_COORDS = ["dosenbach", "power", "seitzman",]
@pytest.mark.parametrize("atlas_name", NILEARN_COORDS)
def test_fetch_nilearn_coords(atlas_name):
    """Test fetching of nilearn atlases using AtlasFetcher."""
    af = AtlasFetcher()
    atlas = af.fetch_atlas(atlas_name)

    for key in ["vol", "labels"]:
        assert atlas[key] is not None, f"Key '{key}' missing in atlas '{atlas_name}' output."


    assert isinstance(atlas["vol"], pd.DataFrame), (
        f"'vol' should be a pandas DataFrame for atlas '{atlas_name}'."
    )
    expected_columns = ["x", "y", "z"]
    for col in expected_columns:
        assert col in atlas["vol"].columns, f"DataFrame missing '{col}' column for atlas '{atlas_name}'."
    assert atlas["vol"].shape[0] > 0, f"'vol' DataFrame is empty for atlas '{atlas_name}'."

    assert ((isinstance(atlas["labels"], list) or isinstance(atlas["labels"], np.ndarray)) and 
            len(atlas["labels"]) > 0), f"Labels should be a non-empty list or numpy array for atlas '{atlas_name}'."

def test_fetch_mne_atlases():
    """Test fetching of an MNE-based atlas."""
    af = AtlasFetcher(data_dir="mne_data")
    atlas = af.fetch_atlas("aparc.a2009s")

    # Validate expected keys for MNE atlases; these might differ slightly from nilearn atlases.
    for key in ["vol", "labels", "indexes"]:
        assert key in atlas, f"Key '{key}' missing in MNE atlas output for 'aparc.a2009s'."

    # Validate that 'vol' is a list (as pack_surf_output returns left/right hemisphere arrays)
    assert isinstance(atlas["vol"], list), (
        "'vol' should be a list for MNE-based atlas 'aparc.a2009s'."
    )

    # Validate that 'indexes' is a numpy array
    assert isinstance(atlas["indexes"], np.ndarray), (
        "'indexes' should be a numpy array for MNE-based atlas 'aparc.a2009s'."
    )

    # Validate that the labels list is not empty
    assert isinstance(atlas["labels"], np.ndarray) or isinstance(atlas["labels"], list), (
        "Labels should be a numpy array or list for MNE-based atlas 'aparc.a2009s'."
    )
    labels = atlas["labels"] if isinstance(atlas["labels"], list) else atlas["labels"].tolist()
    assert len(labels) > 0, "Labels are empty for MNE-based atlas 'aparc.a2009s'."