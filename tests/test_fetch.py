import warnings

import numpy as np
import pandas as pd
import pytest
import nibabel as nib

from coord2region import AtlasFetcher, AtlasFileHandler
from coord2region.utils import fetch_labels, pack_vol_output, pack_surf_output

# --------------------------
# Tests for fetching atlases
# --------------------------

# List of Nilearn atlases to test (volumetric)
NILEARN_ATLASES = [
    "yeo", "harvard-oxford", "juelich", "schaefer",
    "brodmann", "aal", "destrieux", "pauli", "basc"
]

@pytest.mark.parametrize("atlas_name", NILEARN_ATLASES)
def test_fetch_nilearn_atlases(atlas_name):
    """Test fetching of Nilearn volumetric atlases using AtlasFetcher."""
    af = AtlasFetcher()
    # If certain atlases are known to be unavailable in the current Nilearn version, skip.
    if atlas_name in ["brodmann", "aal"]:
        warnings.warn(
            f"Atlas '{atlas_name}' is not available in the current version of Nilearn. Skipping test."
        )
        pytest.skip(f"Skipping atlas '{atlas_name}'")
    atlas = af.fetch_atlas(atlas_name)
    
    for key in ["vol", "hdr", "labels"]:
        assert key in atlas, f"Key '{key}' missing in atlas '{atlas_name}' output."
        assert atlas[key] is not None, f"Key '{key}' is None in atlas '{atlas_name}' output."
    
    # 'vol' should be a non-empty numpy array.
    assert isinstance(atlas["vol"], np.ndarray), (
        f"'vol' should be a numpy array for atlas '{atlas_name}'."
    )
    assert atlas["vol"].size > 0, f"'vol' is empty for atlas '{atlas_name}'."
    
    # If header is provided, check its type and shape.
    if atlas["hdr"] is not None:
        assert isinstance(atlas["hdr"], np.ndarray), (
            f"'hdr' should be a numpy array for atlas '{atlas_name}' if provided."
        )
        assert atlas["hdr"].shape == (4, 4), (
            f"'hdr' should be a 4x4 numpy array for atlas '{atlas_name}' if provided."
        )
    else:
        warnings.warn(f"'hdr' is None for atlas '{atlas_name}'.")
    
    # 'labels' must be a non-empty list.
    assert isinstance(atlas["labels"], list) and len(atlas["labels"]) > 0, (
        f"Labels should be a non-empty list for atlas '{atlas_name}'."
    )

# List of Nilearn coordinate atlases to test
NILEARN_COORDS = ["dosenbach", "power", "seitzman"]

@pytest.mark.parametrize("atlas_name", NILEARN_COORDS)
def test_fetch_nilearn_coords(atlas_name):
    """Test fetching of Nilearn coordinate atlases using AtlasFetcher."""
    af = AtlasFetcher()
    atlas = af.fetch_atlas(atlas_name)
    
    for key in ["vol", "labels"]:
        assert key in atlas, f"Key '{key}' missing in atlas '{atlas_name}' output."
        assert atlas[key] is not None, f"Key '{key}' is None in atlas '{atlas_name}' output."
    
    # Expect vol to be a pandas DataFrame with columns x, y, z.
    assert isinstance(atlas["vol"], pd.DataFrame), (
        f"'vol' should be a pandas DataFrame for atlas '{atlas_name}'."
    )
    for col in ["x", "y", "z"]:
        assert col in atlas["vol"].columns, (
            f"DataFrame missing '{col}' column for atlas '{atlas_name}'."
        )
    assert atlas["vol"].shape[0] > 0, f"'vol' DataFrame is empty for atlas '{atlas_name}'."
    
    # Check labels is non-empty (list or numpy array).
    assert ((isinstance(atlas["labels"], list) or isinstance(atlas["labels"], np.ndarray))
            and len(atlas["labels"]) > 0), f"Labels are empty for atlas '{atlas_name}'."

# List of MNE atlases to test
MNE_ATLASES = [
    "brodmann", "human-connectum project", "pals_b12_lobes", "pals_b12_orbitofrontal",
    "pals_b12_visuotopic", "aparc_sub", "aparc", "aparc.a2009s",
    "aparc.a2005s", "oasis.chubs", "yeo2011"
]

@pytest.mark.parametrize("atlas_name", MNE_ATLASES)
def test_fetch_mne_atlases(atlas_name):
    """Test fetching of MNE-based atlases using AtlasFetcher."""
    af = AtlasFetcher()
    atlas = af.fetch_atlas(atlas_name)
    
    for key in ["vol", "labels", "indexes"]:
        assert key in atlas, f"Key '{key}' missing in MNE atlas output for atlas '{atlas_name}'."
        assert atlas[key] is not None, f"Key '{key}' is None in MNE atlas output for atlas '{atlas_name}'."
    
    # For surface-based atlases, vol should be a list (left/right hemisphere)
    assert isinstance(atlas["vol"], list), (
        f"'vol' should be a list for MNE-based atlas '{atlas_name}'."
    )
    # Indexes should be a numpy array.
    assert isinstance(atlas["indexes"], np.ndarray), (
        f"'indexes' should be a numpy array for MNE-based atlas '{atlas_name}'."
    )
    
    # Ensure labels is non-empty.
    labels = atlas["labels"] if isinstance(atlas["labels"], list) else atlas["labels"].tolist()
    assert len(labels) > 0, f"Labels are empty for MNE-based atlas '{atlas_name}'."

# -----------------------------------------
# Tests for helper functions and caching
# -----------------------------------------

def test_list_available_atlases():
    """Test that list_available_atlases returns a non-empty list."""
    af = AtlasFetcher()
    atlases = af.list_available_atlases()
    assert isinstance(atlases, list), "list_available_atlases should return a list."
    assert len(atlases) > 0, "list_available_atlases returned an empty list."

def test_save_and_load_object(tmp_path):
    """Test that AtlasFileHandler can save and load an object."""
    data_dir = str(tmp_path / "coord2region")
    handler = AtlasFileHandler(data_dir=data_dir)
    test_obj = {"a": 1, "b": [1, 2, 3]}
    filename = "test_obj.pkl"
    handler.save(test_obj, filename)
    loaded_obj = handler.load(filename)
    assert loaded_obj == test_obj, "Loaded object does not match the saved object."

def test_fetch_labels():
    """Test the fetch_labels helper function."""
    # When provided a list, fetch_labels should return the same list.
    labels_input = ["Label1", "Label2"]
    output = fetch_labels(labels_input)
    assert output == labels_input, "fetch_labels did not return the expected list."
    # When provided a string, it should raise NotImplementedError.
    with pytest.raises(NotImplementedError):
        fetch_labels("dummy_labels.txt")

def test_pack_vol_output_with_nifti():
    """Test pack_vol_output with a dummy Nifti1Image."""
    data = np.random.rand(10, 10, 10).astype(np.float32)
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    output = pack_vol_output(img)
    np.testing.assert_allclose(output["vol"], data, err_msg="Volume data mismatch in pack_vol_output.")
    np.testing.assert_allclose(output["hdr"], affine, err_msg="Affine matrix mismatch in pack_vol_output.")

def test_pack_vol_output_with_npz(tmp_path):
    """Test pack_vol_output using a dummy .npz file."""
    data = np.random.rand(10, 10, 10).astype(np.float32)
    affine = np.eye(4)
    npz_file = tmp_path / "dummy_atlas.npz"
    np.savez(npz_file, vol=data, hdr=affine)
    output = pack_vol_output(str(npz_file))
    np.testing.assert_allclose(output["vol"], data, err_msg="Volume data mismatch in pack_vol_output with npz.")
    np.testing.assert_allclose(output["hdr"], affine, err_msg="Affine matrix mismatch in pack_vol_output with npz.")

def test_pack_surf_output(monkeypatch):
    """
    Test pack_surf_output by monkeypatching mne functions to provide dummy labels
    and source space information.
    """
    # Create dummy label objects with necessary attributes.
    class DummyLabel:
        def __init__(self, name, hemi, vertices):
            self.name = name
            self.hemi = hemi
            self.vertices = vertices

    dummy_labels = [
        DummyLabel("Label1", "lh", np.array([0, 2, 4])),
        DummyLabel("Label2", "rh", np.array([1, 3, 5]))
    ]

    # Monkeypatch mne.read_labels_from_annot to return dummy_labels.
    def dummy_read_labels_from_annot(subject, atlas_name, subjects_dir, **kwargs):
        return dummy_labels

    # Monkeypatch mne.setup_source_space to return dummy source space info.
    def dummy_setup_source_space(subject, spacing, subjects_dir, add_dist):
        return [
            {"vertno": np.array([0, 1, 2, 3])},
            {"vertno": np.array([4, 5, 6, 7])}
        ]

    monkeypatch.setattr("mne.read_labels_from_annot", dummy_read_labels_from_annot)
    monkeypatch.setattr("mne.setup_source_space", dummy_setup_source_space)
    
    # Call pack_surf_output with dummy parameters.
    output = pack_surf_output("dummy_atlas", fetcher=None, subject="dummy", subjects_dir="dummy_dir")
    
    for key in ["vol", "hdr", "labels", "indexes"]:
        assert key in output, f"Key '{key}' missing in output of pack_surf_output."
    
    # For surface outputs, hdr is expected to be None.
    assert output["hdr"] is None, "Expected hdr to be None in pack_surf_output output."
    assert len(output["labels"]) > 0, "Labels list is empty in pack_surf_output output."
    assert output["indexes"].size > 0, "Indexes array is empty in pack_surf_output output."
