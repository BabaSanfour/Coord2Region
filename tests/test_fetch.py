import warnings
import os
import zipfile
import tarfile
import gzip
import requests
import urllib.error
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import pytest
import nibabel as nib

from coord2region import AtlasFetcher, AtlasFileHandler
from coord2region.utils import fetch_labels, pack_vol_output, pack_surf_output


# --------------------------
# Tests for fetching atlases
# --------------------------

import importlib

def _locate_fetch_fsaverage():
    import mne
    # 1) Public import
    try:
        from mne.datasets import fetch_fsaverage  # noqa: F401
        return fetch_fsaverage
    except Exception:
        pass
    # 2) Attribute on mne.datasets (lazy proxy safe)
    ds = getattr(mne, "datasets", None)
    fn = getattr(ds, "fetch_fsaverage", None) if ds is not None else None
    if callable(fn):
        return fn
    # 3) Only if it's a real package, try private module
    if hasattr(ds, "__path__"):
        try:
            mod = importlib.import_module("mne.datasets._fsaverage.base")
            fn = getattr(mod, "fetch_fsaverage", None)
            if callable(fn):
                return fn
        except Exception:
            pass
    return None

def test_mne_has_fetch_fsaverage():
    fn = _locate_fetch_fsaverage()
    assert callable(fn), "mne.datasets.fetch_fsaverage is not available in this environment"

# List of Nilearn atlases to test (volumetric)
NILEARN_ATLASES = [
    "yeo", "harvard-oxford", "juelich", "schaefer",
    "brodmann", "aal", "destrieux", "pauli", "basc"
]
@pytest.mark.requires_network
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
    try:
        atlas = af.fetch_atlas(atlas_name)
    except (requests.exceptions.HTTPError, urllib.error.HTTPError) as exc:
        pytest.skip(f"Remote server error while fetching atlas '{atlas_name}': {exc}")

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

@pytest.mark.requires_network
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
    "brodmann", "pals_b12_lobes", "pals_b12_orbitofrontal",
    "pals_b12_visuotopic", "aparc_sub", "aparc", "aparc.a2009s",
    "aparc.a2005s", "oasis.chubs", "yeo2011"
]

@pytest.mark.requires_network
@pytest.mark.parametrize("atlas_name", MNE_ATLASES)
def test_fetch_mne_atlases(atlas_name):
    """Test fetching of MNE-based atlases using AtlasFetcher."""
    af = AtlasFetcher()
    try:
        atlas = af.fetch_atlas(atlas_name)
    except (requests.exceptions.HTTPError, urllib.error.HTTPError) as exc:
        pytest.skip(f"Remote server error while fetching MNE atlas '{atlas_name}': {exc}")

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

def test_fetch_atlas_with_unknown_name():
    """Unknown atlas names should raise ValueError."""
    af = AtlasFetcher()
    with pytest.raises(ValueError):
        af.fetch_atlas("unknown_atlas")

def test_save_and_load_object(tmp_path):
    """Test that AtlasFileHandler can save and load an object."""
    data_dir = str(tmp_path / "coord2region")
    handler = AtlasFileHandler(data_dir=data_dir)
    test_obj = {"a": 1, "b": [1, 2, 3]}
    filename = "test_obj.pkl"
    handler.save(test_obj, filename)
    loaded_obj = handler.load(filename)
    assert loaded_obj == test_obj, "Loaded object does not match the saved object."

def test_pack_vol_output_with_nifti():
    """Test pack_vol_output with a dummy Nifti1Image."""
    data = np.random.rand(10, 10, 10).astype(np.float32)
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    output = pack_vol_output(img)
    np.testing.assert_allclose(output["vol"], data, err_msg="Volume data mismatch in pack_vol_output.")
    np.testing.assert_allclose(output["hdr"], affine, err_msg="Affine matrix mismatch in pack_vol_output.")

def test_pack_vol_output_with_nii_file(tmp_path):
    """pack_vol_output should load data from a .nii.gz file path."""
    data = np.random.rand(4, 4, 4).astype(np.float32)
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    file_path = tmp_path / "dummy.nii.gz"
    nib.save(img, str(file_path))
    output = pack_vol_output(str(file_path))
    np.testing.assert_allclose(output["vol"], data)
    np.testing.assert_allclose(output["hdr"], affine)

def test_pack_vol_output_with_npz(tmp_path):
    """Test pack_vol_output using a dummy .npz file."""
    data = np.random.rand(10, 10, 10).astype(np.float32)
    affine = np.eye(4)
    npz_file = tmp_path / "dummy_atlas.npz"
    np.savez(npz_file, vol=data, hdr=affine)
    output = pack_vol_output(str(npz_file))
    np.testing.assert_allclose(output["vol"], data, err_msg="Volume data mismatch in pack_vol_output with npz.")
    np.testing.assert_allclose(output["hdr"], affine, err_msg="Affine matrix mismatch in pack_vol_output with npz.")

def test_pack_vol_output_with_unsupported_extension(tmp_path):
    """Ensure ValueError is raised for unsupported file formats."""
    invalid_file = tmp_path / "invalid.txt"
    invalid_file.write_text("dummy content")
    with pytest.raises(ValueError):
        pack_vol_output(str(invalid_file))

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
    
    for key in ["vol", "hdr", "labels", "indexes", "regions"]:
        assert key in output, f"Key '{key}' missing in output of pack_surf_output."
    
    # For surface outputs, hdr is expected to be None.
    assert output["hdr"] is None, "Expected hdr to be None in pack_surf_output output."
    assert len(output["labels"]) > 0, "Labels list is empty in pack_surf_output output."
    assert output["indexes"].size > 0, "Indexes array is empty in pack_surf_output output."
    assert isinstance(output["regions"], dict)
    assert "Label1" in output["regions"]
    assert output["regions"]["Label1"].size > 0

def test_pack_surf_output_fetcher_retry(monkeypatch):
    """Ensure fetcher retry path is executed when the first call fails."""

    class DummyLabel:
        def __init__(self, name, hemi, vertices):
            self.name = name
            self.hemi = hemi
            self.vertices = vertices

    dummy_labels = [
        DummyLabel("Label1", "lh", np.array([0, 2, 4])),
        DummyLabel("Label2", "rh", np.array([1, 3, 5])),
    ]

    def dummy_read_labels_from_annot(subject, atlas_name, subjects_dir, **kwargs):
        return dummy_labels

    def dummy_setup_source_space(subject, spacing, subjects_dir, add_dist):
        return [
            {"vertno": np.array([0, 1, 2])},
            {"vertno": np.array([3, 4, 5])},
        ]

    monkeypatch.setattr("mne.read_labels_from_annot", dummy_read_labels_from_annot)
    monkeypatch.setattr("mne.setup_source_space", dummy_setup_source_space)

    calls = []

    def dummy_fetcher(*, subject=None, subjects_dir=None, **kwargs):
        calls.append(subject)
        if subject is not None:
            raise RuntimeError("fail once")
        return None

    output = pack_surf_output(
        "dummy_atlas", fetcher=dummy_fetcher, subject="subj", subjects_dir="dir"
    )

    assert calls == ["subj", None]
    for key in ["vol", "hdr", "labels", "indexes", "regions"]:
        assert key in output
    assert output["hdr"] is None
    assert len(output["labels"]) > 0
    assert output["indexes"].size > 0
    assert isinstance(output["regions"], dict)

def test_fetch_labels_with_list():
    labels = ["Label1", "Label2", "Label3"]
    assert fetch_labels(labels) == labels

def test_fetch_labels_with_valid_xml(tmp_path):
    # Create a dummy XML file with the expected structure.
    xml_content = """
    <root>
      <data>
        <label><name>LabelA</name></label>
        <label><name>LabelB</name></label>
      </data>
    </root>
    """
    xml_file = tmp_path / "labels.xml"
    xml_file.write_text(xml_content.strip())
    expected = ["LabelA", "LabelB"]
    # Pass the file path (as string) to fetch_labels.
    result = fetch_labels(str(xml_file))
    assert result == expected

def test_fetch_labels_with_invalid_xml(tmp_path):
    # Create an XML file missing the 'data' element.
    xml_content = """
    <root>
      <info>
        <label><name>LabelX</name></label>
      </info>
    </root>
    """
    xml_file = tmp_path / "invalid.xml"
    xml_file.write_text(xml_content.strip())
    with pytest.raises(ValueError):
        fetch_labels(str(xml_file))

def test_fetch_labels_with_partial_xml(tmp_path):
    """Ensure labels are collected even if some entries are malformed."""
    xml_content = """
    <root>
      <data>
        <label></label>
        <label><name>LabelB</name></label>
      </data>
    </root>
    """
    xml_file = tmp_path / "partial.xml"
    xml_file.write_text(xml_content.strip())
    result = fetch_labels(str(xml_file))
    assert result == ["LabelB"]

def test_fetch_labels_with_no_label_nodes(tmp_path):
    """XML with <data> but no <label> entries should raise ValueError."""
    xml_content = """
    <root>
      <data></data>
    </root>
    """
    xml_file = tmp_path / "empty.xml"
    xml_file.write_text(xml_content.strip())
    with pytest.raises(ValueError):
        fetch_labels(str(xml_file))

def test_fetch_labels_with_invalid_type():
    """Passing a non-list/non-string should raise ValueError."""
    with pytest.raises(ValueError):
        fetch_labels(123)


# ------------------------------------------------------------------
# Tests for fetch_from_local
# ------------------------------------------------------------------

def create_dummy_npz(file_path, vol_shape=(5, 5, 5)):
    # Create a dummy npz file that pack_vol_output can read.
    vol = np.random.rand(*vol_shape).astype(np.float32)
    hdr = np.eye(4)
    np.savez(file_path, vol=vol, hdr=hdr)
    return vol, hdr

def create_dummy_xml(file_path, labels_list):
    # Create a dummy XML file that contains the given labels.
    root = ET.Element("root")
    data = ET.SubElement(root, "data")
    for lab in labels_list:
        label_elem = ET.SubElement(data, "label")
        name_elem = ET.SubElement(label_elem, "name")
        name_elem.text = lab
    tree = ET.ElementTree(root)
    tree.write(file_path)
    
@pytest.fixture
def dummy_atlas_dir(tmp_path):
    # Create a temporary directory with a dummy atlas file and a dummy labels file.
    atlas_file_name = "dummy_atlas.npz"
    labels_file_name = "dummy_labels.xml"
    
    atlas_path = tmp_path / atlas_file_name
    vol, hdr = create_dummy_npz(atlas_path)
    
    labels_path = tmp_path / labels_file_name
    label_list = ["Region1", "Region2", "Region3"]
    create_dummy_xml(labels_path, label_list)
    
    return str(tmp_path), atlas_file_name, labels_file_name, vol, hdr, label_list

def test_fetch_from_local_with_xml_labels(dummy_atlas_dir):
    atlas_dir, atlas_file, labels_file, vol, hdr, expected_labels = dummy_atlas_dir
    handler = AtlasFileHandler(data_dir=atlas_dir)
    output = handler.fetch_from_local(atlas_file, atlas_dir, labels_file)
    
    # Check that the volume and header match the dummy npz.
    np.testing.assert_allclose(output["vol"], vol)
    np.testing.assert_allclose(output["hdr"], hdr)
    # Check that the labels were correctly extracted from the XML file.
    assert output["labels"] == expected_labels

def test_fetch_from_local_with_list_labels(dummy_atlas_dir):
    atlas_dir, atlas_file, _, vol, hdr, _ = dummy_atlas_dir
    handler = AtlasFileHandler(data_dir=atlas_dir)
    label_list = ["DirectLabel1", "DirectLabel2"]
    output = handler.fetch_from_local(atlas_file, atlas_dir, label_list)
    
    np.testing.assert_allclose(output["vol"], vol)
    np.testing.assert_allclose(output["hdr"], hdr)
    assert output["labels"] == label_list

def test_fetch_from_local_atlas_not_found(tmp_path):
    handler = AtlasFileHandler(data_dir=str(tmp_path))
    # Atlas file does not exist.
    with pytest.raises(FileNotFoundError):
        handler.fetch_from_local("nonexistent.npz", str(tmp_path), [])

def test_fetch_from_local_labels_not_found(dummy_atlas_dir, tmp_path):
    atlas_dir, atlas_file, _, _, _, _ = dummy_atlas_dir
    handler = AtlasFileHandler(data_dir=atlas_dir)
    # Provide a labels filename that does not exist.
    with pytest.raises(FileNotFoundError):
        handler.fetch_from_local(atlas_file, atlas_dir, "nonexistent_labels.xml")

def test_fetch_atlas_local_absolute(tmp_path):
    """AtlasFetcher should load an atlas from an absolute path."""
    atlas_path = tmp_path / "abs_atlas.npz"
    vol, hdr = create_dummy_npz(atlas_path)
    af = AtlasFetcher(data_dir=str(tmp_path / "data"))
    atlas = af.fetch_atlas("local", atlas_file=str(atlas_path), labels=["A", "B"])
    np.testing.assert_allclose(atlas["vol"], vol)
    np.testing.assert_allclose(atlas["hdr"], hdr)
    assert atlas["labels"] == ["A", "B"]


def test_fetch_atlas_local_relative_to_data_dir(tmp_path):
    """AtlasFetcher should load an atlas with path relative to data_dir."""
    data_dir = tmp_path / "coord2region"
    data_dir.mkdir()
    atlas_file = data_dir / "rel_atlas.npz"
    vol, hdr = create_dummy_npz(atlas_file)
    af = AtlasFetcher(data_dir=str(data_dir))
    atlas = af.fetch_atlas("local", atlas_file="rel_atlas.npz", labels=["X"]) 
    np.testing.assert_allclose(atlas["vol"], vol)
    np.testing.assert_allclose(atlas["hdr"], hdr)
    assert atlas["labels"] == ["X"]

# ------------------------------------------------------------------
# Tests for fetch_from_url
# ------------------------------------------------------------------
def test_fetch_atlas_url_branch(tmp_path, monkeypatch):
    data = np.random.rand(3, 3, 3).astype(np.float32)
    affine = np.eye(4)
    atlas_file = tmp_path / "dummy.npz"
    np.savez(atlas_file, vol=data, hdr=affine)

    calls = []

    def fake_fetch_from_url(self, atlas_url, **kwargs):
        calls.append(atlas_url)
        return str(tmp_path)

    monkeypatch.setattr(AtlasFileHandler, "fetch_from_url", fake_fetch_from_url)
    af = AtlasFetcher(data_dir=str(tmp_path))
    atlas = af.fetch_atlas(
        "custom",
        atlas_url="http://example.com/dummy.npz",
        atlas_file="dummy.npz",
        labels=["A"],
    )
    assert calls == ["http://example.com/dummy.npz"]
    np.testing.assert_allclose(atlas["vol"], data)
    np.testing.assert_allclose(atlas["hdr"], affine)
    assert atlas["labels"] == ["A"]


def test_fetch_atlas_with_atlas_image(tmp_path):
    data = np.random.rand(2, 2, 2).astype(np.float32)
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    af = AtlasFetcher(data_dir=str(tmp_path))
    atlas = af.fetch_atlas("img_atlas", atlas_image=img, labels=["L1", "L2"])
    np.testing.assert_allclose(atlas["vol"], data)
    np.testing.assert_allclose(atlas["hdr"], affine)
    assert atlas["labels"] == ["L1", "L2"]


def test_fetch_atlas_fallback(monkeypatch, tmp_path):
    calls = {"n": 0, "m": 0}

    def fail_nilearn(self, key, fetcher_nilearn, **kwargs):
        calls["n"] += 1
        raise RuntimeError("nilearn fail")

    def success_mne(self, key, fetcher_mne, **kwargs):
        calls["m"] += 1
        return {"vol": np.array([1]), "hdr": None, "labels": ["ok"], "description": {}}

    monkeypatch.setattr(AtlasFetcher, "_fetch_atlas_nilearn", fail_nilearn)
    monkeypatch.setattr(AtlasFetcher, "_fetch_atlas_mne", success_mne)

    af = AtlasFetcher(data_dir=str(tmp_path))
    atlas = af.fetch_atlas("brodmann", prefer="nilearn")
    assert calls == {"n": 1, "m": 1}
    np.testing.assert_array_equal(atlas["vol"], np.array([1]))
    assert atlas["labels"] == ["ok"]

class DummyResponse:
    def __init__(self, content):
        self.content = content
    def raise_for_status(self):
        pass
    def iter_content(self, chunk_size=8192):
        # Yield content in one chunk.
        yield self.content
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def dummy_requests_get(*args, **kwargs):
    # This dummy function simulates a successful download with fixed content.
    content = b"dummy atlas file content"
    return DummyResponse(content)

def test_fetch_from_url(tmp_path, monkeypatch):
    handler = AtlasFileHandler(data_dir=str(tmp_path))
    
    # Monkeypatch requests.get in the file_handler module.
    monkeypatch.setattr("requests.get", dummy_requests_get)
    
    # Use a dummy URL with a file extension that does not trigger decompression.
    atlas_url = "http://example.com/dummy_atlas.npz"
    decompressed_path = handler.fetch_from_url(atlas_url)
    
    # Check that the returned path exists.
    assert os.path.exists(decompressed_path)
    # Check that the file content matches the dummy content.
    with open(decompressed_path, "rb") as f:
        file_content = f.read()
    assert file_content == b"dummy atlas file content"
    
    # Calling fetch_from_url again should skip the download.
    decompressed_path2 = handler.fetch_from_url(atlas_url)
    assert decompressed_path == decompressed_path2

def test_fetch_from_url_with_zip(tmp_path, monkeypatch):
    """
    Test fetch_from_url with a dummy zip file.
    This simulates a compressed file download.
    """
    handler = AtlasFileHandler(data_dir=str(tmp_path))
    
    # Create a dummy zip file in a temporary directory.
    dummy_zip_name = "dummy.zip"
    dummy_zip_path = tmp_path / dummy_zip_name
    dummy_file_name = "extracted.txt"
    dummy_content = b"extracted content"
    
    with zipfile.ZipFile(dummy_zip_path, "w") as zipf:
        temp_file = tmp_path / dummy_file_name
        temp_file.write_bytes(dummy_content)
        zipf.write(temp_file, arcname=dummy_file_name)
    # Read the dummy zip file content.
    with open(dummy_zip_path, "rb") as f:
        zip_bytes = f.read()
    
    # Dummy requests.get will return the zip bytes.
    def dummy_get_zip(*args, **kwargs):
        return DummyResponse(zip_bytes)
    
    monkeypatch.setattr("requests.get", dummy_get_zip)
    
    atlas_url = f"http://example.com/{dummy_zip_name}"
    decompressed_path = handler.fetch_from_url(atlas_url)
    
    # The function should return a directory after extraction.
    assert os.path.isdir(decompressed_path)
    # Check that the extracted file exists and has the expected content.
    extracted_file = os.path.join(decompressed_path, dummy_file_name)
    assert os.path.exists(extracted_file)
    with open(extracted_file, "rb") as f:
        extracted_content = f.read()
    assert extracted_content == dummy_content

def test_fetch_from_url_with_tar_gz(tmp_path, monkeypatch):
    """Test fetch_from_url with a dummy .tar.gz archive."""
    handler = AtlasFileHandler(data_dir=str(tmp_path))

    dummy_tar_name = "dummy.tar.gz"
    dummy_file_name = "extracted.txt"
    dummy_content = b"tar gz content"

    dummy_tar_path = tmp_path / dummy_tar_name
    with tarfile.open(dummy_tar_path, "w:gz") as tarf:
        temp_file = tmp_path / dummy_file_name
        temp_file.write_bytes(dummy_content)
        tarf.add(temp_file, arcname=dummy_file_name)
    tar_bytes = dummy_tar_path.read_bytes()

    def dummy_get_tar(*args, **kwargs):
        return DummyResponse(tar_bytes)

    monkeypatch.setattr("requests.get", dummy_get_tar)

    atlas_url = f"http://example.com/{dummy_tar_name}"
    decompressed_path = handler.fetch_from_url(atlas_url)

    assert os.path.isdir(decompressed_path)
    extracted_file = os.path.join(decompressed_path, dummy_file_name)
    assert os.path.exists(extracted_file)
    with open(extracted_file, "rb") as f:
        extracted_content = f.read()
    assert extracted_content == dummy_content


def test_fetch_from_url_with_gz(tmp_path, monkeypatch):
    """Test fetch_from_url with a dummy .gz file."""
    handler = AtlasFileHandler(data_dir=str(tmp_path))

    dummy_gz_name = "dummy.txt.gz"
    dummy_content = b"gzip content"

    dummy_gz_path = tmp_path / dummy_gz_name
    with gzip.open(dummy_gz_path, "wb") as gz:
        gz.write(dummy_content)
    gz_bytes = dummy_gz_path.read_bytes()

    def dummy_get_gz(*args, **kwargs):
        return DummyResponse(gz_bytes)

    monkeypatch.setattr("requests.get", dummy_get_gz)

    atlas_url = f"http://example.com/{dummy_gz_name}"
    decompressed_path = handler.fetch_from_url(atlas_url)

    assert os.path.isfile(decompressed_path)
    with open(decompressed_path, "rb") as f:
        extracted_content = f.read()
    assert extracted_content == dummy_content

def test_fetch_from_url_download_failure(tmp_path, monkeypatch):
    """RuntimeError should be raised if the download fails."""
    handler = AtlasFileHandler(data_dir=str(tmp_path))

    def failing_get(*args, **kwargs):
        raise requests.exceptions.RequestException("fail")

    monkeypatch.setattr("requests.get", failing_get)

    atlas_url = "http://example.com/bad_atlas.nii.gz"
    with pytest.raises(RuntimeError):
        handler.fetch_from_url(atlas_url)

def test_custom_subjects_dir_preserved(tmp_path, monkeypatch):
    """Ensure that a provided subjects_dir is not overridden."""
    import mne

    monkeypatch.setattr(mne, "get_config", lambda key, default=None: "/not/used")
    custom_dir = tmp_path / "subjects"
    handler = AtlasFileHandler(subjects_dir=str(custom_dir))
    assert handler.subjects_dir == str(custom_dir)
