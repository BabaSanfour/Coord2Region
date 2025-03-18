import pytest
import numpy as np

from coord2region.fetching import AtlasFetcher
from coord2region.coord2region import (
    VolumetricAtlasMapper,
    BatchAtlasMapper,
    MultiAtlasMapper
)

# -------------------------------------------------------------------------
# 1) Yeo Atlas Fixtures
# -------------------------------------------------------------------------
# @pytest.fixture(scope="module")
def yeo_data():
    """
    Fetch the Yeo atlas once (by default, version='thick_17' in your fetcher)
    and return its dictionary: {'vol', 'hdr', 'labels', ...}.
    """
    af = AtlasFetcher(data_dir="coord2region_data")
    yeo_atlas = af.fetch_atlas("yeo")
    return yeo_atlas

# @pytest.fixture(scope="module")
def yeo_mapper(yeo_data):
    """
    Create a VolumetricAtlasMapper for the Yeo atlas data.
    """
    return VolumetricAtlasMapper(
        name="yeo",
        vol=yeo_data["vol"],
        hdr=yeo_data["hdr"],
        labels=yeo_data.get("labels", None)
    )

# -------------------------------------------------------------------------
# 2) Basic Tests for VolumetricAtlasMapper (Yeo)
# -------------------------------------------------------------------------
def test_infer_hemisphere_yeo(yeo_mapper):
    """
    Check if we can infer hemisphere from a typical Yeo label name
    (e.g., "7Networks_Visual_L"). This might be "Unknown" if naming differs.
    """
    example_label = "7Networks_Visual_L"  # Adjust if your fetched labels differ.
    hemi = yeo_mapper.infer_hemisphere(example_label)
    # Typically 'L', 'R', or None.
    assert hemi is None or hemi == 'L', f"Expected 'L' or None, got {hemi}"

def test_region_name_for_index_yeo(yeo_mapper):
    """
    Check that we get a string (possibly 'Unknown') for a given index.
    """
    region_name = yeo_mapper.region_name_for_index(1)
    assert isinstance(region_name, str), "region_name_for_index(1) did not return a string."

def test_region_index_for_name_yeo(yeo_mapper):
    """
    If you know a Yeo label exists, test that we get a valid integer index.
    Adjust the label below if needed.
    """
    # Suppose there's a label "7Networks_Visual_L". If not, change to something that definitely exists in your 'labels'.
    idx_val = yeo_mapper.region_index_for_name("7Networks_Visual_L")
    # Could be an integer or "Unknown"
    assert isinstance(idx_val, (int, str)), f"Expected int or 'Unknown', got {idx_val}"

def test_list_all_regions_yeo(yeo_mapper):
    """
    Ensure that we have a non-empty list of region names.
    """
    regions = yeo_mapper.list_all_regions()
    assert isinstance(regions, list), "list_all_regions() should return a list."
    assert len(regions) > 0, "No regions found in Yeo atlas labels."

def test_mni_to_voxel_yeo(yeo_mapper):
    """
    Convert a typical MNI coordinate to voxel indices. 
    We check that it returns 3 integers.
    """
    voxel_idx = yeo_mapper.mni_to_voxel([0., 0., 0.])
    assert len(voxel_idx) == 3, "mni_to_voxel did not return 3D indices."
    for v in voxel_idx:
        assert isinstance(v, int), f"{v} is not an integer voxel index."

def test_mni_to_region_name_yeo(yeo_mapper):
    """
    Convert a coordinate to a Yeo region name (likely 'Unknown' if outside).
    """
    region_name = yeo_mapper.mni_to_region_name([0., 0., 0.])
    assert isinstance(region_name, str), "mni_to_region_name did not return a string."

def test_voxel_to_mni_yeo(yeo_mapper):
    """
    Check the voxel->MNI transform returns a shape-(3,) array for a single voxel.
    """
    coords = yeo_mapper.voxel_to_mni([10, 10, 10])
    assert coords.shape == (3,), f"Expected shape (3,), got {coords.shape}."

def test_region_index_to_mni_yeo(yeo_mapper):
    """
    For a known region index, ensure we get an Nx3 array of coordinates. 
    Might be empty if that index doesn't exist in Yeo. 
    """
    arr = yeo_mapper.region_index_to_mni(1)  # or pick another plausible index
    # Should be 2D, shape (N, 3). Could be N=0 if region=1 doesn't exist.
    assert arr.ndim == 2 and arr.shape[1] == 3, f"Expected Nx3, got {arr.shape}."

def test_region_name_to_mni_yeo(yeo_mapper):
    """
    For a known region name, ensure we get an Nx3 array of coords. Could be empty if doesn't exist.
    """
    arr = yeo_mapper.region_name_to_mni("7Networks_Visual_L")
    # Might be empty if label not recognized. Should still be (N, 3).
    assert arr.ndim == 2 and arr.shape[1] == 3, f"Expected Nx3, got {arr.shape}."

# -------------------------------------------------------------------------
# 3) Tests for BatchAtlasMapper (Yeo)
# -------------------------------------------------------------------------
@pytest.fixture(scope="module")
def yeo_vectorized_mapper(yeo_mapper):
    return BatchAtlasMapper(yeo_mapper)

def test_batch_mni_to_region_name_yeo(yeo_vectorized_mapper, yeo_mapper):
    """
    Example batch test: gather a few coordinates from region_index_to_mni 
    (for indexes 1..5?), then convert them all back to region names in a batch.
    """
    test_coords = []
    # We'll just check indexes 1..5. Adjust as needed if Yeo has fewer indexes.
    for idx_val in range(1, 6):
        arr = yeo_mapper.region_index_to_mni(idx_val)
        if arr.shape[0] > 0:
            test_coords.append(arr[0])  # first voxel
    
    if not test_coords:
        pytest.skip("No valid coords found for indexes 1..5 in Yeo atlas. Cannot batch test.")
    
    names = yeo_vectorized_mapper.batch_mni_to_region_name(test_coords)
    assert len(names) == len(test_coords)
    for nm in names:
        assert isinstance(nm, str), "Batch MNI->region yielded non-string."

# -------------------------------------------------------------------------
# 4) MultiAtlasMapper Test (Yeo Only)
# -------------------------------------------------------------------------
def test_multiatlas_api_yeo():
    """
    Demonstrate using MultiAtlasMapper with just the 'yeo' atlas.
    """
    c2r = MultiAtlasMapper(
        data_dir="coord2region_data",
        atlases={"yeo": {}}
    )
    coords = [[0., 0., 0.], [10., 20., 30.]]
    result_dict = c2r.batch_mni_to_region_names(coords)
    # result_dict should have a single key: "yeo"
    assert "yeo" in result_dict
    assert len(result_dict["yeo"]) == 2
    # Each entry is a string
    for name_out in result_dict["yeo"]:
        assert isinstance(name_out, str), f"Expected string region name, got {type(name_out)}."

if __name__ == "__main__":
    yeodata=yeo_data()
    yeop = yeo_mapper(yeodata)
    test_infer_hemisphere_yeo(yeo_mapper)
    