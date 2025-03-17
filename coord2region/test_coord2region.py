import pytest
import numpy as np

from coord2region.fetching import AtlasFetcher
from coord2region.coord2region import (
    AtlasRegionMapper,
    VectorizedAtlasRegionMapper,
    coord2region
)

# ---------------------- Fixtures ---------------------- #
@pytest.fixture(scope="module")
def harvard_atlas():
    """Fetch the Harvard-Oxford atlas as our test fixture."""
    af = AtlasFetcher()
    return af.fetch_atlas("harvard-oxford")


@pytest.fixture(scope="module")
def harvard_mapper(harvard_atlas):
    """Create the AtlasRegionMapper instance for Harvard-Oxford."""
    return AtlasRegionMapper(
        name="harvard-oxford",
        vol=harvard_atlas["vol"],
        hdr=harvard_atlas["hdr"],
        labels=harvard_atlas["labels"]
    )


@pytest.fixture(scope="module")
def harvard_vectorized_mapper(harvard_mapper):
    """Create the VectorizedAtlasRegionMapper from the single AtlasRegionMapper."""
    return VectorizedAtlasRegionMapper(harvard_mapper)


@pytest.fixture(scope="module")
def c2r_mapper():
    """
    Create a coord2region instance that includes the Harvard-Oxford atlas
    under the key "harvard-oxford".
    """
    return coord2region.coord2region("coord2region_data", {"harvard-oxford": {}})


# ---------------------- Tests for AtlasRegionMapper ---------------------- #
def test_get_hemisphere(harvard_mapper):
    hemisphere = harvard_mapper.get_hemisphere("Frontal Pole")
    # Depending on how your "Frontal Pole" label is set, it might return 'L', 'R', or None
    # if that label doesn't follow the naming convention with '_L' or '_R'.
    # For demonstration, we only assert that the function runs and returns something valid.
    assert hemisphere in ("L", "R", None)

def test_get_region_name_unknown(harvard_mapper):
    # Pass an integer that doesn't exist in the atlas volume
    region = harvard_mapper.get_region_name(999999)
    assert region == "Unknown"

def test_get_region_index(harvard_mapper):
    # Example region that definitely exists in Harvard-Oxford
    index = harvard_mapper.get_region_index("Precentral Gyrus")
    assert index != "Unknown"
    assert isinstance(index, int)

def test_list_of_regions(harvard_mapper):
    # We expect a non-empty list
    regions = harvard_mapper.get_list_of_regions()
    assert isinstance(regions, list)
    assert len(regions) > 0
    assert "Frontal Pole" in regions  # example check

def test_pos_to_source_and_back(harvard_mapper):
    # We'll pick a coordinate (x, y, z) in MNI space near the left frontal area
    pos = [-54., 36., -4.]
    source_ijk = harvard_mapper.pos_to_source(pos)
    assert len(source_ijk) == 3  # Should be a tuple (i, j, k)
    # Round trip test
    mni_again = harvard_mapper.source_to_pos(source_ijk)
    # Because of rounding in voxel space, we expect them to be close but not identical
    np.testing.assert_almost_equal(mni_again, pos, decimal=1)

def test_pos_to_index_and_region(harvard_mapper):
    pos = [-54., 36., -4.]
    idx = harvard_mapper.pos_to_index(pos)
    # idx can be 0 or "Unknown" if the coordinate is out of atlas bounds
    if idx != "Unknown":
        region = harvard_mapper.pos_to_region(pos)
        # We can't guarantee which region it belongs to, but it shouldn't be "Unknown"
        assert region != "Unknown"


# ---------------------- Tests for VectorizedAtlasRegionMapper ---------------------- #
def test_batch_get_region_names_indices(harvard_vectorized_mapper):
    # Suppose 2, 3, 4 are valid indices in the Harvard-Oxford volume
    region_names = harvard_vectorized_mapper.batch_get_region_names([2, 3, 4])
    # Just check that we got valid region names (none of them should be "Unknown" if they exist)
    assert all(name != "Unknown" for name in region_names)

    # Now convert them back to indices
    retrieved_indices = harvard_vectorized_mapper.batch_get_region_indices(region_names)
    # They should match the original [2, 3, 4] except for potential mismatch if some were missing
    for orig_i, ret_i in zip([2, 3, 4], retrieved_indices):
        if ret_i != "Unknown":
            assert orig_i == ret_i

def test_batch_pos_functions(harvard_vectorized_mapper, harvard_mapper):
    # Let's build a small set of coordinates (one for each region) for testing:
    # We'll just pick the 3rd voxel in each region for demonstration
    regions = harvard_mapper.get_list_of_regions()
    coords_list = []
    for reg in regions[:5]:
        # skip background
        if reg.lower() == "background":
            continue
        region_coords = harvard_mapper.region_to_pos(reg)
        if region_coords.shape[0] < 3:
            continue
        # pick the 3rd voxel for reproducibility
        coords_list.append(region_coords[2])

    # Now test the vectorized calls
    region_names = harvard_vectorized_mapper.batch_pos_to_region(coords_list)
    # Should all be valid (but not guaranteed if some coords are out-of-bounds or partial)
    assert len(region_names) == len(coords_list)

    region_indices = harvard_vectorized_mapper.batch_pos_to_index(coords_list)
    assert len(region_indices) == len(coords_list)

    # Convert these positions to voxel indices
    voxel_indices = harvard_vectorized_mapper.batch_pos_to_source(coords_list)
    assert len(voxel_indices) == len(coords_list)

    # Convert those voxels back to MNI coords
    mni_coords = harvard_vectorized_mapper.batch_source_to_pos(voxel_indices)
    # Should shape-match
    assert mni_coords.shape == (len(coords_list), 3)


# ---------------------- Tests for coord2region.coord2region ---------------------- #
def test_coord2region_batch_pos_to_region(c2r_mapper, harvard_mapper):
    # We'll reuse the same idea of picking a few MNI coords from known regions:
    regions = harvard_mapper.get_list_of_regions()[:5]
    coords_list = []
    for reg in regions:
        if reg.lower() == "background":
            continue
        region_coords = harvard_mapper.region_to_pos(reg)
        if region_coords.shape[0] == 0:
            continue
        coords_list.append(region_coords[0])  # just pick the first voxel

    result = c2r_mapper.batch_pos_to_region(coords_list)
    # result is a dict keyed by atlas name, in this case {"harvard-oxford": [...]}
    assert "harvard-oxford" in result
    assert len(result["harvard-oxford"]) == len(coords_list)


def test_coord2region_batch_region_to_pos(c2r_mapper):
    region_names = [
        "Occipital Fusiform Gyrus",
        "Frontal Opercular Cortex",
        "Central Opercular Cortex",
        "Parietal Opercular Cortex",
        "Planum Polare",
    ]
    result = c2r_mapper.batch_region_to_pos(region_names)
    # Should be keyed by "harvard-oxford"
    assert "harvard-oxford" in result
    for coords_array in result["harvard-oxford"]:
        # Each entry is an array of MNI coordinates
        assert isinstance(coords_array, np.ndarray)
        # We can't assume they aren't empty, but let's just show the call worked.
