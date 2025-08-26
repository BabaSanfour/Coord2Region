import pytest
import numpy as np
from coord2region.fetching import AtlasFetcher
from coord2region.coord2region import AtlasMapper, BatchAtlasMapper, MultiAtlasMapper

# Atlas Properties for Validation
PROPERTIES = {
    "harvard-oxford": {
        "infer_hemisphere": [('Frontal Pole', None)],
        "region2index": [('Insular Cortex', 2)],
        "allregions": 49,
    },
    "juelich": {
        "infer_hemisphere": [('GM Primary motor cortex BA4p', None)],
        "region2index": [('GM Amygdala_laterobasal group', 2)],
        "allregions": 63,
    },
    "schaefer": {
        "infer_hemisphere": [('7Networks_LH_Vis_1', 'L'), ('7Networks_RH_Default_PFCv_4', 'R')],
        "region2index": [('7Networks_LH_Vis_3', 2)],
        "allregions": 400,
    },
    "yeo": {
        "infer_hemisphere": [('17Networks_9', None)],
        "region2index": [('17Networks_2', 2)],
        "allregions": 18,
    }
}

# Test coordinates (ground truth needed)
TEST_MNIS = [[-54., 36., -4.],[10., 20., 30.]]
TEST_VOXELS = [[30, 40, 50]]


# Fixture: Load Fresh Atlas Data Per Test
@pytest.fixture(scope="function")
def fresh_atlas_data(request):
    """Loads and returns atlas data ('vol', 'hdr', 'labels') for a given atlas."""
    atlas_name = request.param
    print(f"\nLoading atlas: {atlas_name}")  # Debugging
    af = AtlasFetcher(data_dir="coord2region_data")
    return atlas_name, af.fetch_atlas(atlas_name)


# Fixture: Create Volumetric Mapper
@pytest.fixture(scope="function")
def volumetric_mapper(fresh_atlas_data):
    """Creates a fresh AtlasMapper per test."""
    atlas_name, data = fresh_atlas_data
    return AtlasMapper(
        name=atlas_name,
        vol=data["vol"],
        hdr=data["hdr"],
        labels=data.get("labels", None)
    )


# Fixture: Create BatchAtlasMapper for Generalized Atlas
@pytest.fixture(scope="function")
def vectorized_mapper(fresh_atlas_data):
    """Creates a BatchAtlasMapper for a given atlas."""
    atlas_name, data = fresh_atlas_data
    return BatchAtlasMapper(
        AtlasMapper(
            name=atlas_name,
            vol=data["vol"],
            hdr=data["hdr"],
            labels=data.get("labels", None)
        )
    )


# Test: Debug Parameterization
@pytest.mark.parametrize("fresh_atlas_data", PROPERTIES.keys(), indirect=True)
def test_debug_parametrize(fresh_atlas_data):
    atlas_name, _ = fresh_atlas_data
    print(f"\nRunning test for atlas: {atlas_name}")
    assert atlas_name in PROPERTIES.keys()


# Test: Atlas Structure
@pytest.mark.parametrize("fresh_atlas_data", PROPERTIES.keys(), indirect=True)
def test_atlas_structure(fresh_atlas_data):
    atlas_name, data = fresh_atlas_data
    assert "vol" in data and data["vol"] is not None, f"{atlas_name} missing 'vol'"
    assert "hdr" in data and data["hdr"].shape == (4, 4), f"{atlas_name} missing 'hdr'"
    assert "labels" in data and len(data["labels"]) > 0, f"{atlas_name} missing 'labels'"


# Test: Hemisphere Inference
@pytest.mark.parametrize("fresh_atlas_data", PROPERTIES.keys(), indirect=True)
def test_infer_hemisphere(volumetric_mapper, fresh_atlas_data):
    atlas_name, _ = fresh_atlas_data
    for region, expected in PROPERTIES[atlas_name]['infer_hemisphere']:
        result = volumetric_mapper.infer_hemisphere(region)
        assert result == expected, f"Error in infer_hemisphere for {atlas_name}: expected {expected}, got {result}"


# Test: Region Index Lookup
@pytest.mark.parametrize("fresh_atlas_data", PROPERTIES.keys(), indirect=True)
def test_region_to_index(volumetric_mapper, fresh_atlas_data):
    # skip yeo and schaefer for now
    if fresh_atlas_data[0] in ['yeo', 'schaefer']:
        pytest.skip(f"Skipping test for {fresh_atlas_data[0]} atlas")
    atlas_name, _ = fresh_atlas_data
    for region, expected_index in PROPERTIES[atlas_name]['region2index']:
        idx = volumetric_mapper.region_index_from_name(region)
        assert idx == expected_index, f"Error in region2index for {atlas_name}: expected {expected_index}, got {idx}"


# Test: Batch MNI to Region Name
@pytest.mark.parametrize("fresh_atlas_data", PROPERTIES.keys(), indirect=True)
def test_batch_mni_to_region_name(vectorized_mapper, volumetric_mapper):
    labels = volumetric_mapper.list_all_regions()[:5]
    coords_for_tests = [volumetric_mapper.region_name_to_mni(label)[0] for label in labels if volumetric_mapper.region_name_to_mni(label).shape[0] > 0]

    if not coords_for_tests:
        pytest.skip("No valid coords found for testing batch MNI->region")

    result = vectorized_mapper.batch_mni_to_region_name(coords_for_tests)
    assert len(result) == len(coords_for_tests)
    assert all(isinstance(r, str) for r in result)


# Test: Batch Region Index to Name
@pytest.mark.parametrize("fresh_atlas_data", PROPERTIES.keys(), indirect=True)
def test_batch_region_name_from_index(vectorized_mapper):
    region_names = vectorized_mapper.batch_region_name_from_index([2, 3, 4])
    assert len(region_names) == 3


# Test: Batch Region Name to Index
@pytest.mark.parametrize("fresh_atlas_data", PROPERTIES.keys(), indirect=True)
def test_batch_region_index_from_name(vectorized_mapper):
    example_region1=PROPERTIES[vectorized_mapper.mapper.name]['region2index'][0][0]
    example_region2=PROPERTIES[vectorized_mapper.mapper.name]['infer_hemisphere'][0][0]
    region_indices = vectorized_mapper.batch_region_index_from_name([example_region1,example_region2, "Unknown Region"])
    assert len(region_indices) == 3


# Test: MultiAtlasMapper API
def test_multiatlas_api():
    """Test the high-level MultiAtlasMapper class."""
    # also skipping yeo ! 
    c2r = MultiAtlasMapper(data_dir="coord2region_data", atlases={x: {} for x in PROPERTIES.keys() if x != "yeo"})
    coords = TEST_MNIS
    
    result_dict = c2r.batch_mni_to_region_names(coords)
    for atlas_name in PROPERTIES.keys():
        if atlas_name == "yeo":
            continue
        assert atlas_name in result_dict
        assert len(result_dict[atlas_name]) == len(coords)

    for region, _ in PROPERTIES[atlas_name]['region2index']:
        idx = c2r.batch_region_name_to_mni([region])

        for atlas2 in PROPERTIES.keys():
            if atlas2 == 'yeo':
                continue
            if atlas2 == atlas_name:
                assert idx[atlas2][0].shape[0]!=0, f"Expected non-empty array for {atlas2} when querying {atlas_name} region"
            else:
                assert idx[atlas2][0].shape[0]==0, f"Expected empty array for {atlas2} when querying {atlas_name} region"