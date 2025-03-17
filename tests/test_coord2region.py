import pytest
import numpy as np

from coord2region import fetching, coord2region


@pytest.fixture(scope="module")
def harvard_data():
    """
    This fixture downloads/loads the Harvard-Oxford atlas once
    and returns the dict that includes 'vol', 'hdr', 'labels', etc.
    """
    af = fetching.AtlasFetcher(data_dir="coord2region_data")
    harvard = af.fetch_atlas("harvard-oxford")
    return harvard


@pytest.fixture(scope="module")
def harvard_mapper(harvard_data):
    """
    This fixture creates an AtlasRegionMapper for the Harvard-Oxford atlas.
    """
    return coord2region.AtlasRegionMapper(
        name="harvard-oxford",
        vol=harvard_data["vol"],
        hdr=harvard_data["hdr"],
        labels=harvard_data["labels"],
    )


def test_get_hemisphere(harvard_mapper):
    # Example test for get_hemisphere
    hemi = harvard_mapper.get_hemisphere("Frontal Pole")
    assert hemi is None or hemi in ("L", "R"), \
        f"Expected None or 'L'/'R', got {hemi}"

    # You might not have explicit hemisphere annotations in the label for Harvard-Oxford,
    # so you can adapt this test to match actual behavior.


def test_get_region_name(harvard_mapper):
    # If you know a specific index -> label mapping, you can test it
    # But note that many Harvard-Oxford versions do NOT store numeric indexes in the “labels” dictionary
    # so this might yield "Unknown". Adapt as needed:
    region_name = harvard_mapper.get_region_name(7)
    assert region_name == "Unknown" or isinstance(region_name, str), \
        f"Expected a string or 'Unknown', got {region_name}"


def test_get_region_index(harvard_mapper):
    # For example, check a known label:
    # "Frontal Pole" might or might not yield a valid integer
    # If it doesn't exist as a key in the dict, the function might return "Unknown"
    idx = harvard_mapper.get_region_index("Precentral Gyrus")
    assert isinstance(idx, (int, str)), f"Expected int or 'Unknown', got {idx}"


def test_list_of_regions(harvard_mapper):
    regions = harvard_mapper.get_list_of_regions()
    assert isinstance(regions, list)
    assert len(regions) > 0, "No regions found, unexpected for Harvard-Oxford"


def test_pos_to_source(harvard_mapper):
    # Convert an MNI coordinate to voxel indices.
    voxel_idx = harvard_mapper.pos_to_source([-54., 36., -4.])
    assert len(voxel_idx) == 3, f"Expected 3D voxel index, got {voxel_idx}"
    for v in voxel_idx:
        assert isinstance(v, int), "Voxel indices must be integers"


def test_pos_to_index(harvard_mapper):
    # Convert an MNI coordinate to region index
    region_idx = harvard_mapper.pos_to_index([-54., 36., -4.])
    # Could be an int or "Unknown" if it’s out-of-bounds for the atlas
    assert isinstance(region_idx, (int, str)), f"Expected int or 'Unknown', got {region_idx}"


def test_pos_to_region(harvard_mapper):
    # Convert an MNI coordinate to region name
    region_name = harvard_mapper.pos_to_region([-54., 36., -4.])
    assert isinstance(region_name, str), f"Expected string, got {type(region_name)}"


def test_source_to_pos(harvard_mapper):
    # Convert voxel indices (i, j, k) back to MNI
    # This is tricky if you don't have known ground truth,
    # so just check that it's a 3-element array
    coords = harvard_mapper.source_to_pos([30, 40, 50])
    assert coords.shape == (3,), f"Expected shape (3,) MNI coords, got {coords.shape}"


def test_index_to_pos(harvard_mapper):
    # If you know a region index that definitely exists, test it:
    # For this example, assume index=1 is a real region
    coords = harvard_mapper.index_to_pos(1)
    # coords might be many points, so it’s an array Nx3
    assert coords.ndim == 2 and coords.shape[1] == 3, \
        f"Expected Nx3 array, got shape {coords.shape}"


def test_region_to_pos(harvard_mapper):
    # If you know a region name that definitely exists:
    # e.g., "Frontal Pole"
    coords = harvard_mapper.region_to_pos("Frontal Pole")
    assert coords.ndim == 2 and coords.shape[1] == 3, \
        f"Expected Nx3 array, got shape {coords.shape}"


@pytest.fixture(scope="module")
def vectorized_mapper(harvard_mapper):
    """
    Create a VectorizedAtlasRegionMapper for the Harvard-Oxford atlas.
    """
    return coord2region.VectorizedAtlasRegionMapper(harvard_mapper)


def test_batch_pos_to_region(vectorized_mapper, harvard_mapper):
    # Example: Build a list of MNI coords from a handful of region centroids
    # (completely arbitrary for demonstration)
    labels = harvard_mapper.get_list_of_regions()[:5]  # just first 5 to avoid huge list
    coords_for_tests = []
    for label in labels:
        pos_array = harvard_mapper.region_to_pos(label)
        if pos_array.shape[0] > 0:
            coords_for_tests.append(pos_array[0])  # pick the first voxel as an example

    result = vectorized_mapper.batch_pos_to_region(coords_for_tests)
    assert len(result) == len(coords_for_tests)
    for r in result:
        assert isinstance(r, str)


def test_batch_get_region_names(vectorized_mapper):
    # Suppose we guess a few indexes
    region_names = vectorized_mapper.batch_get_region_names([2, 3, 4])
    assert len(region_names) == 3


def test_batch_get_region_indices(vectorized_mapper):
    # Suppose from the above region_names, we convert them back to indices
    region_names = ["Frontal Pole", "Precentral Gyrus", "Unknown Region"]
    region_indices = vectorized_mapper.batch_get_region_indices(region_names)
    assert len(region_indices) == len(region_names)


def test_c2r_coord2region_api():
    # Now test the high-level coord2region.coord2region class
    c2r = coord2region.coord2region(
        data_dir="coord2region_data", 
        atlases={"harvard-oxford": {}}
    )
    # Provide a few coordinates:
    coords = [[-54., 36., -4.], [10., 20., 30.]]
    # batch_pos_to_region returns a dict keyed by atlas name
    result_dict = c2r.batch_pos_to_region(coords)
    assert "harvard-oxford" in result_dict
    assert len(result_dict["harvard-oxford"]) == 2


def test_c2r_region_to_pos():
    # Similarly for region -> coords
    c2r = coord2region.coord2region(
        data_dir="coord2region_data",
        atlases={"harvard-oxford": {}}
    )
    # This method does not exist in c2r directly, but you might test e.g.:
    # c2r.batch_region_to_pos(["Frontal Pole", "Precentral Gyrus"])
    # By default, `coord2region` only has batch_pos_to_region, batch_get_region_names,
    # batch_get_region_indices. 
    pass

