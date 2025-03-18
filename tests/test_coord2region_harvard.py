import pytest
import numpy as np

from coord2region.fetching import AtlasFetcher
from coord2region.coord2region import (
    VolumetricAtlasMapper,
    BatchAtlasMapper,
    MultiAtlasMapper
)

@pytest.fixture(scope="module")
def harvard_data():
    """
    This fixture downloads/loads the Harvard-Oxford atlas once
    and returns the dict that includes 'vol', 'hdr', 'labels', etc.
    """
    af = AtlasFetcher(data_dir="coord2region_data")
    harvard = af.fetch_atlas("harvard-oxford")
    return harvard


@pytest.fixture(scope="module")
def harvard_mapper(harvard_data):
    """
    This fixture creates a VolumetricAtlasMapper for the Harvard-Oxford atlas.
    """
    return VolumetricAtlasMapper(
        name="harvard-oxford",
        vol=harvard_data["vol"],
        hdr=harvard_data["hdr"],
        labels=harvard_data.get("labels", None)
    )


def test_infer_hemisphere(harvard_mapper):
    # Example test for infer_hemisphere
    hemi = harvard_mapper.infer_hemisphere("Frontal Pole")  # might be 'Unknown' in Harvard-Oxford
    assert hemi is None or hemi in ("L", "R"), \
        f"Expected None or 'L'/'R', got {hemi}"


def test_region_name_for_index(harvard_mapper):
    # If you know a numeric index, e.g. 7, check the region name.
    region_name = harvard_mapper.region_name_for_index(7)
    assert isinstance(region_name, str)


def test_region_index_for_name(harvard_mapper):
    idx = harvard_mapper.region_index_for_name("Precentral Gyrus")
    assert isinstance(idx, (int, str)), f"Expected int or 'Unknown', got {idx}"


def test_list_all_regions(harvard_mapper):
    regions = harvard_mapper.list_all_regions()
    assert isinstance(regions, list)
    assert len(regions) > 0, "No regions found, unexpected for Harvard-Oxford."


def test_mni_to_voxel(harvard_mapper):
    voxel_idx = harvard_mapper.mni_to_voxel([-54., 36., -4.])
    assert len(voxel_idx) == 3, f"Expected 3D voxel index, got {voxel_idx}"
    for v in voxel_idx:
        assert isinstance(v, int), "Voxel indices must be integers."


def test_mni_to_region_index(harvard_mapper):
    region_idx = harvard_mapper.mni_to_region_index([-54., 36., -4.])
    assert isinstance(region_idx, (int, str)), f"Expected int or 'Unknown', got {region_idx}"


def test_mni_to_region_name(harvard_mapper):
    region_name = harvard_mapper.mni_to_region_name([-54., 36., -4.])
    assert isinstance(region_name, str), f"Expected string, got {type(region_name)}"


def test_voxel_to_mni(harvard_mapper):
    coords = harvard_mapper.voxel_to_mni([30, 40, 50])
    assert coords.shape == (3,), f"Expected shape (3,), got {coords.shape}"


def test_region_index_to_mni(harvard_mapper):
    # e.g., if index=1 is a real region
    coords = harvard_mapper.region_index_to_mni(1)
    assert coords.ndim == 2 and coords.shape[1] == 3, \
        f"Expected Nx3 array, got {coords.shape}"


def test_region_name_to_mni(harvard_mapper):
    coords = harvard_mapper.region_name_to_mni("Frontal Pole")
    assert coords.ndim == 2 and coords.shape[1] == 3, \
        f"Expected Nx3 array, got {coords.shape}"


@pytest.fixture(scope="module")
def vectorized_mapper(harvard_mapper):
    """
    Create a BatchAtlasMapper for the Harvard-Oxford atlas.
    """
    return BatchAtlasMapper(harvard_mapper)


def test_batch_mni_to_region_name(vectorized_mapper, harvard_mapper):
    labels = harvard_mapper.list_all_regions()[:5]
    coords_for_tests = []
    for label in labels:
        arr = harvard_mapper.region_name_to_mni(label)
        if arr.shape[0] > 0:
            coords_for_tests.append(arr[0])  # pick first voxel as example

    if len(coords_for_tests) == 0:
        pytest.skip("No valid coords found for testing batch MNI->region")

    result = vectorized_mapper.batch_mni_to_region_name(coords_for_tests)
    assert len(result) == len(coords_for_tests)
    for r in result:
        assert isinstance(r, str)


def test_batch_region_name_for_index(vectorized_mapper):
    region_names = vectorized_mapper.batch_region_name_for_index([2, 3, 4])
    assert len(region_names) == 3


def test_batch_region_index_for_name(vectorized_mapper):
    some_names = ["Frontal Pole", "Precentral Gyrus", "Unknown Region"]
    region_indices = vectorized_mapper.batch_region_index_for_name(some_names)
    assert len(region_indices) == len(some_names)


def test_multiatlas_api():
    # Test the high-level MultiAtlasMapper class
    c2r = MultiAtlasMapper(
        data_dir="coord2region_data", 
        atlases={"harvard-oxford": {}}
    )
    coords = [[-54., 36., -4.], [10., 20., 30.]]
    result_dict = c2r.batch_mni_to_region_names(coords)
    assert "harvard-oxford" in result_dict
    assert len(result_dict["harvard-oxford"]) == 2

    # region -> coords
    # example call:
    # c2r.batch_region_name_to_mni(["Frontal Pole", "Precentral Gyrus"])
    # etc.
