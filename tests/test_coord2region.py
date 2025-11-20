import pytest
import numpy as np
from coord2region.fetching import AtlasFetcher
from coord2region.coord2region import AtlasMapper, BatchAtlasMapper, MultiAtlasMapper, _get_numeric_hemi

pytestmark = pytest.mark.requires_network

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


@pytest.mark.parametrize("fresh_atlas_data", ["harvard-oxford"], indirect=True)
def test_infer_hemisphere_warns_for_unknown_region(volumetric_mapper):
    region = "Lat_Fis-post-rh"
    assert volumetric_mapper.region_index_from_name(region) == "Unknown"
    with pytest.warns(UserWarning, match="not part of the 'harvard-oxford' atlas"):
        assert volumetric_mapper.infer_hemisphere(region) is None


@pytest.mark.parametrize("fresh_atlas_data", ["schaefer"], indirect=True)
def test_schaefer_numeric_index_infer_hemisphere(volumetric_mapper):
    region_L, expected_L = PROPERTIES["schaefer"]["infer_hemisphere"][0]
    region_R, expected_R = PROPERTIES["schaefer"]["infer_hemisphere"][1]
    idx_L = volumetric_mapper.region_index_from_name(region_L)
    idx_R = volumetric_mapper.region_index_from_name(region_R)
    assert volumetric_mapper.infer_hemisphere(idx_L) == expected_L
    assert volumetric_mapper.infer_hemisphere(idx_R) == expected_R

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

def _make_dummy_mapper(region_vertices=None):
    if region_vertices is None:
        region_vertices = {}
    vol = [np.array([]), np.array([])]
    return AtlasMapper(name="dummy", vol=vol, hdr=None, regions=region_vertices)


def test_surface_out_of_bounds():
    mapper = _make_dummy_mapper({"r": np.arange(3)})
    mapper.convert_to_source = lambda mni: np.array([5])
    assert mapper.mni_to_region_index([0, 0, 0]) == "Unknown"


def test_surface_multi_vertex_matches():
    regions = {"r": np.array([10, 20, 30, 40])}
    mapper = _make_dummy_mapper(regions)
    mapper.convert_to_source = lambda mni: np.array([20, 30])
    result = mapper.mni_to_region_index([0, 0, 0])
    assert np.array_equal(result, np.array([20, 30]))

def _surface_mapper():
    vol = [np.array([0, 1]), np.array([2, 3])]
    regions = {"L": np.array([0, 1]), "R": np.array([2, 3])}
    return AtlasMapper(name="dummy", vol=vol, hdr=None, regions=regions)


def _patch_vertex_to_mni(monkeypatch):
    import mne
    coords = {
        0: np.array([-1.0, 0.0, 0.0]),
        1: np.array([-2.0, 0.0, 0.0]),
        2: np.array([1.0, 0.0, 0.0]),
        3: np.array([2.0, 0.0, 0.0]),
    }

    def fake_vertex_to_mni(vertices, hemis, subject, subjects_dir=None):
        verts = np.atleast_1d(vertices)
        return np.array([coords[v] for v in verts])

    monkeypatch.setattr(mne, "vertex_to_mni", fake_vertex_to_mni)


def test_convert_to_source_hemi_restriction(monkeypatch):
    mapper = _surface_mapper()
    _patch_vertex_to_mni(monkeypatch)

    coord = np.array([1.1, 0.0, 0.0])

    # Search both hemispheres (default)
    assert mapper.convert_to_source(coord) == 2

    # Restrict to left hemisphere
    assert mapper.convert_to_source(coord, hemi=0) == 0

    # Restrict to right hemisphere
    assert mapper.convert_to_source(coord, hemi=1) == 2


def test_mni_to_vertex_returns_nearest(monkeypatch):
    mapper = _surface_mapper()
    _patch_vertex_to_mni(monkeypatch)

    coord = np.array([1.8, 0.0, 0.0])

    # Nearest to vertex index 3 (coord [2,0,0])
    assert mapper.convert_to_source(coord) == 3

def test_region_name_to_mni_and_centroid(monkeypatch):
    mapper = _surface_mapper()
    _patch_vertex_to_mni(monkeypatch)
    coords = mapper.region_name_to_mni("L")
    np.testing.assert_allclose(coords, np.array([[-1.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]))
    centroid = mapper.region_centroid("L")
    np.testing.assert_allclose(centroid, np.array([-1.5, 0.0, 0.0]))

def _volume_mapper():
    vol = np.zeros((2, 2, 2))
    hdr = np.eye(4)
    return AtlasMapper(name="dummy", vol=vol, hdr=hdr, labels=None)


def test_mni_to_voxel_nearest_outside():
    mapper = _volume_mapper()
    coord = np.array([2.3, 0.0, 0.0])
    assert mapper.mni_to_voxel(coord) == (1, 0, 0)


def test_convert_to_source_uses_voxel_search():
    mapper = _volume_mapper()
    coord = np.array([2.3, 0.0, 0.0])
    assert tuple(mapper.convert_to_source(coord)) == (1, 0, 0)


def volume_mapper():
    vol = np.zeros((3, 1, 1), dtype=int)
    vol[0, 0, 0] = 1
    vol[2, 0, 0] = 2
    hdr = np.eye(4)
    labels = {"1": "A", "2": "B"}
    return AtlasMapper("vol", vol, hdr, labels=labels)


def coord_mapper():
    coords = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    labels = ["A", "B"]
    indexes = [1, 2]
    return AtlasMapper("coord", coords, hdr=None, labels=labels, indexes=indexes)


def surface_mapper(monkeypatch):
    vol = [np.array([0]), np.array([1])]
    regions = {"A": np.array([0]), "B": np.array([1])}
    mapper = AtlasMapper("surf", vol, hdr=None, regions=regions)

    def fake_vertex_to_mni(vertices, hemi, subject, subjects_dir):
        vertices = np.atleast_1d(vertices)
        out = []
        for v in vertices:
            if int(v) == 0:
                out.append([-1.0, 0.0, 0.0])
            else:
                out.append([1.0, 0.0, 0.0])
        return np.array(out)

    monkeypatch.setattr(
        "coord2region.coord2region.mne.vertex_to_mni", fake_vertex_to_mni
    )
    return mapper


@pytest.mark.parametrize("mapper_fn", [volume_mapper, coord_mapper, surface_mapper])
def test_nearest_region(mapper_fn, monkeypatch):
    mapper = mapper_fn(monkeypatch) if mapper_fn is surface_mapper else mapper_fn()
    coord = [1.0, 0.0, 0.0] if mapper_fn is volume_mapper else [0.0, 0.0, 0.0]
    assert mapper.mni_to_region_name(coord, max_distance=0.5) == "Unknown"
    assert mapper.mni_to_region_index(coord, max_distance=2) in (0, 1)
    name = mapper.mni_to_region_name(coord, max_distance=2)
    assert name in ("A", "Unknown")
    if name != "Unknown":
        assert name == "A"

def _make_dummy_mapper_save():
    vol = np.zeros((2, 2, 2))
    vol[0, 0, 0] = 1
    hdr = np.eye(4)
    labels = {"1": "region1"}
    return AtlasMapper(name="dummy", vol=vol, hdr=hdr, labels=labels)


def test_atlasmapper_save_load(tmp_path):
    mapper = _make_dummy_mapper_save()
    coord = [0, 0, 0]
    before = mapper.mni_to_region_name(coord)
    path = tmp_path / "mapper.pkl"
    mapper.save(path)
    loaded = AtlasMapper.load(path)
    after = loaded.mni_to_region_name(coord)
    assert before == after


def test_multiatlasmapper_save_load(tmp_path):
    mapper1 = _make_dummy_mapper_save()
    mapper2 = _make_dummy_mapper_save()
    multi = MultiAtlasMapper.__new__(MultiAtlasMapper)
    multi.mappers = {
        "one": BatchAtlasMapper(mapper1),
        "two": BatchAtlasMapper(mapper2),
    }
    coords = [[0, 0, 0]]
    before = multi.batch_mni_to_region_names(coords)
    path = tmp_path / "multi.pkl"
    multi.save(path)
    loaded = MultiAtlasMapper.load(path)
    after = loaded.batch_mni_to_region_names(coords)
    assert before == after

def test_mni_to_tal_and_back():
    vol = np.zeros((1, 1, 1))
    hdr = np.eye(4)
    mapper = AtlasMapper(name="dummy", vol=vol, hdr=hdr)
    mni = np.array([30.0, 20.0, 40.0])
    tal = mapper.convert_system(mni, "mni", "tal")
    expected_tal = np.array([29.7, 21.216, 35.786])
    assert np.allclose(tal, expected_tal, atol=1e-3)
    back = mapper.convert_system(tal, "tal", "mni")
    assert np.allclose(back, mni, atol=1e-6)

def coord_mapper():
    coords = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    labels = ["A", "B"]
    indexes = [1, 2]
    return AtlasMapper("coord", coords, hdr=None, labels=labels, indexes=indexes)

def test_probability_decreases_from_centroid():
    mapper = coord_mapper()
    centroid = mapper.region_centroid(1)
    near = centroid
    far = centroid + np.array([2.0, 0.0, 0.0])

    probs_near = mapper.membership_scores(near)
    probs_far = mapper.membership_scores(far)

    assert probs_near["A"] > probs_far["A"]
    assert np.isclose(sum(probs_near.values()), 1.0)
    assert np.isclose(sum(probs_far.values()), 1.0)

def _toy_atlas():
    coords = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], dtype=float)
    labels = ["orig", "x10", "y10"]
    indexes = [1, 2, 3]
    return coords, labels, indexes


def test_coords_mapper_basic():
    coords, labels, idxs = _toy_atlas()
    mapper = AtlasMapper("toy", coords, None, labels=labels, indexes=idxs)
    assert mapper.atlas_type == "coords"
    assert mapper.mni_to_region_name([0, 0, 0]) == "orig"
    assert mapper.mni_to_region_index([10, 0, 0]) == 2
    idx, dist = mapper.mni_to_region_index([9, 0, 0], return_distance=True)
    assert idx == 2 and np.isclose(dist, 1.0)


def test_multiatlas_coords(monkeypatch):
    coords, labels, idxs = _toy_atlas()
    import pandas as pd
    df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "z": coords[:, 2], "label": labels}, index=idxs)
    atlas_data = {"vol": df, "hdr": None, "labels": None}

    def fake_fetch(self, name, **kwargs):
        return atlas_data

    monkeypatch.setattr(AtlasFetcher, "fetch_atlas", fake_fetch)
    mam = MultiAtlasMapper(data_dir="", atlases={"toy": {}})
    res = mam.batch_mni_to_region_names([[0, 0, 0], [9, 0, 0]])
    assert res["toy"] == ["orig", "x10"]
    mapper = mam.mappers["toy"].mapper
    assert mapper.atlas_type == "coords"
    assert mapper.indexes == idxs

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

