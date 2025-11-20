import pytest

from coord2region.config import Coord2RegionConfig, ValidationError


def test_config_inline_coordinates_valid():
    cfg = Coord2RegionConfig.model_validate(
        {
            "input_type": "coords",
            "coordinates": [[1, 2, 3]],
            "outputs": ["summaries", "images"],
            "gemini_api_key": "KEY",
            "atlas_names": ["aal", "juelich"],
            "max_atlases": 3,
            "study_search_radius": 6,
            "summary_models": ["custom"],
            "summary_max_tokens": 256,
            "prompt_type": "custom",
            "custom_prompt": "Prompt for {coord}",
            "providers": {},
        }
    )

    inputs = cfg.collect_inputs(load_coords_file=lambda _: [[9.0, 9.0, 9.0]])
    assert inputs == [[1, 2, 3]]

    runtime = cfg.to_pipeline_runtime(inputs)
    assert runtime["input_type"] == "coords"
    assert runtime["outputs"] == ["summaries", "images"]
    assert runtime["config"]["gemini_api_key"] == "KEY"
    assert runtime["config"]["study_search_radius"] == 6.0
    assert runtime["config"]["summary_models"] == ["custom"]
    assert runtime["config"]["summary_max_tokens"] == 256
    assert runtime["config"]["prompt_type"] == "custom"
    assert runtime["config"]["custom_prompt"] == "Prompt for {coord}"


def test_config_coordinates_file_loader_invoked(tmp_path):
    path = tmp_path / "coords.csv"
    path.write_text("0,0,0\n", encoding="utf8")
    cfg = Coord2RegionConfig.model_validate(
        {
            "input_type": "coords",
            "coordinates_file": str(path),
            "outputs": ["region_labels"],
        }
    )

    def loader(p: str):
        assert p == str(path)
        return [[4, 5, 6]]

    inputs = cfg.collect_inputs(load_coords_file=loader)
    assert inputs == [[4, 5, 6]]


def test_config_missing_coordinates_raises():
    with pytest.raises(ValidationError):
        Coord2RegionConfig.model_validate(
            {
                "input_type": "coords",
                "outputs": ["region_labels"],
            }
        )


def test_config_summary_without_credentials_allowed():
    cfg = Coord2RegionConfig.model_validate(
        {
            "input_type": "coords",
            "coordinates": [[0, 0, 0]],
            "outputs": ["summaries"],
        }
    )
    assert cfg.outputs == ["summaries"]


def test_config_image_prompt_passthrough():
    cfg = Coord2RegionConfig.model_validate(
        {
            "input_type": "coords",
            "coordinates": [[0, 0, 0]],
            "outputs": ["images"],
            "image_prompt_type": "functional",
            "image_custom_prompt": "Show activation at {coordinate}",
        }
    )
    runtime = cfg.to_pipeline_runtime([[0, 0, 0]])
    conf = runtime["config"]
    assert conf["image_prompt_type"] == "functional"
    assert conf["image_custom_prompt"] == "Show activation at {coordinate}"


def test_config_atlas_limit_enforced():
    with pytest.raises(ValidationError):
        Coord2RegionConfig.model_validate(
            {
                "input_type": "coords",
                "coordinates": [[0, 0, 0]],
                "outputs": ["region_labels"],
                "atlas_names": ["aal", "juelich"],
                "max_atlases": 1,
            }
        )


def test_config_region_inputs_from_legacy_field():
    cfg = Coord2RegionConfig.model_validate(
        {
            "input_type": "region_names",
            "inputs": ["Region A", "Region B"],
            "outputs": ["region_labels"],
        }
    )
    assert cfg.region_names == ["Region A", "Region B"]
    inputs = cfg.collect_inputs(load_coords_file=lambda _: [])
    assert inputs == ["Region A", "Region B"]


def test_config_region_names_mni_coordinates_output():
    cfg = Coord2RegionConfig.model_validate(
        {
            "input_type": "region_names",
            "region_names": ["Region"],
            "outputs": ["mni_coordinates"],
        }
    )
    assert cfg.outputs == ["mni_coordinates"]
    inputs = cfg.collect_inputs(load_coords_file=lambda _: [])
    assert inputs == ["Region"]


def test_config_coordinates_from_string_inputs():
    cfg = Coord2RegionConfig.model_validate(
        {
            "input_type": "coords",
            "inputs": ["1 2 3", "4,5,6"],
            "outputs": ["region_labels"],
        }
    )
    inputs = cfg.collect_inputs(load_coords_file=lambda _: [])
    assert inputs == [[1, 2, 3], [4, 5, 6]]


def test_config_sources():
    cfg = Coord2RegionConfig.model_validate(
        {
            "input_type": "coords",
            "coordinates": [[0, 0, 0]],
            "outputs": ["summaries"],
            "gemini_api_key": "KEY",
            "sources": ["Mock"],
        }
    )
    runtime = cfg.to_pipeline_runtime([[0, 0, 0]])
    assert runtime["config"]["sources"] == ["Mock"]


def test_config_summary_models_list():
    cfg = Coord2RegionConfig.model_validate(
        {
            "input_type": "coords",
            "coordinates": [[0, 0, 0]],
            "outputs": ["summaries"],
            "summary_models": ["model-a", "model-b", "model-a"],
        }
    )

    runtime = cfg.to_pipeline_runtime([[0, 0, 0]])
    config = runtime["config"]
    assert config["summary_models"] == ["model-a", "model-b"]


def test_config_legacy_block_passthrough():
    cfg = Coord2RegionConfig.model_validate(
        {
            "inputs": [[1, 2, 3]],
            "outputs": ["region_labels"],
            "config": {
                "atlas_names": ["aal"],
                "working_directory": "/tmp/data",
            },
        }
    )
    runtime = cfg.to_pipeline_runtime(cfg.collect_inputs(load_coords_file=lambda _: []))
    assert runtime["config"]["atlas_names"] == ["aal"]
    assert runtime["config"]["working_directory"] == "/tmp/data"


def test_config_max_atlases_checks_legacy():
    with pytest.raises(ValidationError):
        Coord2RegionConfig.model_validate(
            {
                "inputs": [[0, 0, 0]],
                "outputs": ["region_labels"],
                "max_atlases": 1,
                "config": {"atlas_names": ["aal", "juelich"]},
            }
        )


def test_config_mni_coordinates_requires_region_names():
    with pytest.raises(ValidationError):
        Coord2RegionConfig.model_validate(
            {
                "input_type": "coords",
                "coordinates": [[0, 0, 0]],
                "outputs": ["mni_coordinates"],
            }
        )


def test_config_has_llm_credentials_detection():
    cfg = Coord2RegionConfig.model_validate(
        {
            "input_type": "coords",
            "coordinates": [[0, 0, 0]],
            "outputs": ["summaries"],
            "providers": {"echo": {}},
        }
    )
    assert cfg._has_llm_credentials() is True


def test_config_has_llm_credentials_from_legacy():
    cfg = Coord2RegionConfig.model_validate(
        {
            "inputs": [[0, 0, 0]],
            "outputs": ["summaries"],
            "config": {"openai_api_key": "secret"},
        }
    )
    assert cfg._has_llm_credentials() is True


@pytest.mark.parametrize(
    "value, expected",
    [
        ("", False),
        ("~/atlas.nii.gz", True),
        ("relative/path.nii.gz", True),
        ("http://example.com/atlas", True),
        ("C:/atlas.nii.gz", True),
    ],
)
def test_config_looks_like_path(value, expected):
    assert Coord2RegionConfig._looks_like_path(value) is expected


def test_config_derive_atlas_config_variants():
    url = Coord2RegionConfig._derive_atlas_config("https://example.com/atlas")
    assert url == {"atlas_url": "https://example.com/atlas"}

    path = Coord2RegionConfig._derive_atlas_config("~/atlas.nii.gz")
    assert path == {"atlas_file": "~/atlas.nii.gz"}

    assert Coord2RegionConfig._derive_atlas_config("aal") is None


def test_config_requires_output_name_when_format_set():
    with pytest.raises(ValidationError):
        Coord2RegionConfig.model_validate(
            {
                "input_type": "coords",
                "coordinates": [[0, 0, 0]],
                "outputs": ["region_labels"],
                "output_format": "json",
            }
        )


def test_config_rejects_multiple_coordinate_sources(tmp_path):
    path = tmp_path / "coords.csv"
    path.write_text("0,0,0\n", encoding="utf8")
    with pytest.raises(ValidationError):
        Coord2RegionConfig.model_validate(
            {
                "input_type": "coords",
                "coordinates": [[0, 0, 0]],
                "coordinates_file": str(path),
                "outputs": ["region_labels"],
            }
        )


def test_config_region_inputs_disallow_coordinates():
    with pytest.raises(ValidationError):
        Coord2RegionConfig.model_validate(
            {
                "input_type": "region_names",
                "region_names": ["Region"],
                "coordinates": [[0, 0, 0]],
                "outputs": ["region_labels"],
            }
        )
