from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCHEMA_PATH = Path(__file__).resolve().parents[1] / "docs" / "static" / "schema.json"


def _load_coord2region_config():
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "coord2region" / "config.py"
    spec = importlib.util.spec_from_file_location("coord2region.config", config_path)
    if spec is None or spec.loader is None:
        msg = "Unable to load coord2region.config module"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module.Coord2RegionConfig  # type: ignore[attr-defined]


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def test_schema_matches_model_definition() -> None:
    file_schema = load_schema()

    module, coord2region_config = _load_coord2region_config()
    coord2region_config.model_rebuild(_types_namespace=module.__dict__)
    model_schema = coord2region_config.model_json_schema()

    # Allow UI-driven presentation differences: ignore root title/description
    def _strip_meta(d: dict) -> dict:
        cleaned = dict(d)
        cleaned.pop("title", None)
        cleaned.pop("description", None)
        return cleaned

    assert _strip_meta(file_schema) == _strip_meta(model_schema)

    props = file_schema["properties"]
    assert props["input_type"]["enum"] == ["coords", "region_names"]
    assert props["outputs"]["items"]["enum"] == [
        "region_labels",
        "summaries",
        "images",
        "raw_studies",
        "mni_coordinates",
    ]
    assert props["image_backend"]["enum"] == ["ai", "nilearn", "both"]
    assert props["batch_size"]["minimum"] == 0
    assert props["study_search_radius"]["minimum"] == 0
    region_radius_prop = props["region_search_radius"]
    if "minimum" in region_radius_prop:
        assert region_radius_prop["minimum"] == 0
    else:
        any_of = region_radius_prop.get("anyOf", [])
        assert any_of and any_of[0]["minimum"] == 0
    assert "summary_models" in props
    assert props["coordinates"]["anyOf"][0]["type"] == "array"
    assert file_schema["additionalProperties"] is False
