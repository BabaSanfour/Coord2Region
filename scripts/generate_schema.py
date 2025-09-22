#!/usr/bin/env python3
"""Generate a JSON Schema for :class:`Coord2RegionConfig`."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


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


def main() -> None:
    """Write the generated schema to docs/static/schema.json."""
    module, coord2region_config = _load_coord2region_config()
    coord2region_config.model_rebuild(_types_namespace=module.__dict__)
    schema = coord2region_config.model_json_schema()
    # Remove root title/description for a cleaner UI schema and stable diffs.
    # Tests account for this by ignoring these fields when comparing.
    schema.pop("title", None)
    schema.pop("description", None)

    repo_root = Path(__file__).resolve().parents[1]
    output_path = repo_root / "docs" / "static" / "schema.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(schema, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
