import importlib.util
from pathlib import Path
import pytest

# Load the module directly to avoid heavy package imports
SPEC = importlib.util.spec_from_file_location(
    "paths",
    Path(__file__).resolve().parents[1] / "coord2region" / "utils" / "paths.py",
)
paths = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(paths)
resolve_data_dir = paths.resolve_data_dir


@pytest.mark.unit
def test_resolve_data_dir_absolute(tmp_path):
    result = resolve_data_dir(str(tmp_path))
    assert result == tmp_path.resolve()


@pytest.mark.unit
def test_resolve_data_dir_relative(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    result = resolve_data_dir("relative")
    assert result == (tmp_path / "relative").resolve()


@pytest.mark.unit
def test_resolve_data_dir_invalid():
    with pytest.raises(ValueError):
        resolve_data_dir("invalid\x00path")

