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
resolve_working_directory = paths.resolve_working_directory


@pytest.mark.unit
def test_resolve_working_directory_absolute(tmp_path):
    result = resolve_working_directory(str(tmp_path))
    assert result == tmp_path.resolve()


@pytest.mark.unit
def test_resolve_working_directory_relative(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    result = resolve_working_directory("relative")
    assert result == (tmp_path / "relative").resolve()


@pytest.mark.unit
def test_resolve_working_directory_invalid():
    with pytest.raises(ValueError):
        resolve_working_directory("invalid\x00path")
