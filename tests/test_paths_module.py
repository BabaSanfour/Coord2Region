from pathlib import Path

import pytest

from coord2region.paths import get_data_directory


@pytest.mark.unit
def test_get_data_directory_creates_and_returns_absolute(tmp_path):
    data_dir = tmp_path / "data"
    result = get_data_directory(str(data_dir))

    path = Path(result)
    assert path.is_dir()
    assert path.is_absolute()


@pytest.mark.unit
def test_get_data_directory_default_location(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    result = get_data_directory(None)

    expected = tmp_path / "coord2region"
    path = Path(result)
    assert path == expected
    assert path.is_dir()
    assert path.is_absolute()
