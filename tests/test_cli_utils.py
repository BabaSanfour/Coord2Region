import argparse
import csv

from unittest.mock import patch
import pytest

from coord2region.cli import (
    _parse_coord,
    _load_coords_file,
    _batch,
    _collect_kwargs,
    main,
)


@pytest.mark.unit
def test_parse_coord_invalid_length():
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_coord("1,2")


@pytest.mark.unit
def test_parse_coord_non_numeric():
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_coord("1,2,a")


@pytest.mark.unit
def test_load_coords_file_invalid_columns(tmp_path):
    path = tmp_path / "coords.csv"
    with open(path, "w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow([1, 2])
    with pytest.raises(argparse.ArgumentTypeError):
        _load_coords_file(str(path))


@pytest.mark.unit
def test_batch_size_zero():
    seq = [1, 2, 3]
    assert list(_batch(seq, 0)) == [seq]


@pytest.mark.unit
def test_collect_kwargs():
    args = argparse.Namespace(gemini_api_key="g", openrouter_api_key=None, image_model="m")
    assert _collect_kwargs(args) == {"gemini_api_key": "g", "image_model": "m"}


@pytest.mark.unit
@patch("coord2region.cli.run_from_config")
def test_main_config_invokes_run(mock_run, tmp_path):
    cfg = tmp_path / "cfg.yml"
    cfg.write_text("inputs: []\n")
    main(["--config", str(cfg)])
    mock_run.assert_called_once_with(str(cfg))


@pytest.mark.unit
@patch("coord2region.cli.run_pipeline")
def test_main_no_coords_error(mock_run):
    with pytest.raises(SystemExit):
        main(["coords-to-atlas"])
    mock_run.assert_not_called()

