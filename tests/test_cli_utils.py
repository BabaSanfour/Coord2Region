import argparse
import csv

from unittest.mock import patch
import pandas as pd
import pytest

from coord2region.cli import (
    _parse_coord,
    _parse_coords_tokens,
    _load_coords_file,
    _batch,
    _collect_kwargs,
    run_from_config,
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
def test_parse_coords_tokens_numeric_grouping():
    coords = _parse_coords_tokens(["1", "2", "3", "4", "5", "6"])
    assert coords == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


@pytest.mark.unit
def test_parse_coords_tokens_fallback_strings():
    coords = _parse_coords_tokens(["1,2,3", "4 5 6"])
    assert coords == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


@pytest.mark.unit
def test_parse_coords_tokens_empty():
    assert _parse_coords_tokens([]) == []


@pytest.mark.unit
def test_load_coords_file_invalid_columns(tmp_path):
    path = tmp_path / "coords.csv"
    with open(path, "w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow([1, 2])
    with pytest.raises(argparse.ArgumentTypeError):
        _load_coords_file(str(path))


@pytest.mark.unit
def test_load_coords_file_csv_success(tmp_path):
    path = tmp_path / "coords.csv"
    pd.DataFrame({"x": [1, 4], "y": [2, 5], "z": [3, 6]}).to_csv(
        path, index=False
    )
    coords = _load_coords_file(str(path))
    assert coords == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


@pytest.mark.unit
def test_load_coords_file_excel_branch(monkeypatch):
    df = pd.DataFrame([[7, 8, 9]])

    def fake_read_excel(path):
        return df

    monkeypatch.setattr("coord2region.cli.pd.read_excel", fake_read_excel)
    coords = _load_coords_file("dummy.xlsx")
    assert coords == [[7.0, 8.0, 9.0]]


@pytest.mark.unit
def test_batch_size_zero():
    seq = [1, 2, 3]
    assert list(_batch(seq, 0)) == [seq]


@pytest.mark.unit
def test_batch_chunks():
    seq = [1, 2, 3, 4, 5]
    assert list(_batch(seq, 2)) == [[1, 2], [3, 4], [5]]


@pytest.mark.unit
def test_collect_kwargs():
    args = argparse.Namespace(gemini_api_key="g", openrouter_api_key=None, image_model="m")
    assert _collect_kwargs(args) == {"gemini_api_key": "g", "image_model": "m"}


@pytest.mark.unit
def test_collect_kwargs_with_atlas_names():
    args = argparse.Namespace(
        gemini_api_key=None,
        atlas_names=["harvard-oxford, juelich", " aal"],
        use_atlases=0,
        data_dir="/tmp/data",
        email_for_abstracts="person@example.com",
    )
    kwargs = _collect_kwargs(args)
    assert kwargs == {
        "atlas_names": ["harvard-oxford", "juelich", "aal"],
        "use_atlases": False,
        "data_dir": "/tmp/data",
        "email_for_abstracts": "person@example.com",
    }


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


@pytest.mark.unit
@patch("coord2region.cli._print_results")
@patch("coord2region.cli.run_pipeline", return_value=[])
@patch("coord2region.cli._load_coords_file", return_value=[[10.0, 20.0, 30.0]])
def test_main_coords_to_atlas_pipeline_call(mock_load, mock_run, mock_print, tmp_path):
    path = tmp_path / "points.csv"
    main(
        [
            "coords-to-atlas",
            "1",
            "2",
            "3",
            "--coords-file",
            str(path),
            "--batch-size",
            "2",
            "--atlas",
            "aal,juelich",
        ]
    )
    args, kwargs = mock_run.call_args
    assert args[0] == [[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]]
    assert args[1] == "coords"
    assert args[2] == ["region_labels"]
    assert kwargs["config"]["atlas_names"] == ["aal", "juelich"]


@pytest.mark.unit
@patch("coord2region.cli._print_results")
@patch("coord2region.cli.run_pipeline", return_value=[])
def test_main_coords_to_image_backend(mock_run, mock_print):
    main(
        [
            "coords-to-image",
            "0",
            "0",
            "0",
            "--image-backend",
            "both",
            "--image-model",
            "custom",
        ]
    )
    args, kwargs = mock_run.call_args
    assert args[0] == [[0.0, 0.0, 0.0]]
    assert kwargs["image_backend"] == "both"
    assert kwargs["config"]["image_model"] == "custom"


@pytest.mark.unit
@patch("coord2region.cli._print_results")
@patch("coord2region.cli.run_pipeline", return_value=[])
def test_main_region_to_coords_batches(mock_run, mock_print):
    main(["region-to-coords", "Region A", "Region B", "--batch-size", "1"])
    calls = mock_run.call_args_list
    assert len(calls) == 2
    assert calls[0][0][0] == ["Region A"]
    assert calls[0][0][1] == "region_names"
    assert calls[1][0][0] == ["Region B"]


@pytest.mark.unit
@patch("coord2region.cli._load_coords_file", side_effect=FileNotFoundError)
@patch("coord2region.cli.run_pipeline")
def test_main_coords_to_summary_missing_file(mock_run, mock_load):
    with pytest.raises(FileNotFoundError):
        main(["coords-to-summary", "--coords-file", "missing.csv"])
    mock_run.assert_not_called()
    mock_load.assert_called_once()


@pytest.mark.unit
@patch("coord2region.cli.run_pipeline", return_value=[])
def test_run_from_config_passes_values(mock_run, tmp_path):
    cfg = tmp_path / "cfg.yml"
    cfg.write_text(
        """
inputs:
  - [1, 2, 3]
input_type: coords
outputs: ["summaries"]
output_format: json
output_path: out.json
config:
  atlas_names: ["aal"]
""",
        encoding="utf8",
    )
    run_from_config(str(cfg))
    args, kwargs = mock_run.call_args
    assert args == ([[1, 2, 3]], "coords", ["summaries"], "json", "out.json")
    assert kwargs == {"config": {"atlas_names": ["aal"]}}
