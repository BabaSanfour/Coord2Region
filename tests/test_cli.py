import json
import subprocess
import sys
import types
from pathlib import Path
import textwrap
import shlex

import csv
import argparse
from unittest.mock import patch
import pandas as pd

from coord2region.cli import (
    _parse_coord,
    _parse_coords_tokens,
    _load_coords_file,
    _batch,
    _collect_kwargs,
    _atlas_source_from_value,
    _inputs_to_tokens,
    _commands_from_config,
    run_from_config,
    main,
)


import pytest

ROOT = Path(__file__).resolve().parent.parent


def _run(code: str):
    stub = textwrap.dedent(
        """
import sys
import types
_mne_module = sys.modules.setdefault('mne', types.ModuleType('mne'))
_datasets_module = sys.modules.setdefault('mne.datasets', types.ModuleType('mne.datasets'))
if not hasattr(_mne_module, 'datasets'):
    _mne_module.datasets = _datasets_module
if not hasattr(_datasets_module, 'fetch_fsaverage'):
    def _fetch_fsaverage_stub(*args, **kwargs):
        return None
    _datasets_module.fetch_fsaverage = _fetch_fsaverage_stub
"""
    )
    script = stub + textwrap.dedent(code)
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, cwd=ROOT
    )


def test_coords_to_summary_cli():
    code = """
from unittest.mock import patch
from coord2region.cli import main
from coord2region.pipeline import PipelineResult
with patch("coord2region.cli.run_pipeline", return_value=[PipelineResult(coordinate=[0,0,0], summary="SUM", region_labels={}, studies=[], image=None)]):
    main(["coords-to-summary", "0,0,0"])
"""
    result = _run(code)
    assert result.returncode == 0
    out = json.loads(result.stdout)
    assert out[0]["summary"] == "SUM"


def test_run_from_config_cli(tmp_path):
    cfg = tmp_path / "cfg.yml"
    cfg.write_text(
        """
inputs:
  - [0,0,0]
input_type: coords
outputs: ["summaries"]
gemini_api_key: TEST
""",
        encoding="utf8",
    )
    code = f"""
from unittest.mock import patch
from coord2region.cli import main
from coord2region.pipeline import PipelineResult
with patch("coord2region.cli.run_pipeline", return_value=[PipelineResult(coordinate=[0,0,0], summary="CFG", region_labels={{}}, studies=[], image=None)]):
    main(["run", "--config", r"{cfg}"])
"""
    result = _run(code)
    assert result.returncode == 0
    out = json.loads(result.stdout)
    assert out[0]["summary"] == "CFG"


def test_cli_version_flag_reports_package_version():
    code = """
from coord2region.cli import main
main(["--version"])
"""
    result = _run(code)
    assert result.returncode == 0
    assert result.stdout.strip().startswith("coord2region ")


def test_cli_list_atlases_flag_outputs_names(capsys):
    with patch("coord2region.cli.AtlasFetcher") as mock_fetcher:
        mock_fetcher.return_value.list_available_atlases.return_value = [
            "aal",
            "harvard-oxford",
        ]
        main(["--list-atlases"])
    captured = capsys.readouterr()
    assert captured.out.strip().splitlines() == ["aal", "harvard-oxford"]


def test_run_from_config_validation_error(tmp_path, capsys):
    cfg = tmp_path / "cfg.yml"
    cfg.write_text(
        """
inputs:
  - [0,0,0]
coordinates:
  - [1,1,1]
input_type: coords
outputs: ["region_labels"]
""",
        encoding="utf8",
    )

    with pytest.raises(SystemExit):
        run_from_config(str(cfg))

    captured = capsys.readouterr()
    assert "Specify coordinates either inline" in captured.err


def test_cli_batch_processing():
    code = """
from unittest.mock import patch
from coord2region.cli import main
from coord2region.pipeline import PipelineResult
calls = []

def fake_run_pipeline(inputs, *a, **k):
    calls.append(inputs)
    return [PipelineResult(coordinate=i, summary=None, region_labels={}, studies=[], image=None) for i in inputs]

with patch("coord2region.cli.run_pipeline", side_effect=fake_run_pipeline):
    main(["coords-to-summary", "0,0,0", "1,1,1", "--batch-size", "1"])
print(len(calls))
"""
    result = _run(code)
    assert result.returncode == 0
    lines = result.stdout.strip().splitlines()
    assert lines[-1] == "2"


def test_run_from_config_cli_dry_run(tmp_path):
    cfg = tmp_path / "cfg.yml"
    cfg.write_text(
        """
inputs:
  - [0,0,0]
input_type: coords
outputs: ["region_labels"]
""",
        encoding="utf8",
    )
    code = f"""
from coord2region.cli import main
main(["run", "--config", r"{cfg}", "--dry-run"])
"""
    result = _run(code)
    assert result.returncode == 0
    lines = [line for line in result.stdout.strip().splitlines() if line]
    assert lines == [
        "coord2region coords-to-atlas 0 0 0",
    ]


def test_cli_custom_atlas_sources(tmp_path):
    code = """
import json
from unittest.mock import patch
from coord2region.cli import main
from coord2region.pipeline import PipelineResult

captured = {}

def fake_run_pipeline(inputs, input_type, outputs, output_format, output_name, **kwargs):
    captured['config'] = kwargs.get('config', {})
    return [PipelineResult(coordinate=inputs[0], summary=None, region_labels={}, studies=[], image=None)]

with patch("coord2region.cli.run_pipeline", side_effect=fake_run_pipeline):
    main([
        "coords-to-atlas",
        "0,0,0",
        "--atlas",
        "https://example.com/custom.nii.gz",
    ])

print(json.dumps(captured['config'].get('atlas_names')))
print(json.dumps(captured['config'].get('atlas_configs', {}).get('https://example.com/custom.nii.gz', {})))
"""
    result = _run(code)
    assert result.returncode == 0
    parsed = []
    for raw in result.stdout.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            parsed.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    assert parsed, "Expected JSON output from CLI stub"

    atlas_names = next((item for item in parsed if isinstance(item, list)), [])
    atlas_config = next(
        (item for item in parsed if isinstance(item, dict) and "atlas_url" in item),
        {}
    )

    assert "https://example.com/custom.nii.gz" in atlas_names
    assert atlas_config.get("atlas_url") == "https://example.com/custom.nii.gz"


@pytest.mark.unit
def test_atlas_source_from_value_variants(tmp_path):
    assert _atlas_source_from_value("https://example.org/file.nii.gz") == {
        "atlas_url": "https://example.org/file.nii.gz"
    }
    local_path = tmp_path / "atlas.nii.gz"
    local_path.write_text("", encoding="utf8")
    assert _atlas_source_from_value(str(local_path)) == {"atlas_file": str(local_path)}
    assert _atlas_source_from_value("~/atlas.nii.gz") == {
        "atlas_file": "~/atlas.nii.gz"
    }
    assert _atlas_source_from_value("C:\\atlas.nii.gz") == {
        "atlas_file": "C:\\atlas.nii.gz"
    }


@pytest.mark.unit
def test_inputs_to_tokens_formats_values_and_errors():
    coord_tokens = _inputs_to_tokens("coords", [[1, 2.5, 3.0], (4.0, 5, 6)])
    assert coord_tokens == ["1", "2.5", "3", "4", "5", "6"]
    names = _inputs_to_tokens("region_names", ["A", "B"])
    assert names == ["A", "B"]
    with pytest.raises(ValueError):
        _inputs_to_tokens("unknown", [])


@pytest.mark.unit
def test_commands_from_config_coords_to_insights_includes_flags():
    runtime = {
        "input_type": "coords",
        "inputs": [[0, 0, 0]],
        "outputs": ["region_labels", "raw_studies", "summaries", "images"],
        "output_format": "json",
        "output_name": "results.json",
        "image_backend": "both",
        "config": {
            "working_directory": "/tmp/work",
            "gemini_api_key": "KEY",
            "huggingface_api_key": "HF",
            "sources": ["neurosynth"],
            "email_for_abstracts": "user@example.com",
            "atlas_names": ["aal"],
            "atlas_configs": {"aal": {"atlas_url": "https://example/aal.nii.gz"}},
            "image_model": "hf/model",
            "image_prompt_type": "functional",
            "image_custom_prompt": "focus {coordinate}",
        },
    }
    commands = _commands_from_config(runtime)
    assert len(commands) == 1
    tokens = shlex.split(commands[0])
    assert tokens[:2] == ["coord2region", "coords-to-insights"]
    assert "--atlas" in tokens and tokens[tokens.index("--atlas") + 1] == "aal"
    assert "--output-format" in tokens
    assert "--image-prompt-type" in tokens
    assert "--image-custom-prompt" in tokens
    assert "--gemini-api-key" in tokens
    assert "--huggingface-api-key" in tokens
    assert "--image-model" in tokens


@pytest.mark.unit
def test_commands_from_config_region_single_atlas_required():
    runtime = {
        "input_type": "region_names",
        "inputs": ["Region"],
        "outputs": ["mni_coordinates", "raw_studies"],
        "config": {
            "atlas_names": ["aal", "juelich"],
        },
    }
    with pytest.raises(ValueError):
        _commands_from_config(runtime)

    runtime["config"]["atlas_names"] = ["aal"]
    commands = _commands_from_config(runtime)
    tokens = shlex.split(commands[0])
    assert tokens[:2] == ["coord2region", "region-to-study"]
    assert "--atlas" in tokens


def test_collect_kwargs_comprehensive(tmp_path):
    ns = argparse.Namespace(
        gemini_api_key="G",
        openrouter_api_key="R",
        openai_api_key="O",
        anthropic_api_key="A",
        huggingface_api_key="H",
        image_model="vision",
        image_prompt_type="custom",
        image_custom_prompt="Prompt",
        working_directory=str(tmp_path / "work"),
        email_for_abstracts="user@example.com",
        sources=["SourceA", "SourceA,SourceB", "sourceb"],
        atlas_names=[
            "aal",
            "~/relative/path.nii.gz",
            "https://example.com/custom-atlas.nii.gz",
        ],
        atlas_urls=["aal=https://mirror.example.com/aal.nii.gz"],
        atlas_files=["aal=/tmp/atlas.nii.gz", "custom=/data/atlas2.nii.gz"],
    )

    kwargs = _collect_kwargs(ns)

    assert kwargs["gemini_api_key"] == "G"
    assert kwargs["openrouter_api_key"] == "R"
    assert kwargs["openai_api_key"] == "O"
    assert kwargs["anthropic_api_key"] == "A"
    assert kwargs["huggingface_api_key"] == "H"
    assert kwargs["image_model"] == "vision"
    assert kwargs["image_prompt_type"] == "custom"
    assert kwargs["image_custom_prompt"] == "Prompt"
    assert kwargs["working_directory"] == str(tmp_path / "work")
    assert kwargs["email_for_abstracts"] == "user@example.com"

    # Sources are de-duplicated but retain order
    assert kwargs["sources"] == ["SourceA", "SourceB", "sourceb"]

    atlas_names = kwargs["atlas_names"]
    assert "aal" in atlas_names
    assert "~/relative/path.nii.gz" in atlas_names
    assert "https://example.com/custom-atlas.nii.gz" in atlas_names

    configs = kwargs["atlas_configs"]
    assert configs["aal"]["atlas_url"] == "https://mirror.example.com/aal.nii.gz"
    assert configs["aal"]["atlas_file"] == "/tmp/atlas.nii.gz"
    assert configs["custom"]["atlas_file"] == "/data/atlas2.nii.gz"
    assert configs["https://example.com/custom-atlas.nii.gz"]["atlas_url"] == (
        "https://example.com/custom-atlas.nii.gz"
    )
    assert configs["~/relative/path.nii.gz"]["atlas_file"] == "~/relative/path.nii.gz"


def test_collect_kwargs_atlas_url_validation():
    ns = argparse.Namespace(atlas_urls=["invalid"], atlas_names=None, atlas_files=None)
    with pytest.raises(argparse.ArgumentTypeError):
        _collect_kwargs(ns)


def test_collect_kwargs_atlas_file_validation():
    ns = argparse.Namespace(atlas_urls=None, atlas_names=None, atlas_files=["broken"])
    with pytest.raises(argparse.ArgumentTypeError):
        _collect_kwargs(ns)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("", None),
        ("https://example.com/atlas.nii.gz", {"atlas_url": "https://example.com/atlas.nii.gz"}),
        ("~/atlas.nii.gz", {"atlas_file": "~/atlas.nii.gz"}),
        ("relative/path.nii.gz", {"atlas_file": "relative/path.nii.gz"}),
        ("C:/atlas.nii.gz", {"atlas_file": "C:/atlas.nii.gz"}),
        ("aal", None),
    ],
)
def test_atlas_source_from_value(value, expected):
    assert _atlas_source_from_value(value) == expected


def test_batch_handles_large_size():
    seq = [1, 2, 3]
    assert list(_batch(seq, 0)) == [seq]
    assert list(_batch(seq, 5)) == [seq]
    assert list(_batch(seq, 2)) == [[1, 2], [3]]



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
        working_directory="/tmp/data",
        email_for_abstracts="person@example.com",
    )
    kwargs = _collect_kwargs(args)
    assert kwargs == {
        "atlas_names": ["harvard-oxford", "juelich", "aal"],
        "working_directory": "/tmp/data",
        "email_for_abstracts": "person@example.com",
    }


@pytest.mark.unit
@patch("coord2region.cli.run_from_config")
def test_main_config_invokes_run(mock_run, tmp_path):
    cfg = tmp_path / "cfg.yml"
    cfg.write_text("inputs: []\n")
    main(["run", "--config", str(cfg)])
    mock_run.assert_called_once_with(str(cfg), dry_run=False)


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
            "--image-prompt-type",
            "schematic",
            "--image-custom-prompt",
            "Draw networks",
        ]
    )
    args, kwargs = mock_run.call_args
    assert args[0] == [[0.0, 0.0, 0.0]]
    assert args[2] == ["region_labels", "raw_studies", "images"]
    assert kwargs["image_backend"] == "both"
    assert kwargs["config"]["image_model"] == "custom"
    assert kwargs["config"]["image_prompt_type"] == "schematic"
    assert kwargs["config"]["image_custom_prompt"] == "Draw networks"


@pytest.mark.unit
@patch("coord2region.cli._print_results")
@patch("coord2region.cli.run_pipeline", return_value=[])
def test_main_coords_to_study(mock_run, mock_print):
    main([
        "coords-to-study",
        "0",
        "0",
        "0",
        "--sources",
        "neurosynth",
    ])
    args, kwargs = mock_run.call_args
    assert args[2] == ["region_labels", "raw_studies"]
    assert kwargs["config"]["sources"] == ["neurosynth"]


@pytest.mark.unit
@patch("coord2region.cli._print_results")
@patch("coord2region.cli.run_pipeline", return_value=[])
def test_main_coords_to_insights(mock_run, mock_print):
    main(
        [
            "coords-to-insights",
            "1",
            "2",
            "3",
            "--atlas",
            "aal",
            "--huggingface-api-key",
            "HF",
            "--gemini-api-key",
            "G",
        ]
    )
    args, kwargs = mock_run.call_args
    assert args[2] == [
        "region_labels",
        "raw_studies",
        "summaries",
        "images",
    ]
    assert kwargs["config"]["huggingface_api_key"] == "HF"
    assert kwargs["config"]["gemini_api_key"] == "G"


@pytest.mark.unit
@patch("coord2region.cli._print_results")
@patch("coord2region.cli.run_pipeline", return_value=[])
def test_main_region_to_coords_batches(mock_run, mock_print):
    main([
        "region-to-coords",
        "Region A",
        "Region B",
        "--batch-size",
        "1",
        "--atlas",
        "aal",
    ])
    calls = mock_run.call_args_list
    assert len(calls) == 2
    assert calls[0][0][0] == ["Region A"]
    assert calls[0][0][1] == "region_names"
    assert calls[1][0][0] == ["Region B"]


@pytest.mark.unit
def test_region_to_coords_requires_single_atlas():
    with pytest.raises(SystemExit):
        main([
            "region-to-coords",
            "Region A",
            "--atlas",
            "aal",
            "--atlas",
            "juelich",
        ])


@pytest.mark.unit
@patch("coord2region.cli._print_results")
@patch("coord2region.cli.run_pipeline", return_value=[])
def test_region_to_study_outputs(mock_run, mock_print):
    main([
        "region-to-study",
        "Region A",
        "--atlas",
        "aal",
        "--sources",
        "neurosynth",
    ])
    args, kwargs = mock_run.call_args
    assert args[2] == ["mni_coordinates", "raw_studies"]
    assert kwargs["config"]["atlas_names"] == ["aal"]


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
output_name: out.json
config:
  atlas_names: ["aal"]
""",
        encoding="utf8",
    )
    run_from_config(str(cfg))
    args, kwargs = mock_run.call_args
    assert args == ()
    assert kwargs["inputs"] == [[1, 2, 3]]
    assert kwargs["input_type"] == "coords"
    assert kwargs["outputs"] == ["summaries"]
    assert kwargs["output_format"] == "json"
    assert kwargs["output_name"] == "out.json"
    assert kwargs["image_backend"] == "ai"
    assert kwargs["config"]["atlas_names"] == ["aal"]


@pytest.mark.unit
def test_run_from_config_dry_run_outputs_commands(tmp_path, capsys):
    cfg = tmp_path / "cfg.yml"
    cfg.write_text(
        """
inputs:
  - [1, 2, 3]
  - [4, 5, 6]
input_type: coords
outputs: [region_labels, raw_studies, summaries]
config:
  atlas_names: [aal, juelich]
  working_directory: /tmp/data
""",
        encoding="utf8",
    )

    run_from_config(str(cfg), dry_run=True)
    out = capsys.readouterr().out.strip().splitlines()
    assert out == [
        "coord2region coords-to-summary 1 2 3 4 5 6 --working-directory /tmp/data --atlas aal --atlas juelich",
    ]


def test_run_from_config_dry_run_includes_image_prompt(tmp_path, capsys):
    cfg = tmp_path / "cfg.yml"
    cfg.write_text(
        """
        inputs:
        - [1, 2, 3]
        input_type: coords
        outputs: [region_labels, raw_studies, images]
        config:
        atlas_names: [aal]
        image_prompt_type: functional
        image_custom_prompt: Draw atlas context
        """,
        encoding="utf8",
    )

    run_from_config(str(cfg), dry_run=True)
    out = capsys.readouterr().out.strip().splitlines()
    # Should be coords-to-image and include image flags
    assert out and out[0].startswith("coord2region coords-to-image 1 2 3 ")
    assert "--image-prompt-type functional" in out[0]
    assert "--image-custom-prompt 'Draw atlas context'" in out[0] or "--image-custom-prompt Draw atlas context" in out[0]
