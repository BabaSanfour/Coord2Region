import json
import subprocess
import sys
import types
from pathlib import Path
import textwrap

import pytest

sys.modules.setdefault('mne', types.ModuleType('mne'))
sys.modules.setdefault('mne.datasets', types.ModuleType('mne.datasets'))

from coord2region.cli import run_from_config

ROOT = Path(__file__).resolve().parent.parent


def _run(code: str):
    stub = textwrap.dedent(
        """
import sys
import types
sys.modules.setdefault('mne', types.ModuleType('mne'))
sys.modules.setdefault('mne.datasets', types.ModuleType('mne.datasets'))
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

def fake_run_pipeline(inputs, input_type, outputs, output_format, output_path, **kwargs):
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
