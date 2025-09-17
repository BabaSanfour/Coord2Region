import json
import subprocess
import sys
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parent.parent


def _run(code: str):
    script = textwrap.dedent(code)
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
