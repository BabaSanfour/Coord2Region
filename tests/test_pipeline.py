import json
import os
from unittest.mock import AsyncMock, patch

import pytest

from coord2region.pipeline import run_pipeline


@pytest.mark.unit
@patch("coord2region.pipeline.generate_region_image", return_value=b"imgdata")
@patch("coord2region.pipeline.generate_summary", return_value="SUMMARY")
@patch(
    "coord2region.pipeline.get_studies_for_coordinate", return_value=[{"id": "1"}]
)
@patch("coord2region.pipeline.prepare_datasets", return_value={"Combined": object()})
@patch("coord2region.pipeline.AIModelInterface")
def test_run_pipeline_coords(
    mock_ai, mock_prepare, mock_get, mock_summary, mock_image, tmp_path
):
    out_file = tmp_path / "results.json"
    results = run_pipeline(
        inputs=[[0, 0, 0]],
        input_type="coords",
        outputs=["raw_studies", "summaries", "images"],
        output_format="json",
        output_path=str(out_file),
        brain_insights_kwargs={
            "use_atlases": False,
            "data_dir": str(tmp_path),
            "gemini_api_key": "key",
        },
    )

    assert results[0].studies == [{"id": "1"}]
    assert results[0].summary == "SUMMARY"
    assert results[0].image and os.path.exists(results[0].image)

    with open(out_file, "r", encoding="utf8") as f:
        exported = json.load(f)
    assert exported[0]["summary"] == "SUMMARY"


@pytest.mark.unit
@patch("coord2region.pipeline.generate_summary", return_value="SUMMARY")
@patch("coord2region.pipeline.AIModelInterface")
def test_run_pipeline_studies(mock_ai, mock_summary):
    study = {"id": "1"}
    results = run_pipeline(
        inputs=[study],
        input_type="studies",
        outputs=["summaries", "raw_studies"],
        brain_insights_kwargs={
            "use_atlases": False,
            "use_cached_dataset": False,
            "gemini_api_key": "key",
        },
    )

    assert results[0].studies == [study]
    assert results[0].summary == "SUMMARY"


@pytest.mark.unit
@patch(
    "coord2region.pipeline.get_studies_for_coordinate", return_value=[{"id": "1"}]
)
@patch("coord2region.pipeline.prepare_datasets", return_value={"Combined": object()})
@patch("coord2region.pipeline.AIModelInterface")
def test_run_pipeline_async(mock_ai, mock_prepare, mock_get):
    async_mock = AsyncMock(return_value="ASYNC")
    with patch(
        "coord2region.pipeline.generate_summary_async", new=async_mock
    ):
        progress_calls = []

        def cb(done, total, res):
            progress_calls.append((done, res.summary))

        results = run_pipeline(
            inputs=[[0, 0, 0], [1, 1, 1]],
            input_type="coords",
            outputs=["summaries"],
            brain_insights_kwargs={
                "use_atlases": False,
                "gemini_api_key": "key",
            },
            async_mode=True,
            progress_callback=cb,
        )

    assert [r.summary for r in results] == ["ASYNC", "ASYNC"]
    assert len(progress_calls) == 2
