import csv
import json
import os
from io import BytesIO
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from PIL import Image

from coord2region.pipeline import PipelineResult, _export_results, run_pipeline


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


@pytest.mark.unit
@patch("coord2region.pipeline.generate_summary", side_effect=["S1", "S2"])
@patch("coord2region.pipeline.get_studies_for_coordinate", return_value=[{"id": "1"}])
@patch("coord2region.pipeline.prepare_datasets", return_value={"Combined": object()})
@patch("coord2region.pipeline.AIModelInterface")
def test_run_pipeline_batch_coords(mock_ai, mock_prepare, mock_get, mock_summary):
    results = run_pipeline(
        inputs=[[0, 0, 0], [1, 1, 1]],
        input_type="coords",
        outputs=["summaries", "raw_studies"],
        brain_insights_kwargs={
            "use_atlases": False,
            "gemini_api_key": "key",
        },
    )
    assert [r.summary for r in results] == ["S1", "S2"]
    assert all(r.studies == [{"id": "1"}] for r in results)


@pytest.mark.unit
@patch("coord2region.pipeline.save_as_pdf")
@patch("coord2region.pipeline.generate_summary", return_value="SUM")
@patch("coord2region.pipeline.get_studies_for_coordinate", return_value=[])
@patch("coord2region.pipeline.prepare_datasets", return_value={"Combined": object()})
@patch("coord2region.pipeline.AIModelInterface")
def test_run_pipeline_export_pdf(
    mock_ai, mock_prepare, mock_get, mock_summary, mock_save_pdf, tmp_path
):
    out_file = tmp_path / "results.pdf"
    res = run_pipeline(
        inputs=[[0, 0, 0]],
        input_type="coords",
        outputs=["summaries"],
        output_format="pdf",
        output_path=str(out_file),
        brain_insights_kwargs={
            "use_atlases": False,
            "gemini_api_key": "key",
        },
    )
    assert res[0].summary == "SUM"
    mock_save_pdf.assert_called_once()


@pytest.mark.unit
@patch("coord2region.pipeline.generate_mni152_image")
def test_pipeline_nilearn_backend(mock_gen, tmp_path):
    buf = BytesIO()
    Image.new("RGB", (10, 10), color="white").save(buf, format="PNG")
    mock_gen.return_value = buf.getvalue()

    res = run_pipeline(
        inputs=[[0, 0, 0]],
        input_type="coords",
        outputs=["images"],
        image_backend="nilearn",
        brain_insights_kwargs={
            "use_atlases": False,
            "use_cached_dataset": False,
            "data_dir": str(tmp_path),
        },
    )
    path = res[0].images.get("nilearn")
    assert path and os.path.exists(path)


@pytest.mark.unit
@patch("coord2region.ai_model_interface.AIModelInterface.generate_image")
def test_pipeline_ai_watermark(mock_generate, tmp_path):
    buf = BytesIO()
    Image.new("RGB", (100, 50), color="black").save(buf, format="PNG")
    mock_generate.return_value = buf.getvalue()

    res = run_pipeline(
        inputs=[[0, 0, 0]],
        input_type="coords",
        outputs=["images"],
        image_backend="ai",
        brain_insights_kwargs={
            "use_atlases": False,
            "use_cached_dataset": False,
            "data_dir": str(tmp_path),
            "gemini_api_key": "key",
        },
    )

    path = res[0].images.get("ai")
    assert path and os.path.exists(path)
    arr = np.array(Image.open(path))
    bottom = arr[int(arr.shape[0] * 0.8) :, :, :]
    assert np.any(bottom > 0)


@pytest.mark.unit
def test_export_results_invalid_format(tmp_path):
    with pytest.raises(ValueError):
        _export_results([PipelineResult()], "xml", str(tmp_path / "out"))


@pytest.mark.unit
def test_export_results_csv(tmp_path):
    csv_path = tmp_path / "out" / "res.csv"
    _export_results([PipelineResult(summary="A")], "csv", str(csv_path))
    assert csv_path.exists()
    with open(csv_path, newline="", encoding="utf8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["summary"] == "A"


@pytest.mark.unit
def test_export_results_directory(tmp_path):
    out_dir = tmp_path / "batch"
    _export_results([PipelineResult(summary="B")], "directory", str(out_dir))
    assert (out_dir / "result_1" / "result.json").exists()
