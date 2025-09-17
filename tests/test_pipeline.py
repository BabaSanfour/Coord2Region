import csv
import json
import os
import pickle
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
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
        config={
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
        config={
            "use_atlases": False,
            "use_cached_dataset": False,
            "gemini_api_key": "key",
        },
    )

    assert results[0].studies == [study]
    assert results[0].summary == "SUMMARY"


@pytest.mark.unit
@patch("coord2region.pipeline.generate_summary", return_value="SUMMARY")
@patch(
    "coord2region.pipeline.get_studies_for_coordinate",
    return_value=[{"id": "1"}, {"id": "2"}],
)
@patch("coord2region.pipeline.prepare_datasets", return_value={"Mock": object()})
@patch("coord2region.pipeline.AIModelInterface")
def test_pipeline_study_config_controls(
    mock_ai, mock_prepare, mock_get, mock_summary
):
    mock_ai.return_value = object()

    results = run_pipeline(
        inputs=[[0, 0, 0]],
        input_type="coords",
        outputs=["summaries", "raw_studies"],
        config={
            "use_atlases": False,
            "gemini_api_key": "key",
            "study_radius": 7.5,
            "study_limit": 1,
            "study_sources": ["Mock"],
            "summary_model": "custom-model",
            "summary_type": "detailed",
            "summary_prompt_template": "Prompt",
            "summary_max_tokens": 42,
            "summary_cache_size": 2,
        },
    )

    assert results[0].studies == [{"id": "1"}]
    args, kwargs = mock_get.call_args
    assert pytest.approx(kwargs["radius"]) == 7.5
    assert kwargs["sources"] == ["Mock"]

    _, summary_kwargs = mock_summary.call_args
    assert summary_kwargs["model"] == "custom-model"
    assert summary_kwargs["summary_type"] == "detailed"
    assert summary_kwargs["prompt_template"] == "Prompt"
    assert summary_kwargs["max_tokens"] == 42
    assert summary_kwargs["cache_size"] == 2


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
            config={
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
        config={
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
        config={
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
        config={
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
        config={
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
def test_pipeline_both_backends(tmp_path):
    buf = BytesIO()
    Image.new("RGB", (10, 10), color="white").save(buf, format="PNG")
    ai_bytes = buf.getvalue()
    buf.seek(0)
    nilearn_bytes = buf.getvalue()

    with patch(
        "coord2region.pipeline.generate_region_image", return_value=ai_bytes
    ) as mock_ai, patch(
        "coord2region.pipeline.generate_mni152_image", return_value=nilearn_bytes
    ) as mock_nl, patch("coord2region.pipeline.AIModelInterface"):
        res = run_pipeline(
            inputs=[[0, 0, 0]],
            input_type="coords",
            outputs=["images"],
            image_backend="both",
            config={
                "use_atlases": False,
                "use_cached_dataset": False,
                "data_dir": str(tmp_path),
                "gemini_api_key": "k",
            },
        )

    imgs = res[0].images
    assert set(imgs.keys()) == {"ai", "nilearn"}
    mock_ai.assert_called_once()
    mock_nl.assert_called_once()


@pytest.mark.unit
def test_pipeline_async_both_backends(tmp_path):
    buf = BytesIO()
    Image.new("RGB", (1, 1), color="white").save(buf, format="PNG")
    img_bytes = buf.getvalue()

    with patch(
        "coord2region.pipeline.generate_region_image", return_value=img_bytes
    ), patch(
        "coord2region.pipeline.generate_mni152_image", return_value=img_bytes
    ), patch("coord2region.pipeline.AIModelInterface"):
        res = run_pipeline(
            inputs=[[0, 0, 0]],
            input_type="coords",
            outputs=["images"],
            image_backend="both",
            async_mode=True,
            config={
                "use_atlases": False,
                "use_cached_dataset": False,
                "data_dir": str(tmp_path),
                "gemini_api_key": "k",
            },
        )

    imgs = res[0].images
    assert set(imgs.keys()) == {"ai", "nilearn"}
    for path in imgs.values():
        assert path and os.path.exists(path)


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
def test_export_results_pickle(tmp_path):
    pkl_path = tmp_path / "res.pkl"
    res = PipelineResult(summary="A")
    _export_results([res], "pickle", str(pkl_path))
    assert pkl_path.exists()
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    assert data == [asdict(res)]


@pytest.mark.unit
def test_export_results_directory(tmp_path):
    out_dir = tmp_path / "batch"
    _export_results([PipelineResult(summary="B")], "directory", str(out_dir))
    assert (out_dir / "result_1" / "result.json").exists()


@pytest.mark.unit
def test_run_pipeline_invalid_input_type():
    with pytest.raises(ValueError):
        run_pipeline([1], "invalid", [], config={"use_atlases": False})


@pytest.mark.unit
def test_run_pipeline_invalid_output():
    with pytest.raises(ValueError):
        run_pipeline([[0, 0, 0]], "coords", ["bad"], config={"use_atlases": False})


@pytest.mark.unit
def test_run_pipeline_missing_output_path():
    with pytest.raises(ValueError):
        run_pipeline([[0, 0, 0]], "coords", ["summaries"], output_format="json", config={"use_atlases": False})


@pytest.mark.unit
def test_run_pipeline_invalid_image_backend():
    with pytest.raises(ValueError):
        run_pipeline([[0, 0, 0]], "coords", ["images"], image_backend="wrong", config={"use_atlases": False})


@pytest.mark.unit
@patch("coord2region.pipeline.AtlasFetcher")
@patch("coord2region.pipeline.AIModelInterface")
@patch("coord2region.pipeline.prepare_datasets", return_value={})
def test_run_pipeline_register_provider(_mock_prepare, mock_ai, _mock_fetcher):
    run_pipeline(
        inputs=[],
        input_type="coords",
        outputs=[],
        config={"providers": {"echo": {}}, "use_atlases": False},
    )
    mock_ai.assert_called_once_with()
    mock_ai.return_value.register_provider.assert_called_once_with("echo", **{})


@pytest.mark.unit
def test_run_pipeline_none_coord(tmp_path):
    results = run_pipeline(
        inputs=[None],
        input_type="coords",
        outputs=["region_labels"],
        config={
            "use_atlases": False,
            "use_cached_dataset": False,
            "data_dir": str(tmp_path),
        },
    )
    assert len(results) == 1
    res = results[0]
    assert res.coordinate is None
    assert res.region_labels == {}


@pytest.mark.unit
@patch("coord2region.pipeline.MultiAtlasMapper")
def test_run_pipeline_multiatlas_error(mock_multi, tmp_path):
    class RaisingMultiAtlas:
        def __init__(self, *_args, **_kwargs):
            pass

        def batch_mni_to_region_names(self, coords):
            raise RuntimeError("boom")

    mock_multi.side_effect = lambda *a, **k: RaisingMultiAtlas()
    results = run_pipeline(
        inputs=[[0, 0, 0]],
        input_type="coords",
        outputs=["region_labels"],
        config={
            "use_atlases": True,
            "use_cached_dataset": False,
            "data_dir": str(tmp_path),
            "atlas_names": ["dummy"],
        },
    )
    assert results[0].region_labels == {}


@pytest.mark.unit
@patch("coord2region.pipeline.MultiAtlasMapper")
def test_run_pipeline_atlas_labels_success(mock_multi, tmp_path):
    captured = {}

    class DummyMulti:
        def __init__(self, base_dir, atlases):
            captured["base_dir"] = base_dir
            captured["atlases"] = atlases

        def batch_mni_to_region_names(self, coords):
            captured["coords"] = coords
            return {"custom": ["Region"]}

    mock_multi.side_effect = lambda *args, **kwargs: DummyMulti(*args, **kwargs)

    results = run_pipeline(
        inputs=[[0, 0, 0]],
        input_type="coords",
        outputs=["region_labels"],
        config={
            "use_atlases": True,
            "use_cached_dataset": False,
            "data_dir": str(tmp_path),
            "atlas_names": ["custom"],
        },
    )

    assert results[0].region_labels == {"custom": "Region"}
    assert captured["atlases"] == {"custom": {}}
    assert captured["coords"] == [[0, 0, 0]]


@pytest.mark.unit
@patch("coord2region.pipeline.generate_summary", return_value="SUMMARY")
@patch("coord2region.pipeline.MultiAtlasMapper")
@patch("coord2region.pipeline.AIModelInterface")
def test_run_pipeline_region_names_to_coords(
    mock_ai, mock_multi, mock_summary, tmp_path
):
    class DummyMulti:
        def __init__(self, *_args, **_kwargs):
            pass

        def batch_region_name_to_mni(self, names):
            assert names == ["Region"]
            return {"custom": [np.array([1.0, 2.0, 3.0])]}

        def batch_mni_to_region_names(self, coords):
            assert coords == [[1.0, 2.0, 3.0]]
            return {"custom": ["Resolved"]}

    mock_multi.side_effect = lambda *a, **k: DummyMulti()

    results = run_pipeline(
        inputs=["Region"],
        input_type="region_names",
        outputs=["region_labels", "summaries"],
        config={
            "use_atlases": True,
            "use_cached_dataset": False,
            "data_dir": str(tmp_path),
            "gemini_api_key": "key",
        },
    )

    res = results[0]
    assert res.coordinate == [1.0, 2.0, 3.0]
    assert res.region_labels == {"custom": "Resolved"}
    mock_summary.assert_called_once()


@pytest.mark.unit
@patch("coord2region.pipeline.save_as_csv")
def test_run_pipeline_relative_output_path(mock_save_csv, tmp_path):
    run_pipeline(
        inputs=[[0, 0, 0]],
        input_type="coords",
        outputs=[],
        output_format="csv",
        output_path="nested/out.csv",
        config={
            "use_atlases": False,
            "use_cached_dataset": False,
            "data_dir": str(tmp_path),
        },
    )
    saved_path = Path(mock_save_csv.call_args[0][1])
    assert saved_path.parent.parent.name == "results"
