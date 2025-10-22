import csv
import json
import os
import pickle
from dataclasses import asdict
from io import BytesIO
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from PIL import Image

from coord2region.pipeline import (
    PipelineResult,
    _export_results,
    _get_summary_models,
    _normalize_model_list,
    run_pipeline,
)


@pytest.mark.unit
@patch("coord2region.pipeline.generate_region_image", return_value=b"imgdata")
@patch("coord2region.pipeline.generate_summary", return_value="SUMMARY")
@patch(
    "coord2region.pipeline.get_studies_for_coordinate", return_value=[{"id": "1"}]
)
@patch("coord2region.pipeline.prepare_datasets", return_value={"Combined": object()})
@patch("coord2region.pipeline.MultiAtlasMapper")
@patch("coord2region.pipeline.AIModelInterface")
def test_run_pipeline_coords(
    mock_ai, mock_multi, mock_prepare, mock_get, mock_summary, mock_image, tmp_path
):
    mock_multi.return_value.batch_mni_to_region_names.return_value = {}
    mock_multi.return_value.batch_region_name_to_mni.return_value = {}
    output_name = "results.json"
    results = run_pipeline(
        inputs=[[0, 0, 0]],
        input_type="coords",
        outputs=["raw_studies", "summaries", "images"],
        output_format="json",
        output_name=output_name,
        config={
            "working_directory": str(tmp_path),
            "gemini_api_key": "key",
        },
    )

    assert results[0].studies == [{"id": "1"}]
    assert results[0].summary == "SUMMARY"
    assert results[0].summaries == {"gpt-4o-mini": "SUMMARY"}
    assert results[0].image and os.path.exists(results[0].image)

    export_path = tmp_path / "results" / output_name
    with open(export_path, "r", encoding="utf8") as f:
        exported = json.load(f)
    assert exported[0]["summary"] == "SUMMARY"
    assert exported[0]["summaries"] == {"gpt-4o-mini": "SUMMARY"}


    # Note: input_type='studies' is no longer supported


@pytest.mark.unit
@patch("coord2region.pipeline.generate_summary", return_value="SUMMARY")
@patch(
    "coord2region.pipeline.get_studies_for_coordinate",
    return_value=[{"id": "1"}, {"id": "2"}],
)
@patch("coord2region.pipeline.prepare_datasets", return_value={"Mock": object()})
@patch("coord2region.pipeline.MultiAtlasMapper")
@patch("coord2region.pipeline.AIModelInterface")
def test_pipeline_study_config_controls(
    mock_ai, mock_multi, mock_prepare, mock_get, mock_summary
):
    mock_ai.return_value = object()
    mock_multi.return_value.batch_mni_to_region_names.return_value = {}
    mock_multi.return_value.batch_region_name_to_mni.return_value = {}

    results = run_pipeline(
        inputs=[[0, 0, 0]],
        input_type="coords",
        outputs=["summaries", "raw_studies"],
        config={
            "gemini_api_key": "key",
            "study_search_radius": 7.5,
            "sources": ["Mock"],
            "summary_models": ["custom-model"],
            "prompt_type": "custom",
            "custom_prompt": "Prompt",
            "summary_max_tokens": 42,
        },
    )

    assert results[0].studies == [{"id": "1"}, {"id": "2"}]
    args, kwargs = mock_get.call_args
    assert pytest.approx(kwargs["radius"]) == 7.5
    assert kwargs["sources"] == ["Mock"]

    first_call_kwargs = mock_summary.call_args_list[0][1]
    assert first_call_kwargs["model"] == "custom-model"
    assert first_call_kwargs["prompt_type"] == "custom"
    assert first_call_kwargs["custom_prompt"] == "Prompt"
    assert first_call_kwargs["max_tokens"] == 42
    assert results[0].summaries == {"custom-model": "SUMMARY"}


@pytest.mark.unit
@patch("coord2region.pipeline.generate_summary", side_effect=["S-A", "S-B"])
@patch(
    "coord2region.pipeline.get_studies_for_coordinate", return_value=[{"id": "1"}]
)
@patch("coord2region.pipeline.prepare_datasets", return_value={"Combined": object()})
@patch("coord2region.pipeline.MultiAtlasMapper")
@patch("coord2region.pipeline.AIModelInterface")
def test_run_pipeline_multiple_summary_models(
    mock_ai, mock_multi, mock_prepare, mock_get, mock_summary
):
    mock_multi.return_value.batch_mni_to_region_names.return_value = {}
    mock_multi.return_value.batch_region_name_to_mni.return_value = {}

    results = run_pipeline(
        inputs=[[0, 0, 0]],
        input_type="coords",
        outputs=["summaries"],
        config={
            "gemini_api_key": "key",
            "summary_models": ["model-a", "model-b"],
        },
    )

    assert results[0].summary == "S-A"
    assert results[0].summaries == {"model-a": "S-A", "model-b": "S-B"}
    assert mock_summary.call_args_list[0][1]["model"] == "model-a"
    assert mock_summary.call_args_list[1][1]["model"] == "model-b"


@pytest.mark.unit
@patch(
    "coord2region.pipeline.get_studies_for_coordinate", return_value=[{"id": "1"}]
)
@patch("coord2region.pipeline.prepare_datasets", return_value={"Combined": object()})
@patch("coord2region.pipeline.MultiAtlasMapper")
@patch("coord2region.pipeline.AIModelInterface")
def test_run_pipeline_async(mock_ai, mock_multi, mock_prepare, mock_get):
    mock_multi.return_value.batch_mni_to_region_names.return_value = {}
    mock_multi.return_value.batch_region_name_to_mni.return_value = {}
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
                "gemini_api_key": "key",
            },
            async_mode=True,
            progress_callback=cb,
        )

    assert [r.summary for r in results] == ["ASYNC", "ASYNC"]
    assert [r.summaries for r in results] == [
        {"gpt-4o-mini": "ASYNC"},
        {"gpt-4o-mini": "ASYNC"},
    ]
    assert len(progress_calls) == 2


@pytest.mark.unit
@patch("coord2region.pipeline.generate_summary", side_effect=["S1", "S2"])
@patch("coord2region.pipeline.get_studies_for_coordinate", return_value=[{"id": "1"}])
@patch("coord2region.pipeline.prepare_datasets", return_value={"Combined": object()})
@patch("coord2region.pipeline.MultiAtlasMapper")
@patch("coord2region.pipeline.AIModelInterface")
def test_run_pipeline_batch_coords(
    mock_ai, mock_multi, mock_prepare, mock_get, mock_summary
):
    mock_multi.return_value.batch_mni_to_region_names.return_value = {}
    mock_multi.return_value.batch_region_name_to_mni.return_value = {}
    results = run_pipeline(
        inputs=[[0, 0, 0], [1, 1, 1]],
        input_type="coords",
        outputs=["summaries", "raw_studies"],
        config={
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
@patch("coord2region.pipeline.MultiAtlasMapper")
@patch("coord2region.pipeline.AIModelInterface")
def test_run_pipeline_export_pdf(
    mock_ai, mock_multi, mock_prepare, mock_get, mock_summary, mock_save_pdf, tmp_path
):
    mock_multi.return_value.batch_mni_to_region_names.return_value = {}
    mock_multi.return_value.batch_region_name_to_mni.return_value = {}
    output_name = "results.pdf"
    res = run_pipeline(
        inputs=[[0, 0, 0]],
        input_type="coords",
        outputs=["summaries"],
        output_format="pdf",
        output_name=output_name,
        config={
            "working_directory": str(tmp_path),
            "gemini_api_key": "key",
        },
    )
    assert res[0].summary == "SUM"
    mock_save_pdf.assert_called_once()


@pytest.mark.unit
@patch("coord2region.pipeline.MultiAtlasMapper")
@patch("coord2region.pipeline.generate_mni152_image")
def test_pipeline_nilearn_backend(mock_gen, mock_multi, tmp_path):
    buf = BytesIO()
    Image.new("RGB", (10, 10), color="white").save(buf, format="PNG")
    mock_gen.return_value = buf.getvalue()

    mock_multi.return_value.batch_mni_to_region_names.return_value = {}
    mock_multi.return_value.batch_region_name_to_mni.return_value = {}

    res = run_pipeline(
        inputs=[[0, 0, 0]],
        input_type="coords",
        outputs=["images"],
        image_backend="nilearn",
        config={
            "use_cached_dataset": False,
            "working_directory": str(tmp_path),
        },
    )
    path = res[0].images.get("nilearn")
    assert path and os.path.exists(path)


@pytest.mark.unit
@patch("coord2region.pipeline.MultiAtlasMapper")
@patch("coord2region.ai_model_interface.AIModelInterface.generate_image")
def test_pipeline_ai_watermark(mock_generate, mock_multi, tmp_path):
    buf = BytesIO()
    Image.new("RGB", (100, 50), color="black").save(buf, format="PNG")
    mock_generate.return_value = buf.getvalue()

    mock_multi.return_value.batch_mni_to_region_names.return_value = {}
    mock_multi.return_value.batch_region_name_to_mni.return_value = {}

    res = run_pipeline(
        inputs=[[0, 0, 0]],
        input_type="coords",
        outputs=["images"],
        image_backend="ai",
        config={
            "use_cached_dataset": False,
            "working_directory": str(tmp_path),
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
    ) as mock_nl, patch("coord2region.pipeline.AIModelInterface"), patch(
        "coord2region.pipeline.MultiAtlasMapper"
    ) as mock_multi:
        mock_multi.return_value.batch_mni_to_region_names.return_value = {}
        mock_multi.return_value.batch_region_name_to_mni.return_value = {}
        res = run_pipeline(
            inputs=[[0, 0, 0]],
            input_type="coords",
            outputs=["images"],
            image_backend="both",
            config={
                "use_cached_dataset": False,
                "working_directory": str(tmp_path),
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
    ), patch("coord2region.pipeline.AIModelInterface"), patch(
        "coord2region.pipeline.MultiAtlasMapper"
    ) as mock_multi:
        mock_multi.return_value.batch_mni_to_region_names.return_value = {}
        mock_multi.return_value.batch_region_name_to_mni.return_value = {}
        res = run_pipeline(
            inputs=[[0, 0, 0]],
            input_type="coords",
            outputs=["images"],
            image_backend="both",
            async_mode=True,
            config={
                "use_cached_dataset": False,
                "working_directory": str(tmp_path),
                "gemini_api_key": "k",
            },
        )

    imgs = res[0].images
    assert set(imgs.keys()) == {"ai", "nilearn"}
    for path in imgs.values():
        assert path and os.path.exists(path)


@pytest.mark.unit
@patch("coord2region.pipeline.generate_region_image", return_value=b"PNGDATA")
@patch("coord2region.pipeline.MultiAtlasMapper")
@patch("coord2region.pipeline.AIModelInterface")
def test_pipeline_image_prompt_custom_sync(
    mock_ai, mock_multi, mock_gen, tmp_path
):
    mock_multi.return_value.batch_mni_to_region_names.return_value = {}
    mock_multi.return_value.batch_region_name_to_mni.return_value = {}

    res = run_pipeline(
        inputs=[[0, 0, 0]],
        input_type="coords",
        outputs=["images"],
        image_backend="ai",
        config={
            "use_cached_dataset": False,
            "working_directory": str(tmp_path),
            "gemini_api_key": "k",
            "image_model": "my-model",
            "image_prompt_type": "custom",
            "image_custom_prompt": "Custom prompt for {coordinate} :: {atlas_context}",
        },
    )
    assert res and res[0].images.get("ai")
    args, kwargs = mock_gen.call_args
    assert kwargs["image_type"] == "custom"
    assert kwargs["model"] == "my-model"
    assert kwargs["prompt_template"] == "Custom prompt for {coordinate} :: {atlas_context}"
    assert kwargs.get("watermark", False) is True


@pytest.mark.unit
@patch("coord2region.pipeline.generate_region_image", return_value=b"PNGDATA")
@patch("coord2region.pipeline.MultiAtlasMapper")
@patch("coord2region.pipeline.AIModelInterface")
def test_pipeline_image_prompt_noncustom_sync(
    mock_ai, mock_multi, mock_gen, tmp_path
):
    mock_multi.return_value.batch_mni_to_region_names.return_value = {}
    mock_multi.return_value.batch_region_name_to_mni.return_value = {}

    _ = run_pipeline(
        inputs=[[0, 0, 0]],
        input_type="coords",
        outputs=["images"],
        image_backend="ai",
        config={
            "use_cached_dataset": False,
            "working_directory": str(tmp_path),
            "gemini_api_key": "k",
            "image_model": "another-model",
            "image_prompt_type": "functional",
        },
    )
    args, kwargs = mock_gen.call_args
    assert kwargs["image_type"] == "functional"
    assert kwargs["model"] == "another-model"
    assert kwargs.get("prompt_template") is None


@pytest.mark.unit
@patch("coord2region.pipeline.generate_region_image", return_value=b"PNGDATA")
@patch("coord2region.pipeline.MultiAtlasMapper")
@patch("coord2region.pipeline.AIModelInterface")
def test_pipeline_image_prompt_custom_async(
    mock_ai, mock_multi, mock_gen, tmp_path
):
    mock_multi.return_value.batch_mni_to_region_names.return_value = {}
    mock_multi.return_value.batch_region_name_to_mni.return_value = {}

    res = run_pipeline(
        inputs=[[0, 0, 0]],
        input_type="coords",
        outputs=["images"],
        image_backend="ai",
        async_mode=True,
        config={
            "use_cached_dataset": False,
            "working_directory": str(tmp_path),
            "gemini_api_key": "k",
            "image_model": "async-model",
            "image_prompt_type": "custom",
            "image_custom_prompt": "Async custom {coordinate}",
        },
    )
    assert res and res[0].images.get("ai")
    args, kwargs = mock_gen.call_args
    assert kwargs["image_type"] == "custom"
    assert kwargs["model"] == "async-model"
    assert kwargs["prompt_template"] == "Async custom {coordinate}"


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
        run_pipeline([1], "invalid", [])


@pytest.mark.unit
def test_run_pipeline_invalid_output():
    with pytest.raises(ValueError):
        run_pipeline([[0, 0, 0]], "coords", ["bad"])


@pytest.mark.unit
def test_run_pipeline_missing_output_name():
    with pytest.raises(ValueError):
        run_pipeline([[0, 0, 0]], "coords", ["summaries"], output_format="json")


@pytest.mark.unit
def test_run_pipeline_invalid_image_backend():
    with pytest.raises(ValueError):
        run_pipeline([[0, 0, 0]], "coords", ["images"], image_backend="wrong")


def test_normalize_model_list_variants():
    assert _normalize_model_list(None) == []
    assert _normalize_model_list("model") == ["model"]
    assert _normalize_model_list(["a", "A", "", None]) == ["a", "A"]
    assert _normalize_model_list(123) == ["123"]


def test_get_summary_models_defaults():
    assert _get_summary_models({}, "fallback") == ["fallback"]
    assert _get_summary_models({"summary_models": "one"}, "fallback") == ["one"]
    assert _get_summary_models({"summary_models": []}, "fallback") == ["fallback"]


@pytest.mark.unit
@patch("coord2region.pipeline.AtlasFetcher")
@patch("coord2region.pipeline.AIModelInterface")
@patch("coord2region.pipeline.MultiAtlasMapper")
@patch("coord2region.pipeline.prepare_datasets", return_value={})
def test_run_pipeline_register_provider(
    _mock_prepare, mock_multi, mock_ai, _mock_fetcher
):
    mock_multi.return_value.batch_mni_to_region_names.return_value = {}
    mock_multi.return_value.batch_region_name_to_mni.return_value = {}
    run_pipeline(
        inputs=[],
        input_type="coords",
        outputs=[],
        config={"providers": {"echo": {}}},
    )
    mock_ai.assert_called_once_with()
    mock_ai.return_value.register_provider.assert_called_once_with("echo", **{})


@pytest.mark.unit
@patch("coord2region.pipeline.MultiAtlasMapper")
def test_run_pipeline_none_coord(mock_multi, tmp_path):
    mock_multi.return_value.batch_mni_to_region_names.return_value = {}
    mock_multi.return_value.batch_region_name_to_mni.return_value = {}
    results = run_pipeline(
        inputs=[None],
        input_type="coords",
        outputs=["region_labels"],
        config={
            "use_cached_dataset": False,
            "working_directory": str(tmp_path),
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

        def batch_mni_to_region_names(self, coords, max_distance=None, hemi=None):
            raise RuntimeError("boom")

    mock_multi.side_effect = lambda *a, **k: RaisingMultiAtlas()
    results = run_pipeline(
        inputs=[[0, 0, 0]],
        input_type="coords",
        outputs=["region_labels"],
        config={
            "use_cached_dataset": False,
            "working_directory": str(tmp_path),
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

        def batch_mni_to_region_names(self, coords, max_distance=None, hemi=None):
            captured["coords"] = coords
            return {"custom": ["Region"]}

    mock_multi.side_effect = lambda *args, **kwargs: DummyMulti(*args, **kwargs)

    results = run_pipeline(
        inputs=[[0, 0, 0]],
        input_type="coords",
        outputs=["region_labels"],
        config={
            "use_cached_dataset": False,
            "working_directory": str(tmp_path),
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

        def batch_mni_to_region_names(self, coords, max_distance=None, hemi=None):
            assert coords == [[1.0, 2.0, 3.0]]
            return {"custom": ["Resolved"]}

    mock_multi.side_effect = lambda *a, **k: DummyMulti()

    results = run_pipeline(
        inputs=["Region"],
        input_type="region_names",
        outputs=["region_labels", "summaries"],
        config={
            "use_cached_dataset": False,
            "working_directory": str(tmp_path),
            "gemini_api_key": "key",
        },
    )

    res = results[0]
    assert res.coordinate == [1.0, 2.0, 3.0]
    assert res.region_labels == {"custom": "Resolved"}
    mock_summary.assert_called_once()


@pytest.mark.unit
@patch("coord2region.pipeline.MultiAtlasMapper")
def test_run_pipeline_region_names_mni_only(mock_multi, tmp_path):
    class DummyMulti:
        def __init__(self, *_args, **_kwargs):
            pass

        def batch_region_name_to_mni(self, names):
            assert names == ["Region"]
            return {"custom": [np.array([4.0, 5.0, 6.0])]}

        def batch_mni_to_region_names(self, _coords, max_distance=None, hemi=None):
            raise AssertionError("region lookup should not occur when only MNI requested")

    mock_multi.side_effect = lambda *a, **k: DummyMulti()

    results = run_pipeline(
        inputs=["Region"],
        input_type="region_names",
        outputs=["mni_coordinates"],
        config={
            "use_cached_dataset": False,
            "working_directory": str(tmp_path),
        },
    )

    res = results[0]
    assert res.coordinate == [4.0, 5.0, 6.0]
    assert res.mni_coordinates == [4.0, 5.0, 6.0]


@pytest.mark.unit
@patch("coord2region.pipeline.MultiAtlasMapper")
def test_run_pipeline_region_names_missing_warning(mock_multi, tmp_path):
    class DummyMulti:
        def __init__(self, *_args, **_kwargs):
            pass

        def batch_region_name_to_mni(self, names):
            assert names == ["Missing"]
            return {"custom": []}

        def batch_mni_to_region_names(
            self, _coords, max_distance=None, hemi=None
        ):  # pragma: no cover - defensive
            raise AssertionError("Should not look up region names when coordinate missing")

    mock_multi.side_effect = lambda *a, **k: DummyMulti()

    results = run_pipeline(
        inputs=["Missing"],
        input_type="region_names",
        outputs=["mni_coordinates"],
        config={
            "use_cached_dataset": False,
            "working_directory": str(tmp_path),
        },
    )

    res = results[0]
    assert res.coordinate is None
    assert res.mni_coordinates is None
    assert res.warnings == [
        "Region 'Missing' could not be resolved to coordinates with the configured atlases."
    ]


@pytest.mark.unit
@patch("coord2region.pipeline.MultiAtlasMapper")
def test_run_pipeline_output_name_rejects_paths(mock_multi, tmp_path):
    mock_multi.return_value.batch_mni_to_region_names.return_value = {}
    mock_multi.return_value.batch_region_name_to_mni.return_value = {}
    with pytest.raises(ValueError):
        run_pipeline(
            inputs=[[0, 0, 0]],
            input_type="coords",
            outputs=[],
            output_format="csv",
            output_name="nested/out.csv",
            config={
                "use_cached_dataset": False,
                "working_directory": str(tmp_path),
            },
        )
