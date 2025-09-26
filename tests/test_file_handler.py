import csv
import io
import json
import os
from pathlib import Path
import zipfile

import numpy as np
import requests

import pytest

from coord2region.pipeline import PipelineResult
from coord2region.utils.file_handler import (
    AtlasFileHandler,
    save_as_csv,
    save_as_pdf,
    save_batch_folder,
)


def _make_handler(tmp_path, monkeypatch, *, data_subdir="data"):
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(
        "coord2region.utils.file_handler.mne.datasets.sample.data_path",
        lambda *a, **k: str(sample_dir),
    )
    monkeypatch.setattr(
        "coord2region.utils.file_handler.mne.get_config",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "coord2region.utils.file_handler.mne.utils.set_config",
        lambda *a, **k: None,
    )
    subjects_dir = tmp_path / "subjects"
    subjects_dir.mkdir(exist_ok=True)
    monkeypatch.setenv("SUBJECTS_DIR", str(subjects_dir))
    return AtlasFileHandler(data_dir=str(tmp_path / data_subdir))


@pytest.mark.unit
def test_missing_sample_dataset_allows_continuation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "coord2region.utils.file_handler.mne.get_config", lambda *a, **k: None
    )
    monkeypatch.delenv("SUBJECTS_DIR", raising=False)
    def raise_missing_dataset(*args, **kwargs):
        raise RuntimeError("no dataset")

    monkeypatch.setattr(
        "coord2region.utils.file_handler.mne.datasets.sample.data_path",
        raise_missing_dataset,
    )
    monkeypatch.setattr(
        "coord2region.utils.file_handler.mne.utils.set_config", lambda *a, **k: None
    )

    handler = AtlasFileHandler(data_dir=str(tmp_path / "data"))
    assert handler.subjects_dir is None


@pytest.mark.unit
def test_sample_dataset_path_str_is_normalized(tmp_path, monkeypatch):
    sample_root = tmp_path / "mne-sample"
    monkeypatch.setattr(
        "coord2region.utils.file_handler.mne.get_config", lambda *a, **k: None
    )
    monkeypatch.delenv("SUBJECTS_DIR", raising=False)

    monkeypatch.setattr(
        "coord2region.utils.file_handler.mne.datasets.sample.data_path",
        lambda *a, **k: str(sample_root),
    )

    set_config_calls = []

    def fake_set_config(key, value, set_env=False):
        set_config_calls.append((key, value, set_env))

    monkeypatch.setattr(
        "coord2region.utils.file_handler.mne.utils.set_config", fake_set_config
    )

    handler = AtlasFileHandler(data_dir=str(tmp_path / "data"))

    expected_subjects_dir = sample_root / "subjects"
    assert handler.subjects_dir == str(expected_subjects_dir)
    assert expected_subjects_dir.exists()
    assert set_config_calls
    assert set_config_calls[-1] == (
        "SUBJECTS_DIR",
        str(expected_subjects_dir),
        True,
    )


@pytest.mark.unit
def test_save_as_csv(tmp_path):
    path = tmp_path / "subdir" / "results.csv"
    res1 = PipelineResult(coordinate=[1, 2, 3], region_labels={"a": "b"}, summary="S", studies=[{"id": 1}], images={"ai": "img.png"})
    res2 = {"summary": "T"}
    res3 = [("summary", "U")]
    save_as_csv([res1, res2, res3], str(path))
    assert path.exists()
    with open(path, newline="", encoding="utf8") as f:
        rows = list(csv.DictReader(f))
    assert [r["summary"] for r in rows] == ["S", "T", "U"]
    assert json.loads(rows[0]["region_labels"]) == {"a": "b"}


@pytest.mark.unit
def test_save_as_pdf(monkeypatch, tmp_path):
    output_paths = []

    class DummyFPDF:
        def add_page(self):
            pass

        def set_font(self, *args, **kwargs):
            pass

        def multi_cell(self, *args, **kwargs):
            pass

        def image(self, *args, **kwargs):
            pass

        def output(self, path):
            output_paths.append(path)

    monkeypatch.setattr("coord2region.utils.file_handler.FPDF", DummyFPDF)

    # directory export with multiple results
    dir_path = tmp_path / "pdfs"
    save_as_pdf([PipelineResult(summary="A"), {"summary": "B"}], str(dir_path))
    assert dir_path.is_dir()
    assert output_paths[:2] == [str(dir_path / "result_1.pdf"), str(dir_path / "result_2.pdf")]

    # single file export
    file_path = tmp_path / "single.pdf"
    save_as_pdf([{"summary": "C"}], str(file_path))
    assert output_paths[2] == str(file_path)


@pytest.mark.unit
def test_save_batch_folder(tmp_path):
    img = tmp_path / "img.png"
    img.write_text("data")
    extra = tmp_path / "extra.png"
    extra.write_text("data")
    res = PipelineResult(summary="A", image=str(img), images={"extra": str(extra)})
    save_batch_folder([res, [("summary", "B")]], str(tmp_path / "out"))
    r1 = tmp_path / "out" / "result_1"
    r2 = tmp_path / "out" / "result_2"
    assert (r1 / "result.json").exists()
    assert (r1 / "img.png").exists()
    assert (r1 / "extra.png").exists()
    assert json.loads((r2 / "result.json").read_text())["summary"] == "B"


@pytest.mark.unit
def test_save_error(tmp_path, monkeypatch):
    handler = _make_handler(tmp_path, monkeypatch, data_subdir="data_dir")
    with pytest.raises(Exception):
        handler.save(lambda x: x, "bad.pkl")


@pytest.mark.unit
def test_fetch_from_local_missing(tmp_path, monkeypatch):
    handler = _make_handler(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError):
        handler.fetch_from_local("atlas.nii.gz", str(tmp_path), [])


@pytest.mark.unit
def test_fetch_from_local_success(tmp_path, monkeypatch):
    atlas_dir = tmp_path / "atlas"
    atlas_dir.mkdir()
    vol = np.ones((2, 2, 2), dtype=np.float32)
    hdr = np.eye(4, dtype=np.float32)
    np.savez(atlas_dir / "atlas.npz", vol=vol, hdr=hdr)

    labels_xml = atlas_dir / "labels.xml"
    labels_xml.write_text(
        "<root><data><label><name>A</name></label><label><name>B</name></label></data></root>",
        encoding="utf8",
    )

    handler = _make_handler(tmp_path, monkeypatch)
    res = handler.fetch_from_local("atlas.npz", str(atlas_dir), ["L1", "L2"])
    np.testing.assert_allclose(res["vol"], vol)
    np.testing.assert_allclose(res["hdr"], hdr)
    assert res["labels"] == ["L1", "L2"]

    res_xml = handler.fetch_from_local("atlas.npz", str(atlas_dir), "labels.xml")
    assert res_xml["labels"] == ["A", "B"]


@pytest.mark.unit
def test_fetch_from_url_zip(monkeypatch, tmp_path):
    handler = _make_handler(tmp_path, monkeypatch)

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as zf:
        zf.writestr("atlas/info.txt", "data")
    zip_bytes = payload.getvalue()

    calls = []

    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def iter_content(self, chunk_size=8192):
            yield zip_bytes

        def raise_for_status(self):
            return None

    def fake_get(url, stream=True, timeout=30, verify=True):
        calls.append(url)
        return DummyResponse()

    monkeypatch.setattr(requests, "get", fake_get)

    url = "https://example.com/archive.zip"
    path = handler.fetch_from_url(url)
    assert os.path.isdir(path)
    extracted_files = list(Path(path).rglob("info.txt"))
    assert extracted_files, "Expected decompressed content"
    assert calls == [url]

    # Re-fetching should skip the download but still return the existing path
    path_again = handler.fetch_from_url(url)
    assert path_again == path
    assert calls == [url]
