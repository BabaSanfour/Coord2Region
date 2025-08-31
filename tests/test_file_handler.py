import csv
import json

import pytest

from coord2region.pipeline import PipelineResult
from coord2region.utils.file_handler import (
    AtlasFileHandler,
    save_as_csv,
    save_as_pdf,
    save_batch_folder,
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
def test_save_error(tmp_path):
    handler = AtlasFileHandler(data_dir=str(tmp_path))
    with pytest.raises(Exception):
        handler.save(lambda x: x, "bad.pkl")
