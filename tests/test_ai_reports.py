import json
from unittest.mock import MagicMock

import pytest

from coord2region.ai_reports import (
    ReasonedReportContext,
    _context_to_payload,
    build_reasoned_report_messages,
    parse_reasoned_report_output,
    run_reasoned_report,
    build_region_image_request,
    infer_hemisphere,
)


@pytest.mark.unit
def test_infer_hemisphere_variants():
    assert infer_hemisphere([10, 0, 0]) == "right"
    assert infer_hemisphere([-10, 0, 0]) == "left"
    assert infer_hemisphere([0, 0, 0]) == "midline"
    assert infer_hemisphere([]) == "unknown"


@pytest.mark.unit
def test_context_to_payload_includes_optional_fields():
    ctx = ReasonedReportContext(
        coordinate_mni=[1.0, 2.0, 3.0],
        hemisphere="left",
        boundary_proximity_mm=2.5,
        atlas={"name": "atlas"},
        atlas_notes=["note"],
        studies=[{"id": 1}],
        allowed_domains=["a"],
        format_instructions=["Follow JSON"],
    )
    payload = _context_to_payload(ctx)
    assert payload["coordinate_mni"] == [1.0, 2.0, 3.0]
    assert payload["hemisphere"] == "left"
    assert payload["atlas_notes"] == ["note"]
    assert payload["format_instructions"] == ["Follow JSON"]


@pytest.mark.unit
def test_build_reasoned_report_messages_embeds_context():
    ctx = ReasonedReportContext(coordinate_mni=[1, 2, 3])
    messages = build_reasoned_report_messages(ctx, max_words=50)
    assert messages[0]["role"] == "system"
    assert "Coordinate context" in messages[1]["content"]
    assert "50" in messages[1]["content"]


@pytest.mark.unit
def test_parse_reasoned_report_output_with_json_block():
    output = "Narrative\n```json\n{\"quality\": \"high\"}\n```"
    report = parse_reasoned_report_output(output)
    assert report.narrative == "Narrative"
    assert report.json_data == {"quality": "high"}
    assert report.json_error is None


@pytest.mark.unit
def test_parse_reasoned_report_output_without_json():
    output = "Narrative without json"
    report = parse_reasoned_report_output(output)
    assert report.json_data is None
    assert report.json_error is not None


@pytest.mark.unit
def test_run_reasoned_report_returns_metadata():
    ctx = ReasonedReportContext(coordinate_mni=[1, 2, 3])

    class DummyAI:
        def generate_text(self, model, prompt, max_tokens, retries):
            assert model == "m"
            assert isinstance(prompt, list)
            self.called = True
            return "Story\n```json\n{\"status\":\"ok\"}\n```"

        def provider_name(self, model):
            return "DummyProvider"

    ai = DummyAI()
    report, metadata = run_reasoned_report(
        ai, "m", ctx, max_tokens=20, retries=1, system_message="sys"
    )
    assert report.json_data == {"status": "ok"}
    assert metadata["provider"] == "DummyProvider"
    assert metadata["model"] == "m"


@pytest.mark.unit
def test_build_region_image_request_constructs_prompts():
    ctx = ReasonedReportContext(
        coordinate_mni=[-4, 10, 20],
        atlas={"name": "AtlasX", "primary_label": "RegionY"},
    )
    request = build_region_image_request([ -4, 10, 20 ], ctx, sphere_radius_mm=8)
    assert request["spec"]["atlas"]["name"] == "AtlasX"
    assert request["spec"]["annotations"][0]["text"].startswith("RegionY")
    assert "MNI [-4,10,20]" in request["positive_prompt"]
