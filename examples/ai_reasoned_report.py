"""
Build a structured reasoned report with narrative and machine-readable JSON.
============================================================================

This script combines atlas/study lookups with :func:`coord2region.ai_reports`
helpers to produce a narrative plus a strict JSON summary for downstream
systems.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from coord2region.ai_reports import (
    DEFAULT_SYSTEM_MESSAGE,
    ReasonedReportContext,
    build_region_image_request,
    infer_hemisphere,
    run_reasoned_report,
)
from coord2region.pipeline import run_pipeline
from coord2region.ai_helpers import (
    TEXT_MODEL_CANDIDATES,
    build_interface,
    getenv_str,
    load_environment,
    select_model,
)


COORDINATE = [30, -22, 50]
WORKING_DIR = Path("ai_examples_workspace")


def _prepare_context(result) -> ReasonedReportContext:
    """Convert pipeline output into :class:`ReasonedReportContext`."""

    atlas_label = result.region_labels.get("harvard-oxford", "Unknown region")
    atlas_info = {
        "name": "Harvard-Oxford Cortical Structural Atlas",
        "primary_label": atlas_label,
    }
    coordinate = result.coordinate or COORDINATE
    return ReasonedReportContext(
        coordinate_mni=coordinate,
        hemisphere=infer_hemisphere(coordinate),
        atlas=atlas_info,
        studies=result.studies,
        format_instructions=[
            "Return the narrative first (<= 180 words).",
            "Follow with a STRICT JSON object in ```json fences containing fields domain, evidence, and recommendations.",
            "Use PMIDs from the study list for inline citations.",
        ],
        allowed_domains=[
            "motor",
            "somatosensory",
            "visual",
            "auditory",
            "language",
            "memory",
            "executive",
            "affective",
            "other",
        ],
    )


def main() -> int:
    """Run the end-to-end reasoned report workflow."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_environment()

    ai = build_interface()
    text_model = select_model(
        ai,
        TEXT_MODEL_CANDIDATES,
        explicit=getenv_str("COORD2REGION_TEXT_MODEL") or "gpt-4o-mini",
        kind="text",
    )
    if text_model is None:
        print("No supported text model available. Configure API keys and retry.")
        return 1

    WORKING_DIR.mkdir(parents=True, exist_ok=True)

    results = run_pipeline(
        inputs=[COORDINATE],
        input_type="coords",
        outputs=["region_labels", "raw_studies"],
        config={
            "study_search_radius": 10.0,
            "region_search_radius": 5.0,
            "working_directory": str(WORKING_DIR),
            "email_for_abstracts": getenv_str("EMAIL_FOR_ABSTRACTS"),
            "openai_api_key": getenv_str("OPENAI_API_KEY"),
            "openai_project": getenv_str("OPENAI_PROJECT"),
        },
    )

    if not results or not results[0].studies:
        print("No studies were retrieved for the coordinate.")
        return 1

    context = _prepare_context(results[0])
    report, metadata = run_reasoned_report(
        ai,
        text_model,
        context,
        max_tokens=700,
        system_message=DEFAULT_SYSTEM_MESSAGE,
    )

    print(f"Model: {metadata['model']} ({metadata['provider']})")
    print(f"Duration: {metadata['duration_s']:.2f}s")
    print("\n--- Narrative ---\n")
    print(report.narrative or "No narrative returned.")

    print("\n--- Structured JSON ---\n")
    if report.json_data:
        print(json.dumps(report.json_data, indent=2, sort_keys=True))
    elif report.json_text:
        print(report.json_text)
        print("\n(Note: JSON parsing failed; see metadata JSON text.)")
    else:
        print("Model response did not include a JSON block.")

    image_prompt = build_region_image_request(
        COORDINATE,
        context,
        sphere_radius_mm=6,
    )
    print("\n--- Image Prompt Preview ---\n")
    print(json.dumps(image_prompt["spec"], indent=2, sort_keys=True))
    print("\nPositive prompt:\n")
    print(image_prompt["positive_prompt"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
