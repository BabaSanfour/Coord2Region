"""
Generate an AI image and a deterministic Nilearn reference for a coordinate.
============================================================================

This workflow runs :func:`coord2region.pipeline.run_pipeline` to collect atlas
labels, nearby studies, and AI text summaries, then generates both an AI image
and a deterministic Nilearn visualization for the target coordinate.
"""

from __future__ import annotations

import logging
from pathlib import Path

from coord2region.llm import generate_mni152_image, generate_region_image
from coord2region.utils.image_utils import build_side_by_side_panel
from coord2region.pipeline import run_pipeline
from coord2region.ai_helpers import (
    IMAGE_MODEL_CANDIDATES,
    TEXT_MODEL_CANDIDATES,
    build_interface,
    getenv_str,
    load_environment,
    select_model,
)


COORDINATE = [30, -22, 50]
WORKING_DIR = Path("ai_examples_workspace")
OUTPUT_DIR = Path("ai_examples_outputs")


def main() -> int:
    """Create AI and Nilearn images for a target coordinate."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_environment()

    ai = build_interface()
    image_model = select_model(
        ai,
        IMAGE_MODEL_CANDIDATES,
        explicit=getenv_str("COORD2REGION_IMAGE_MODEL"),
        kind="image",
    )
    if image_model is None:
        print("No image-capable model available; only Nilearn reference will be generated.")

    text_model = select_model(
        ai,
        TEXT_MODEL_CANDIDATES,
        explicit=getenv_str("COORD2REGION_TEXT_MODEL") or "gpt-4o-mini",
        kind="text",
    )

    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "region_search_radius": 4.0,
        "study_search_radius": 8.0,
        "working_directory": str(WORKING_DIR),
        "email_for_abstracts": getenv_str("EMAIL_FOR_ABSTRACTS"),
        "openai_api_key": getenv_str("OPENAI_API_KEY"),
        "openai_project": getenv_str("OPENAI_PROJECT"),
    }
    if text_model:
        config["summary_models"] = [text_model]

    results = run_pipeline(
        inputs=[COORDINATE],
        input_type="coords",
        outputs=["region_labels", "summaries", "raw_studies"],
        config=config,
    )

    if not results:
        print("Pipeline produced no output; cannot generate images.")
        return 1

    result = results[0]
    region_info = {
        "summary": result.summary or "",
        "atlas_labels": result.region_labels,
        "studies": result.studies,
    }

    ai_bytes: bytes | None = None
    nilearn_bytes: bytes | None = None

    if image_model:
        try:
            ai_bytes = generate_region_image(
                ai,
                COORDINATE,
                region_info,
                image_type="functional",
                model=image_model,
                watermark=True,
            )
            ai_path = OUTPUT_DIR / "ai_image.png"
            ai_path.write_bytes(ai_bytes)
            print(f"Saved AI-generated image to {ai_path}")
        except Exception as exc:
            print(f"AI image generation failed: {exc}")

    try:
        nilearn_bytes = generate_mni152_image(COORDINATE)
        nilearn_path = OUTPUT_DIR / "nilearn_reference.png"
        nilearn_path.write_bytes(nilearn_bytes)
        print(f"Saved Nilearn reference image to {nilearn_path}")
    except Exception as exc:
        print(f"Nilearn reference generation failed: {exc}")

    if ai_bytes and nilearn_bytes:
        try:
            comparison_bytes = build_side_by_side_panel(
                ai_bytes,
                nilearn_bytes,
                left_title=f"AI model: {image_model}",
                right_title="Nilearn reference (MNI152)",
            )
            comparison_path = OUTPUT_DIR / "ai_vs_nilearn.png"
            comparison_path.write_bytes(comparison_bytes)
            print(f"Saved side-by-side comparison to {comparison_path}")
        except Exception as exc:
            print(f"Failed to build comparison panel: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
