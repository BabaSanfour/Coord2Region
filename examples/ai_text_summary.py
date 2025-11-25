"""
Generate an atlas-aware summary for a coordinate using Coord2Region.
====================================================================

Runs :func:`coord2region.pipeline.run_pipeline` to fetch atlas context and
studies, then renders a concise LLM summary tailored to the requested target.
"""

from __future__ import annotations

import logging
from pathlib import Path

from coord2region.pipeline import run_pipeline
from coord2region.llm import generate_llm_prompt
from coord2region.paths import get_working_directory
from coord2region.ai_helpers import (
    TEXT_MODEL_CANDIDATES,
    build_interface,
    getenv_str,
    load_environment,
    select_model,
)


COORDINATE = [48, -38, -24]


def main() -> int:
    """Resolve atlas labels, fetch studies, and build an AI summary."""

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_environment()

    config_path = Path("config") / "coord2region-config.yaml"
    if config_path.exists():
        logging.info("Loaded credentials from %s", config_path)
    elif not Path(".env").exists():
        logging.warning(
            "No config/coord2region-config.yaml or .env file detected. "
            "Run `python scripts/configure_coord2region.py` to set up credentials."
        )

    workdir_name = getenv_str("COORD2REGION_WORKDIR") or "coord2region_full"
    working_dir = get_working_directory(workdir_name)
    prompt_type = getenv_str("COORD2REGION_PROMPT_TYPE") or "summary"
    custom_prompt = getenv_str("COORD2REGION_CUSTOM_PROMPT")
    show_prompt_value = (getenv_str("COORD2REGION_SHOW_PROMPT") or "").lower()
    show_prompt = show_prompt_value in {"1", "true", "yes"}

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

    # Working directory is created by get_working_directory()

    results = run_pipeline(
        inputs=[COORDINATE],
        input_type="coords",
        outputs=["region_labels", "raw_studies", "summaries"],
        config={
            "summary_models": [text_model],
            "study_search_radius": 2.0,
            "region_search_radius": 4.0,
            "working_directory": working_dir,
            "email_for_abstracts": getenv_str("EMAIL_FOR_ABSTRACTS"),
            # Pass through provider credentials so the pipeline can construct
            # its own AI interface for summary generation.
            "openrouter_api_key": getenv_str("OPENROUTER_API_KEY"),
            "openai_api_key": getenv_str("OPENAI_API_KEY"),
            "openai_project": getenv_str("OPENAI_PROJECT"),
            "anthropic_api_key": getenv_str("ANTHROPIC_API_KEY"),
            "gemini_api_key": getenv_str("GEMINI_API_KEY"),
            "huggingface_api_key": getenv_str("HUGGINGFACE_API_KEY"),
            # Optional LLM tuning
            "summary_max_tokens": 900,
            "prompt_type": prompt_type,
            "custom_prompt": custom_prompt if prompt_type == "custom" else None,
            "use_cached_dataset": True,
        },
    )

    if not results:
        print("Pipeline returned no results; check atlas configuration.")
        return 1

    result = results[0]
    print(f"Coordinate: {result.coordinate}")
    print(f"Region labels: {result.region_labels}")
    print(f"Summary model: {text_model}")
    print(f"Studies found: {len(result.studies)}")

    # Optional: show the exact prompt that will be used for the summary
    if show_prompt and result.studies:
        prompt = generate_llm_prompt(
            result.studies,
            result.coordinate or COORDINATE,
            prompt_type=prompt_type,
            prompt_template=custom_prompt if prompt_type == "custom" else None,
        )
        if result.region_labels:
            parts = prompt.split("STUDIES REPORTING ACTIVATION AT MNI COORDINATE")
            atlas_info = "\nATLAS LABELS FOR THIS COORDINATE:\n"
            for atlas_name, label in result.region_labels.items():
                atlas_info += f"- {atlas_name}: {label}\n"
            if len(parts) >= 2:
                intro = parts[0]
                rest = "STUDIES REPORTING ACTIVATION AT MNI COORDINATE" + parts[1]
                prompt = intro + atlas_info + "\n" + rest
            else:
                prompt = atlas_info + prompt
        print("\n--- Prompt Preview ---\n")
        print(prompt)
    print("\n--- Summary ---\n")
    if result.summary:
        print(result.summary)
    else:
        print("No summary returned.")
    print("\n--- Studies ---\n")
    if result.studies:
        for study in result.studies[:5]:
            title = study.get("title", "Untitled study")
            source = study.get("source", "Unknown source")
            distance = study.get("distance_mm", "n/a")
            print(f"- {title} ({source}, distance {distance} mm)")
        if len(result.studies) > 5:
            print(f"... {len(result.studies) - 5} more studies omitted.")
    else:
        print("No studies found for the search radius.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
