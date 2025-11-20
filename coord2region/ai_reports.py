"""High-level helpers for generating structured AI reports and image specs.

This module centralises the logic that was previously spread across gallery
examples, making it easier to build real applications around the AI features.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .ai_model_interface import AIModelInterface, build_generation_summary


DEFAULT_SYSTEM_MESSAGE = (
    "You are a careful neuroscience assistant. You convert MNI brain coordinates "
    "into clear, evidence-grounded explanations using ONLY the data provided in the "
    "prompt (atlas labels, neighbors, and the study list).\n"
    'If something is missing or conflicting, explicitly report "insufficient_evidence" '
    "for that part--do not invent facts or citations."
)

DEFAULT_NEGATIVE_PROMPT = (
    "cartoon, abstract art, texture noise, extra brains, low resolution, blurry, "
    "distorted anatomy, overexposed, text blocks covering the brain, bright "
    "multicolor palettes, dramatic lighting, artistic shadows"
)


@dataclass
class ReasonedReportContext:
    """Structured payload describing the coordinate, atlas, and studies."""

    coordinate_mni: Sequence[float]
    hemisphere: Optional[str] = None
    boundary_proximity_mm: Optional[float] = None
    atlas: Dict[str, Any] = field(default_factory=dict)
    atlas_notes: List[str] = field(default_factory=list)
    studies: List[Dict[str, Any]] = field(default_factory=list)
    allowed_domains: Optional[Sequence[str]] = None
    format_instructions: List[str] = field(default_factory=list)


@dataclass
class ReasonedReport:
    """Parsed result returned by :func:`run_reasoned_report`."""

    narrative: str
    json_text: Optional[str] = None
    json_data: Optional[Dict[str, Any]] = None
    json_error: Optional[str] = None


def infer_hemisphere(coord: Sequence[float]) -> str:
    """Infer the hemisphere from an MNI coordinate.

    Parameters
    ----------
    coord : sequence of float
        MNI coordinate to evaluate.

    Returns
    -------
    str
        ``"left"``, ``"right"``, ``"midline"``, or ``"unknown"`` depending on
        the x coordinate.
    """
    if not coord:
        return "unknown"
    x = float(coord[0])
    if x > 3:
        return "right"
    if x < -3:
        return "left"
    if -3 <= x <= 3:
        return "midline"
    return "unknown"


def _context_to_payload(context: ReasonedReportContext) -> Dict[str, Any]:
    """Convert :class:`ReasonedReportContext` to a JSON-serialisable payload.

    Parameters
    ----------
    context : ReasonedReportContext
        Structured context describing the coordinate, atlas, and studies.

    Returns
    -------
    dict
        Flattened payload ready for inclusion in a user prompt.
    """
    hemisphere = context.hemisphere or infer_hemisphere(context.coordinate_mni)

    payload: Dict[str, Any] = {
        "coordinate_mni": list(context.coordinate_mni),
        "hemisphere": hemisphere,
    }
    if context.boundary_proximity_mm is not None:
        payload["boundary_proximity_mm"] = context.boundary_proximity_mm
    if context.atlas:
        payload["atlas"] = context.atlas
    if context.atlas_notes:
        payload["atlas_notes"] = list(context.atlas_notes)
    if context.studies:
        payload["studies"] = list(context.studies)
    if context.allowed_domains is not None:
        payload["allowed_domains"] = list(context.allowed_domains)
    if context.format_instructions:
        payload["format_instructions"] = list(context.format_instructions)
    return payload


def build_reasoned_report_messages(
    context: ReasonedReportContext,
    *,
    system_message: str = DEFAULT_SYSTEM_MESSAGE,
    max_words: int = 180,
) -> List[Dict[str, str]]:
    """Construct chat messages for the reasoned report prompt.

    Parameters
    ----------
    context : ReasonedReportContext
        Context describing the coordinate, atlas, and studies.
    system_message : str, optional
        System prompt guiding the AI assistant.
    max_words : int, optional
        Maximum narrative word count requested from the assistant.

    Returns
    -------
    list of dict
        Stream-ready chat messages for the AI request.
    """
    payload = _context_to_payload(context)
    user_prompt = (
        "Coordinate context for the Coord2Region reasoned report:\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n\n"
        f"Return the narrative (<= {max_words} words) followed immediately by the "
        "STRICT JSON object in ```json fences. Do not include extra commentary or "
        "surrounding text."
    )
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]


def parse_reasoned_report_output(output: str) -> ReasonedReport:
    """Split the reasoned report narrative and JSON payload.

    Parameters
    ----------
    output : str
        Raw text returned by the AI assistant.

    Returns
    -------
    ReasonedReport
        Parsed narrative along with optional JSON payload information.
    """
    narrative = output.strip()
    json_text: Optional[str] = None
    json_data: Optional[Dict[str, Any]] = None
    json_error: Optional[str] = None

    fenced_pattern = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
    fenced_match = fenced_pattern.search(output)

    if fenced_match:
        json_text = fenced_match.group(1).strip()
        narrative = output[: fenced_match.start()].strip()
        try:
            json_data = json.loads(json_text)
        except json.JSONDecodeError as exc:
            json_error = f"{exc.__class__.__name__}: {exc}"
    else:
        first_brace = output.find("{")
        if first_brace != -1:
            tail = output[first_brace:]
            last_brace = tail.rfind("}")
            if last_brace != -1:
                candidate = tail[: last_brace + 1].strip()
                if candidate:
                    json_text = candidate
                    narrative = output[:first_brace].strip()
                    try:
                        json_data = json.loads(candidate)
                    except json.JSONDecodeError as exc:
                        json_error = f"{exc.__class__.__name__}: {exc}"
        if json_text is None:
            json_error = "JSON block not found."

    return ReasonedReport(
        narrative=narrative,
        json_text=json_text,
        json_data=json_data,
        json_error=json_error,
    )


def run_reasoned_report(
    ai: AIModelInterface,
    model: str,
    context: ReasonedReportContext,
    *,
    max_tokens: int = 512,
    retries: int = 3,
    system_message: str = DEFAULT_SYSTEM_MESSAGE,
) -> Tuple[ReasonedReport, Dict[str, Any]]:
    """Generate and parse a reasoned report, returning metadata alongside.

    Parameters
    ----------
    ai : AIModelInterface
        AI interface used to generate text.
    model : str
        Model name to use for the completion.
    context : ReasonedReportContext
        Structured context describing the coordinate, atlas, and studies.
    max_tokens : int, optional
        Maximum number of tokens requested from the model.
    retries : int, optional
        Number of retry attempts for the text generation call.
    system_message : str, optional
        System message that guides the assistant's tone and scope.

    Returns
    -------
    tuple
        Tuple containing the parsed :class:`ReasonedReport` and metadata.
    """
    messages = build_reasoned_report_messages(context, system_message=system_message)
    start = time.perf_counter()
    completion = ai.generate_text(
        model=model,
        prompt=messages,
        max_tokens=max_tokens,
        retries=retries,
    )
    duration = time.perf_counter() - start
    provider = ai.provider_name(model)
    summary = build_generation_summary(model, completion, provider)
    report = parse_reasoned_report_output(completion)
    metadata = {
        "model": model,
        "provider": provider,
        "duration_s": duration,
        "summary": summary,
        "raw_text": completion,
        "messages": messages,
    }
    return report, metadata


def build_region_image_request(
    coord: Sequence[float],
    context: ReasonedReportContext,
    *,
    sphere_radius_mm: float = 6,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
) -> Dict[str, Any]:
    """Return prompt components for :meth:`AIModelInterface.generate_image`.

    Parameters
    ----------
    coord : sequence of float
        Target MNI coordinate for the image.
    context : ReasonedReportContext
        Context used to describe the coordinate and atlas.
    sphere_radius_mm : float, optional
        Radius of the spherical highlight in millimetres.
    negative_prompt : str, optional
        Negative prompt guiding the image generation.

    Returns
    -------
    dict
        Dictionary containing both structured specification and text prompts.
    """
    atlas = context.atlas or {}
    hemisphere = context.hemisphere or atlas.get("hemisphere")
    if not hemisphere:
        hemisphere = infer_hemisphere(coord)
    primary_label = atlas.get("primary_label") or atlas.get("label") or "Unknown Region"
    coord_list = [float(value) for value in coord]
    coord_int = [int(round(value)) for value in coord_list]
    coord_str = f"[{coord_int[0]},{coord_int[1]},{coord_int[2]}]"

    spec = {
        "figure_goal": (
            "Show the anatomical location of the given MNI coordinate on an "
            "MNI152-like template with overlays."
        ),
        "coordinate_mni": coord_int,
        "atlas": {
            "name": atlas.get("name", "Unknown Atlas"),
            "version": atlas.get("version", "unknown"),
            "primary_label": primary_label,
            "hemisphere": hemisphere,
        },
        "views": [
            {"plane": "axial", "z_mm": coord_int[2]},
            {"plane": "coronal", "y_mm": coord_int[1]},
            {"plane": "sagittal", "x_mm": coord_int[0]},
        ],
        "overlays": [
            {"type": "crosshair", "thickness": "thin"},
            {"type": "sphere", "radius_mm": sphere_radius_mm, "opacity": 0.65},
        ],
        "annotations": [
            {"text": f"{primary_label} ({hemisphere})", "anchor": "near_coordinate"},
            {"text": f"MNI {coord_str}", "anchor": "lower_left"},
        ],
        "style": {
            "figure_type": "clean medical figure",
            "background": "T1-weighted MNI152 appearance",
            "palette": "grayscale anatomy with a single red overlay",
            "resolution": "1024x1024",
            "layout": "3-view grid (axial, coronal, sagittal)",
            "watermark": "Illustrative",
        },
        "constraints": [
            "No diagnostic claims.",
            "No artistic textures.",
            "Crisp labels and legible fonts.",
            "Avoid cartoonish elements.",
        ],
    }

    positive_prompt = (
        "Publication-quality structural brain figure on an MNI152 template, "
        "3-view grid (axial, coronal, sagittal), grayscale T1 anatomy, "
        f"bold red spherical highlight (radius {sphere_radius_mm} mm) centered at "
        f"MNI {coord_str},  thin white crosshair at the coordinate, "
        f'labels: "{primary_label} ({hemisphere})" near the highlight '
        f'and "MNI {coord_str}" bottom-left, '
        "clean medical styling, high contrast, crisp lines, 1024x1024,"
        "subtle 'Illustrative' watermark in a corner."
    )

    return {
        "spec": spec,
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
    }


__all__ = [
    "ReasonedReportContext",
    "ReasonedReport",
    "DEFAULT_SYSTEM_MESSAGE",
    "DEFAULT_NEGATIVE_PROMPT",
    "infer_hemisphere",
    "build_reasoned_report_messages",
    "parse_reasoned_report_output",
    "run_reasoned_report",
    "build_region_image_request",
]
