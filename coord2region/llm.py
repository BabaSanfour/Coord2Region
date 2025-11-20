"""LLM utilities for prompt construction and summary generation.

The summary helpers keep an in-memory LRU cache keyed by ``(model, prompt)``.
The cache currently uses a fixed size controlled by :data:`SUMMARY_CACHE_SIZE`.
"""

from collections import OrderedDict
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from .utils.image_utils import generate_mni152_image, add_watermark
from .ai_model_interface import AIModelInterface


SUMMARY_CACHE_SIZE = 128

# ---------------------------------------------------------------------------
# Exposed prompt templates
# ---------------------------------------------------------------------------

# Templates for the introductory portion of LLM prompts. Users can inspect and
# customize these as needed before passing them to :func:`generate_llm_prompt`.
LLM_PROMPT_TEMPLATES: Dict[str, str] = {
    "summary": (
        "You are an advanced AI with expertise in neuroanatomy and cognitive "
        "neuroscience. The user is interested in understanding the significance "
        "of MNI coordinate {coord}.\n\n"
        "Below is a list of neuroimaging studies that report activation at this "
        "coordinate. Your task is to integrate and synthesize the knowledge from "
        "these studies, focusing on:\n"
        "1) The anatomical structure(s) most commonly associated with this coordinate\n"
        "2) The typical functional roles or processes linked to activation in this "
        "region\n"
        "3) The main tasks or experimental conditions in which it was reported\n"
        "4) Patterns, contradictions, or debates in the findings\n\n"
        "Do NOT simply list each study separately. Provide an integrated, cohesive "
        "summary.\n"
    ),
    "region_name": (
        "You are a neuroanatomy expert. The user wants to identify the probable "
        "anatomical labels for MNI coordinate {coord}. The following studies "
        "reported activation around this location. Incorporate anatomical "
        "knowledge and any direct references to brain regions from these studies. "
        "If multiple labels are possible, mention all and provide rationale and "
        "confidence levels.\n\n"
    ),
    "function": (
        "You are a cognitive neuroscience expert. The user wants a deep "
        "functional profile of the brain region(s) around MNI coordinate {coord}. "
        "The studies below report activation at or near this coordinate. "
        "Synthesize a clear description of:\n"
        "1) Core functions or cognitive processes\n"
        "2) Typical experimental paradigms or tasks\n"
        "3) Known functional networks or connectivity\n"
        "4) Divergent or debated viewpoints in the literature\n\n"
    ),
    "default": (
        "Please analyze the following neuroimaging studies reporting activation at "
        "MNI coordinate {coord} and provide a concise yet thorough discussion of "
        "its anatomical location and functional significance.\n\n"
    ),
}


# Templates for image prompt generation. Each template can be formatted with
# ``coordinate``, ``first_paragraph``, ``atlas_context``,
# and ``study_context`` variables.
IMAGE_PROMPT_TEMPLATES: Dict[str, str] = {
    "anatomical": (
        "Create a scientific brain visualization showing exactly three orthogonal MRI"
        "slices arranged horizontally: coronal (left), sagittal (middle),"
        "and axial (right) views. "
        "Use grayscale T1-weighted MRI brain anatomy on a black background. "
        "Place bright yellow or white crosshairs (+) at MNI coordinate {coordinate},"
        "with the crosshairs extending across each slice to mark the exact location. "
        "Label each view with the coordinate values shown. "
        "Add L/R orientation markers. The style should match standard neuroimaging "
        "software output like FSLeyes or Nilearn, with no artistic interpretation. "
        "Ensure the crosshairs intersect precisely at the specified coordinate point.\n"
        "Coordinate location: x={x_coord}, y={y_coord}, z={z_coord}\n"
        "{atlas_context}"
    ),
    "functional": (
        "Produce a Nilearn-style activation map with sagittal, coronal, and axial "
        "panels centred on coordinate {coordinate}.\n"
        "Functional interpretation: {first_paragraph}\n"
        "{atlas_context}{study_context}"
        "Overlay activation intensities as a heat map on the MNI152 template, include "
        "legend ticks, slice coordinates, and crosshairs precisely at the specified "
        "location."
    ),
    "schematic": (
        "Draw a network schematic anchored on MNI coordinate {coordinate}. Include an "
        "inset miniature of the Nilearn-style orthogonal slices marking the focus.\n"
        "Conceptual summary: {first_paragraph}\n"
        "{atlas_context}{study_context}"
        "Label interacting regions, indicate connectivity directions when supported, "
        "and keep the overall style technical and publication-ready."
    ),
    "artistic": (
        "Create a stylised yet anatomically faithful visualization spotlighting "
        "coordinate {coordinate}. Retain Nilearn-like slice framing so the activation "
        "can be compared to reference material.\n"
        "Narrative focus: {first_paragraph}\n"
        "{atlas_context}{study_context}"
        "Blend scientific structure with thoughtful lighting or texture while keeping "
        "the coordinate marker and orthogonal slices clear."
    ),
    "default": (
        "Render a Nilearn-style comparative figure centred on coordinate "
        "{coordinate}. Provide orthogonal MNI152 slices with crosshairs, legend, and "
        "activation emphasis.\n"
        "Primary description: {first_paragraph}\n"
        "{atlas_context}{study_context}"
        "Ensure the output resembles neuroimaging data ready for side-by-side "
        "comparison with a deterministic Nilearn export."
    ),
}


def generate_llm_prompt(
    studies: List[Dict[str, Any]],
    coordinate: Union[List[float], Tuple[float, float, float]],
    prompt_type: str = "summary",
    prompt_template: Optional[str] = None,
) -> str:
    """Generate a detailed prompt for language models based on studies.

    Parameters
    ----------
    studies : list of dict
        Study metadata dictionaries that describe the activation evidence.
    coordinate : sequence of float
        MNI coordinate used for formatting the prompt header.
    prompt_type : str, optional
        Key that selects a built-in template from :data:`LLM_PROMPT_TEMPLATES`.
    prompt_template : str, optional
        Custom template string requiring ``coord`` and ``studies`` placeholders.

    Returns
    -------
    str
        Fully formatted prompt ready for submission to a language model.
    """
    # Format coordinate string safely.
    try:
        coord_str = "[{:.2f}, {:.2f}, {:.2f}]".format(
            float(coordinate[0]), float(coordinate[1]), float(coordinate[2])
        )
    except Exception:
        coord_str = str(coordinate)

    if not studies:
        return (
            "No neuroimaging studies were found reporting activation at "
            f"MNI coordinate {coord_str}."
        )

    # Build the studies section efficiently.
    study_lines: List[str] = []
    for i, study in enumerate(studies, start=1):
        study_lines.append(f"\n--- STUDY {i} ---\n")
        study_lines.append(f"ID: {study.get('id', 'Unknown ID')}\n")
        study_lines.append(f"Title: {study.get('title', 'No title available')}\n")
        abstract_text = study.get("abstract", "No abstract available")
        study_lines.append(f"Abstract: {abstract_text}\n")
    studies_section = "".join(study_lines)

    if prompt_type == "custom" and not prompt_template:
        raise ValueError("prompt_template must be provided when prompt_type='custom'")

    # If a custom template is provided, use it.
    if prompt_template:
        return prompt_template.format(coord=coord_str, studies=studies_section)

    # Build the prompt header using the templates dictionary.
    template = LLM_PROMPT_TEMPLATES.get(prompt_type, LLM_PROMPT_TEMPLATES["default"])
    prompt_intro = template.format(coord=coord_str)

    prompt_body = (
        "STUDIES REPORTING ACTIVATION AT MNI COORDINATE "
        + coord_str
        + ":\n"
        + studies_section
    )

    prompt_outro = (
        "\nUsing ALL of the information above, produce a single cohesive "
        "synthesis. Avoid bullet-by-bullet summaries of each study. Instead, "
        "integrate the findings across them to describe the region's "
        "location, function, and context."
    )

    return prompt_intro + prompt_body + prompt_outro


def generate_region_image_prompt(
    coordinate: Union[List[float], Tuple[float, float, float]],
    region_info: Dict[str, Any],
    image_type: str = "anatomical",
    include_atlas_labels: bool = True,
    prompt_template: Optional[str] = None,
) -> str:
    """Generate a prompt for creating images of brain regions.

    Parameters
    ----------
    coordinate : sequence of float
        Target MNI coordinate to highlight in the visualization.
    region_info : dict
        Metadata describing the region, such as ``summary`` and atlas labels.
    image_type : str, optional
        Template key selecting the style of the image prompt.
    include_atlas_labels : bool, optional
        Whether atlas label descriptions should be inserted into the prompt.
    prompt_template : str, optional
        Custom template string overriding the built-in prompt dictionary.

    Returns
    -------
    str
        Fully formatted prompt with atlas and study context injected.
    """
    # Safely get the summary and a short first paragraph.
    summary = region_info.get("summary", "No summary available.")
    first_paragraph = summary.split("\n\n", 1)[0]

    # Format the coordinate for inclusion in the prompt.
    try:
        x_val = float(coordinate[0])
        y_val = float(coordinate[1])
        z_val = float(coordinate[2])
        coord_str = "[{:.2f}, {:.2f}, {:.2f}]".format(x_val, y_val, z_val)
        x_coord = f"{x_val:.0f}"
        y_coord = f"{y_val:.0f}"
        z_coord = f"{z_val:.0f}"
    except Exception:
        # Fallback to the raw coordinate representation.
        coord_str = str(coordinate)
        x_coord = y_coord = z_coord = "0"

    # Build atlas context if requested and available.
    atlas_context = ""
    atlas_labels = region_info.get("atlas_labels") or {}
    if include_atlas_labels and isinstance(atlas_labels, dict) and atlas_labels:
        atlas_parts = [
            f"{atlas_name}: {label}" for atlas_name, label in atlas_labels.items()
        ]
        atlas_context = (
            "According to brain atlases, this region corresponds to: "
            + ", ".join(atlas_parts)
            + ". "
        )

    # Build study context - not used for anatomical images but
    # needed for template compatibility
    study_context = ""
    studies = region_info.get("studies") or []
    if studies and image_type in ["functional", "schematic", "artistic", "default"]:
        study_lines = []
        for i, study in enumerate(studies[:3], 1):  # Limit to first 3 studies
            title = study.get("title", "").strip()
            if title:
                study_lines.append(f"Study {i}: {title[:80]}...")
        if study_lines:
            study_context = "Related research: " + "; ".join(study_lines) + ". "

    # If a custom template is provided, use it directly.
    if prompt_template:
        return prompt_template.format(
            coordinate=coord_str,
            x_coord=x_coord,
            y_coord=y_coord,
            z_coord=z_coord,
            first_paragraph=first_paragraph,
            atlas_context=atlas_context,
            study_context=study_context,
        )
    # Retrieve prompt template by image type or fall back to default.
    template = IMAGE_PROMPT_TEMPLATES.get(image_type, IMAGE_PROMPT_TEMPLATES["default"])
    return template.format(
        coordinate=coord_str,
        x_coord=x_coord,
        y_coord=y_coord,
        z_coord=z_coord,
        first_paragraph=first_paragraph,
        atlas_context=atlas_context,
        study_context=study_context,
    )


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------


def generate_region_image(
    ai: "AIModelInterface",
    coordinate: Union[List[float], Tuple[float, float, float]],
    region_info: Dict[str, Any],
    image_type: str = "anatomical",
    model: str = "stabilityai/stable-diffusion-2",
    include_atlas_labels: bool = True,
    prompt_template: Optional[str] = None,
    retries: int = 3,
    watermark: bool = True,
    **kwargs: Any,
) -> bytes:
    """Generate an image for a brain region using an AI model.

    Parameters
    ----------
    ai : AIModelInterface
        Interface used to generate images.
    coordinate : sequence of float
        MNI coordinate for the target region.
    region_info : dict
        Dictionary containing region summary and atlas labels.
    image_type : str, optional
        Type of image to generate. Defaults to ``"anatomical"``.
    model : str, optional
        Name of the AI model to use. Defaults to
        ``"stabilityai/stable-diffusion-2"``.
    include_atlas_labels : bool, optional
        Whether to include atlas label context in the prompt. Defaults to ``True``.
    prompt_template : str, optional
        Custom template overriding default prompts.
    retries : int, optional
        Number of times to retry generation on failure. Defaults to ``3``.
    watermark : bool, optional
        When ``True`` (default), a semi-transparent watermark is applied to the
        resulting image.
    **kwargs : Any
        Additional keyword arguments passed to the underlying AI provider.

    Returns
    -------
    bytes
        PNG image bytes, optionally watermarked.
    """
    prompt = generate_region_image_prompt(
        coordinate,
        region_info,
        image_type=image_type,
        include_atlas_labels=include_atlas_labels,
        prompt_template=prompt_template,
    )
    img_bytes = ai.generate_image(model=model, prompt=prompt, retries=retries, **kwargs)
    if watermark:
        img_bytes = add_watermark(img_bytes)
    return img_bytes


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------


def generate_summary(
    ai: "AIModelInterface",
    studies: List[Dict[str, Any]],
    coordinate: Union[List[float], Tuple[float, float, float]],
    prompt_type: str = "summary",
    model: str = "gemini-2.0-flash",
    atlas_labels: Optional[Dict[str, str]] = None,
    custom_prompt: Optional[str] = None,
    max_tokens: int = 1000,
) -> str:
    """Generate a text summary for a coordinate based on studies.

    Parameters
    ----------
    ai : AIModelInterface
        AI backend used to create the summary.
    studies : list of dict
        Studies reporting activation at the target coordinate.
    coordinate : sequence of float
        MNI coordinate around which the summary should focus.
    prompt_type : str, optional
        Key into :data:`LLM_PROMPT_TEMPLATES`. Use ``"custom"`` with
        ``custom_prompt`` to provide a bespoke template.
    model : str, optional
        Name of the text generation model. Defaults to ``"gemini-2.0-flash"``.
    atlas_labels : dict, optional
        Atlas-derived labels to prepend to the prompt for extra context.
    custom_prompt : str, optional
        Template string formatted with ``coord`` and ``studies`` placeholders.
    max_tokens : int, optional
        Maximum number of tokens requested from the language model.

    Returns
    -------
    str
        Textual summary returned by the AI model.
    """
    # Build base prompt with study information
    prompt = generate_llm_prompt(
        studies,
        coordinate,
        prompt_type=prompt_type,
        prompt_template=custom_prompt if prompt_type == "custom" else None,
    )

    # Insert atlas label information when provided
    if atlas_labels:
        parts = prompt.split("STUDIES REPORTING ACTIVATION AT MNI COORDINATE")
        atlas_info = "\nATLAS LABELS FOR THIS COORDINATE:\n"
        for atlas_name, label in atlas_labels.items():
            atlas_info += f"- {atlas_name}: {label}\n"
        if len(parts) >= 2:
            intro = parts[0]
            rest = "STUDIES REPORTING ACTIVATION AT MNI COORDINATE" + parts[1]
            prompt = intro + atlas_info + "\n" + rest
        else:
            prompt = atlas_info + prompt

    # Generate and return the summary using the AI interface with caching
    key = (model, prompt)
    cache: "OrderedDict[Tuple[str, str], str]" = generate_summary._cache
    if SUMMARY_CACHE_SIZE > 0:
        cached = cache.get(key)
        if cached is not None:
            cache.move_to_end(key)
            return cached

    result = ai.generate_text(model=model, prompt=prompt, max_tokens=max_tokens)

    if SUMMARY_CACHE_SIZE > 0:
        cache[key] = result
        cache.move_to_end(key)
        while len(cache) > SUMMARY_CACHE_SIZE:
            cache.popitem(last=False)

    return result


def generate_batch_summaries(
    ai: "AIModelInterface",
    coord_studies_pairs: List[
        Tuple[Union[List[float], Tuple[float, float, float]], List[Dict[str, Any]]]
    ],
    prompt_type: str = "summary",
    model: str = "gemini-2.0-flash",
    custom_prompt: Optional[str] = None,
    max_tokens: int = 1000,
) -> List[str]:
    """Generate summaries for multiple coordinates.

    Parameters
    ----------
    ai : AIModelInterface
        AI backend used to create the summaries.
    coord_studies_pairs : list of tuple
        Coordinate-study pairs to summarise.
    prompt_type : str, optional
        Template key used for each summary prompt.
    model : str, optional
        Model used for text generation. Defaults to ``"gemini-2.0-flash"``.
    custom_prompt : str, optional
        Template string overriding the built-in prompt for every coordinate.
    max_tokens : int, optional
        Maximum tokens requested from each AI call.

    Returns
    -------
    list of str
        Generated summaries for the provided coordinate pairs.

    """
    if not coord_studies_pairs:
        return []

    if not ai.supports_batching(model):
        return [
            generate_summary(
                ai,
                studies,
                coord,
                prompt_type=prompt_type,
                model=model,
                custom_prompt=custom_prompt,
                max_tokens=max_tokens,
            )
            for coord, studies in coord_studies_pairs
        ]

    delimiter = "\n@@@\n"
    prompts: List[str] = []
    for coord, studies in coord_studies_pairs:
        prompts.append(
            generate_llm_prompt(
                studies,
                coord,
                prompt_type=prompt_type,
                prompt_template=custom_prompt if prompt_type == "custom" else None,
            )
        )

    combined_prompt = (
        "Provide separate summaries for each coordinate below. "
        f"Separate each summary with the delimiter '{delimiter.strip()}'.\n\n"
        + delimiter.join(prompts)
    )

    key = (model, combined_prompt)
    cache: "OrderedDict[Tuple[str, str], List[str]]" = generate_batch_summaries._cache
    if SUMMARY_CACHE_SIZE > 0:
        cached = cache.get(key)
        if cached is not None:
            cache.move_to_end(key)
            return cached

    response = ai.generate_text(
        model=model, prompt=combined_prompt, max_tokens=max_tokens
    )
    results = [part.strip() for part in response.split(delimiter) if part.strip()]

    if SUMMARY_CACHE_SIZE > 0:
        cache[key] = results
        cache.move_to_end(key)
        while len(cache) > SUMMARY_CACHE_SIZE:
            cache.popitem(last=False)

    return results


async def generate_summary_async(
    ai: "AIModelInterface",
    studies: List[Dict[str, Any]],
    coordinate: Union[List[float], Tuple[float, float, float]],
    prompt_type: str = "summary",
    model: str = "gemini-2.0-flash",
    atlas_labels: Optional[Dict[str, str]] = None,
    custom_prompt: Optional[str] = None,
    max_tokens: int = 1000,
) -> str:
    """Asynchronously generate a text summary for a coordinate.

    Parameters
    ----------
    ai : AIModelInterface
        AI backend used to create the summary asynchronously.
    studies : list of dict
        Studies reporting activation at the target coordinate.
    coordinate : sequence of float
        MNI coordinate for the summary.
    prompt_type : str, optional
        Prompt template key defaulting to ``"summary"``.
    model : str, optional
        Model name, defaulting to ``"gemini-2.0-flash"``.
    atlas_labels : dict, optional
        Atlas-derived labels to include in the prompt.
    custom_prompt : str, optional
        User-supplied template applied via ``str.format``.
    max_tokens : int, optional
        Maximum number of tokens requested for the summary.

    Returns
    -------
    str
        Generated summary text.
    """
    prompt = generate_llm_prompt(
        studies,
        coordinate,
        prompt_type=prompt_type,
        prompt_template=custom_prompt if prompt_type == "custom" else None,
    )

    if atlas_labels:
        parts = prompt.split("STUDIES REPORTING ACTIVATION AT MNI COORDINATE")
        atlas_info = "\nATLAS LABELS FOR THIS COORDINATE:\n"
        for atlas_name, label in atlas_labels.items():
            atlas_info += f"- {atlas_name}: {label}\n"
        if len(parts) >= 2:
            intro = parts[0]
            rest = "STUDIES REPORTING ACTIVATION AT MNI COORDINATE" + parts[1]
            prompt = intro + atlas_info + "\n" + rest
        else:
            prompt = atlas_info + prompt

    key = (model, prompt)
    cache: "OrderedDict[Tuple[str, str], str]" = generate_summary_async._cache
    if SUMMARY_CACHE_SIZE > 0:
        cached = cache.get(key)
        if cached is not None:
            cache.move_to_end(key)
            return cached

    result = await ai.generate_text_async(
        model=model, prompt=prompt, max_tokens=max_tokens
    )

    if SUMMARY_CACHE_SIZE > 0:
        cache[key] = result
        cache.move_to_end(key)
        while len(cache) > SUMMARY_CACHE_SIZE:
            cache.popitem(last=False)

    return result


def stream_summary(
    ai: "AIModelInterface",
    studies: List[Dict[str, Any]],
    coordinate: Union[List[float], Tuple[float, float, float]],
    prompt_type: str = "summary",
    model: str = "gemini-2.0-flash",
    atlas_labels: Optional[Dict[str, str]] = None,
    custom_prompt: Optional[str] = None,
    max_tokens: int = 1000,
) -> Iterator[str]:
    """Stream a text summary for a coordinate in chunks.

    Parameters
    ----------
    ai : AIModelInterface
        Streaming AI backend used to generate the summary.
    studies : list of dict
        Studies reporting activation at the target coordinate.
    coordinate : sequence of float
        MNI coordinate for the summary.
    prompt_type : str, optional
        Prompt template key defaulting to ``"summary"``.
    model : str, optional
        Model name, defaulting to ``"gemini-2.0-flash"``.
    atlas_labels : dict, optional
        Atlas-derived labels to include in the prompt.
    custom_prompt : str, optional
        User-supplied template applied via ``str.format``.
    max_tokens : int, optional
        Maximum number of tokens requested for the summary.

    Returns
    -------
    iterator of str
        Chunks of text yielded by the streaming AI backend.
    """
    prompt = generate_llm_prompt(
        studies,
        coordinate,
        prompt_type=prompt_type,
        prompt_template=custom_prompt if prompt_type == "custom" else None,
    )

    if atlas_labels:
        parts = prompt.split("STUDIES REPORTING ACTIVATION AT MNI COORDINATE")
        atlas_info = "\nATLAS LABELS FOR THIS COORDINATE:\n"
        for atlas_name, label in atlas_labels.items():
            atlas_info += f"- {atlas_name}: {label}\n"
        if len(parts) >= 2:
            intro = parts[0]
            rest = "STUDIES REPORTING ACTIVATION AT MNI COORDINATE" + parts[1]
            prompt = intro + atlas_info + "\n" + rest
        else:
            prompt = atlas_info + prompt

    key = (model, prompt)
    cache: "OrderedDict[Tuple[str, str], List[str]]" = stream_summary._cache
    if SUMMARY_CACHE_SIZE > 0:
        cached_chunks = cache.get(key)
        if cached_chunks is not None:
            cache.move_to_end(key)
            for chunk in cached_chunks:
                yield chunk
            return

    chunks: List[str] = []
    try:
        for chunk in ai.stream_generate_text(
            model=model, prompt=prompt, max_tokens=max_tokens
        ):
            chunks.append(chunk)
            yield chunk
    finally:
        if SUMMARY_CACHE_SIZE > 0 and chunks:
            cache[key] = chunks
            cache.move_to_end(key)
            while len(cache) > SUMMARY_CACHE_SIZE:
                cache.popitem(last=False)


generate_summary._cache = OrderedDict()
generate_batch_summaries._cache = OrderedDict()
generate_summary_async._cache = OrderedDict()
stream_summary._cache = OrderedDict()


__all__ = [
    "LLM_PROMPT_TEMPLATES",
    "IMAGE_PROMPT_TEMPLATES",
    "generate_llm_prompt",
    "generate_region_image_prompt",
    "generate_region_image",
    "generate_mni152_image",
    "generate_summary",
    "generate_batch_summaries",
    "generate_summary_async",
    "stream_summary",
]
