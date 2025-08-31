"""LLM utilities for prompt construction and summary generation.

The summary generation helpers provide an in-memory LRU cache keyed by
``(model, prompt)``. The cache size can be controlled with the
``cache_size`` parameter on the public functions; setting it to ``0``
disables caching.
"""

from collections import OrderedDict
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from .image_utils import generate_mni152_image

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
# ``coordinate``, ``first_paragraph``, and ``atlas_context`` variables.
IMAGE_PROMPT_TEMPLATES: Dict[str, str] = {
    "anatomical": (
        "Create a detailed anatomical illustration of the brain region at MNI "
        "coordinate {coordinate}.\nBased on neuroimaging studies, this location "
        "corresponds to: {first_paragraph}\n"
        "{atlas_context}Show a clear, labeled anatomical visualization with the "
        "specific coordinate marked. Include surrounding brain structures for "
        "context. Use a professional medical illustration style with accurate "
        "colors and textures of brain tissue."
    ),
    "functional": (
        "Create a functional brain activation visualization showing activity at "
        "MNI coordinate {coordinate}.\nThis region corresponds to: {first_paragraph}\n"
        "{atlas_context}Show the activation as a heat map or colored overlay on a "
        "standardized brain template. Use a scientific visualization style similar "
        "to fMRI results in neuroscience publications, with the activation at the "
        "specified coordinate clearly highlighted."
    ),
    "schematic": (
        "Create a schematic diagram of brain networks involving the region at "
        "MNI coordinate {coordinate}.\nThis coordinate corresponds to: "
        "{first_paragraph}\n{atlas_context}Show this region as a node in its "
        "relevant brain networks, with connections to other regions. Use a "
        "simplified, clean diagram style with labeled regions and connection lines "
        "indicating functional or structural connectivity. Include a small reference "
        "brain to indicate the location."
    ),
    "artistic": (
        "Create an artistic visualization of the brain region at MNI coordinate "
        "{coordinate}.\nThis region is: {first_paragraph}\n"
        "{atlas_context}Create an artistic interpretation that conveys the function "
        "of this region through metaphorical or abstract elements, while still "
        "maintaining scientific accuracy in the brain anatomy. Balance creativity "
        "with neuroscientific precision."
    ),
    "default": (
        "Create a clear visualization of the brain region at MNI coordinate "
        "{coordinate}.\n"
        "Based on neuroimaging studies, this region corresponds to: {first_paragraph}\n"
        "{atlas_context}Show this region clearly marked on a standard brain template "
        "with proper anatomical context."
    ),
}


def generate_llm_prompt(
    studies: List[Dict[str, Any]],
    coordinate: Union[List[float], Tuple[float, float, float]],
    prompt_type: str = "summary",
    prompt_template: Optional[str] = None,
) -> str:
    """Generate a detailed prompt for language models based on studies."""
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
    """Generate a prompt for creating images of brain regions."""
    # Safely get the summary and a short first paragraph.
    summary = region_info.get("summary", "No summary available.")
    first_paragraph = summary.split("\n\n", 1)[0]

    # Format the coordinate for inclusion in the prompt.
    try:
        coord_str = "[{:.2f}, {:.2f}, {:.2f}]".format(
            float(coordinate[0]), float(coordinate[1]), float(coordinate[2])
        )
    except Exception:
        # Fallback to the raw coordinate representation.
        coord_str = str(coordinate)

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

    # If a custom template is provided, use it directly.
    if prompt_template:
        return prompt_template.format(
            coordinate=coord_str,
            first_paragraph=first_paragraph,
            atlas_context=atlas_context,
        )

    # Construct prompts for supported image types.
    if image_type == "anatomical":
        prompt = (
            "Create a detailed anatomical illustration of the brain region at "
            f"MNI coordinate {coord_str}. Based on neuroimaging studies, this "
            f"location corresponds to: {first_paragraph} {atlas_context}Show a "
            "clear, labeled anatomical visualization with the specific "
            "coordinate marked. Include surrounding brain structures for "
            "context. Use a professional medical illustration style with "
            "accurate colors and textures of brain tissue."
        )

    elif image_type == "functional":
        prompt = (
            "Create a functional brain activation visualization showing activity "
            f"at MNI coordinate {coord_str}. This region corresponds to: "
            f"{first_paragraph} {atlas_context}Show the activation as a heat "
            "map or colored overlay on a standardized brain template. Use a "
            "scientific visualization style similar to fMRI results in "
            "neuroscience publications, with the activation at the specified "
            "coordinate clearly highlighted."
        )

    elif image_type == "schematic":
        prompt = (
            "Create a schematic diagram of brain networks involving the region "
            f"at MNI coordinate {coord_str}. This coordinate corresponds to: "
            f"{first_paragraph} {atlas_context}Show this region as a node in "
            "its relevant brain networks, with connections to other regions. "
            "Use a simplified, clean diagram style with labeled regions and "
            "connection lines indicating functional or structural connectivity. "
            "Include a small reference brain to indicate the location."
        )

    elif image_type == "artistic":
        prompt = (
            "Create an artistic visualization of the brain region at MNI "
            f"coordinate {coord_str}. This region is: {first_paragraph} "
            f"{atlas_context}Create an artistic interpretation that conveys "
            "the function of this region through metaphorical or abstract "
            "elements, while still maintaining scientific accuracy in the "
            "brain anatomy. Balance creativity with neuroscientific precision."
        )

    else:
        prompt = (
            "Create a clear visualization of the brain region at MNI coordinate "
            f"{coord_str}. Based on neuroimaging studies, this region "
            f"corresponds to: {first_paragraph} {atlas_context}Show this region "
            "clearly marked on a standard brain template with proper anatomical "
            "context."
        )

    return prompt


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

try:  # Optional dependency
    from .ai_model_interface import AIModelInterface
except Exception:  # pragma: no cover - used only when dependencies missing
    AIModelInterface = None  # type: ignore


def generate_region_image(
    ai: "AIModelInterface",
    coordinate: Union[List[float], Tuple[float, float, float]],
    region_info: Dict[str, Any],
    image_type: str = "anatomical",
    model: str = "stabilityai/stable-diffusion-2",
    include_atlas_labels: bool = True,
    prompt_template: Optional[str] = None,
    retries: int = 3,
    **kwargs: Any,
) -> bytes:
    """Generate an image for a brain region using an AI model."""
    prompt = generate_region_image_prompt(
        coordinate,
        region_info,
        image_type=image_type,
        include_atlas_labels=include_atlas_labels,
        prompt_template=prompt_template,
    )
    return ai.generate_image(
        model=model, prompt=prompt, retries=retries, **kwargs
    )


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------


def generate_summary(
    ai: "AIModelInterface",
    studies: List[Dict[str, Any]],
    coordinate: Union[List[float], Tuple[float, float, float]],
    summary_type: str = "summary",
    model: str = "gemini-2.0-flash",
    atlas_labels: Optional[Dict[str, str]] = None,
    prompt_template: Optional[str] = None,
    max_tokens: int = 1000,
    cache_size: int = 128,
) -> str:
    """Generate a text summary for a coordinate based on studies.

    Results are cached in an LRU cache keyed by ``(model, prompt)``. Use
    ``cache_size=0`` to disable caching or increase the size if needed.
    """
    # Build base prompt with study information
    prompt = generate_llm_prompt(
        studies,
        coordinate,
        prompt_type=summary_type,
        prompt_template=prompt_template,
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
    if cache_size > 0:
        cached = cache.get(key)
        if cached is not None:
            cache.move_to_end(key)
            return cached

    result = ai.generate_text(model=model, prompt=prompt, max_tokens=max_tokens)

    if cache_size > 0:
        cache[key] = result
        cache.move_to_end(key)
        while len(cache) > cache_size:
            cache.popitem(last=False)

    return result


def generate_batch_summaries(
    ai: "AIModelInterface",
    coord_studies_pairs: List[
        Tuple[Union[List[float], Tuple[float, float, float]], List[Dict[str, Any]]]
    ],
    summary_type: str = "summary",
    model: str = "gemini-2.0-flash",
    prompt_template: Optional[str] = None,
    max_tokens: int = 1000,
    cache_size: int = 128,
) -> List[str]:
    """Generate summaries for multiple coordinates.

    If the underlying provider reports that it supports batching, the prompts
    for all coordinates are combined into a single request and the response is
    split using an internal delimiter. Otherwise, each summary is generated
    sequentially via :func:`generate_summary`.
    """

    if not coord_studies_pairs:
        return []

    if not ai.supports_batching(model):
        return [
            generate_summary(
                ai,
                studies,
                coord,
                summary_type=summary_type,
                model=model,
                prompt_template=prompt_template,
                max_tokens=max_tokens,
                cache_size=cache_size,
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
                prompt_type=summary_type,
                prompt_template=prompt_template,
            )
        )

    combined_prompt = (
        "Provide separate summaries for each coordinate below. "
        f"Separate each summary with the delimiter '{delimiter.strip()}'.\n\n"
        + delimiter.join(prompts)
    )

    key = (model, combined_prompt)
    cache: "OrderedDict[Tuple[str, str], List[str]]" = generate_batch_summaries._cache
    if cache_size > 0:
        cached = cache.get(key)
        if cached is not None:
            cache.move_to_end(key)
            return cached

    response = ai.generate_text(model=model, prompt=combined_prompt, max_tokens=max_tokens)
    results = [part.strip() for part in response.split(delimiter) if part.strip()]

    if cache_size > 0:
        cache[key] = results
        cache.move_to_end(key)
        while len(cache) > cache_size:
            cache.popitem(last=False)

    return results


async def generate_summary_async(
    ai: "AIModelInterface",
    studies: List[Dict[str, Any]],
    coordinate: Union[List[float], Tuple[float, float, float]],
    summary_type: str = "summary",
    model: str = "gemini-2.0-flash",
    atlas_labels: Optional[Dict[str, str]] = None,
    prompt_template: Optional[str] = None,
    max_tokens: int = 1000,
    cache_size: int = 128,
) -> str:
    """Asynchronously generate a text summary for a coordinate.

    Results are cached in an LRU cache keyed by ``(model, prompt)``.
    """
    prompt = generate_llm_prompt(
        studies,
        coordinate,
        prompt_type=summary_type,
        prompt_template=prompt_template,
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
    if cache_size > 0:
        cached = cache.get(key)
        if cached is not None:
            cache.move_to_end(key)
            return cached

    result = await ai.generate_text_async(
        model=model, prompt=prompt, max_tokens=max_tokens
    )

    if cache_size > 0:
        cache[key] = result
        cache.move_to_end(key)
        while len(cache) > cache_size:
            cache.popitem(last=False)

    return result


def stream_summary(
    ai: "AIModelInterface",
    studies: List[Dict[str, Any]],
    coordinate: Union[List[float], Tuple[float, float, float]],
    summary_type: str = "summary",
    model: str = "gemini-2.0-flash",
    atlas_labels: Optional[Dict[str, str]] = None,
    prompt_template: Optional[str] = None,
    max_tokens: int = 1000,
    cache_size: int = 128,
) -> Iterator[str]:
    """Stream a text summary for a coordinate in chunks.

    Responses are cached and subsequent calls with the same ``model`` and
    ``prompt`` will yield cached chunks. Disable caching with
    ``cache_size=0``.
    """
    prompt = generate_llm_prompt(
        studies,
        coordinate,
        prompt_type=summary_type,
        prompt_template=prompt_template,
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
    if cache_size > 0:
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
        if cache_size > 0 and chunks:
            cache[key] = chunks
            cache.move_to_end(key)
            while len(cache) > cache_size:
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
