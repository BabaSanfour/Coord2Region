"""Utilities for constructing prompts for language and image models."""

from typing import Any, Dict, List, Optional, Tuple, Union

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
    """Generate a detailed prompt for language models based on studies.

    This function creates a structured prompt that includes study IDs,
    titles, and abstracts formatted for optimal LLM analysis and
    summarization.

    Parameters
    ----------
    studies : List[Dict[str, Any]]
        List of study metadata dictionaries.
    coordinate : Union[List[float], Tuple[float, float, float]]
        The MNI coordinate [x, y, z] that was searched.
    prompt_type : str, optional
        Type of prompt to generate ("summary", "region_name", "function",
        etc.). Default is "summary".
    prompt_template : str, optional
        Custom prompt template. If provided, it should contain the
        placeholders ``{coord}`` for the coordinate string and
        ``{studies}`` for the formatted study list. When supplied, this
        template overrides the built-in prompt generation logic.

    Returns
    -------
    str
        A detailed prompt for language models, incorporating all relevant
        study information.
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

    # If a custom template is provided, use it.
    if prompt_template:
        return prompt_template.format(coord=coord_str, studies=studies_section)

    # Build the prompt header with clear instructions depending on type.
    if prompt_type == "summary":
        prompt_intro = (
            "You are an advanced AI with expertise in neuroanatomy and "
            "cognitive neuroscience. The user is interested in understanding "
            f"the significance of MNI coordinate {coord_str}.\n\n"
            "Below is a list of neuroimaging studies that report activation at "
            "this coordinate. Your task is to integrate and synthesize the "
            "knowledge from these studies, focusing on:\n"
            "1) The anatomical structure(s) most commonly associated with this "
            "coordinate\n"
            "2) The typical functional roles or processes linked to activation "
            "in this region\n"
            "3) The main tasks or experimental conditions in which it was "
            "reported\n"
            "4) Patterns, contradictions, or debates in the findings\n\n"
            "Do NOT simply list each study separately. Provide an integrated, "
            "cohesive summary.\n"
        )
    elif prompt_type == "region_name":
        prompt_intro = (
            "You are a neuroanatomy expert. The user wants to identify the "
            "probable anatomical labels for MNI coordinate "
            f"{coord_str}. The following studies reported activation around "
            "this location. Incorporate anatomical knowledge and any direct "
            "references to brain regions from these studies. If multiple "
            "labels are possible, mention all and provide rationale and "
            "confidence levels.\n\n"
        )
    elif prompt_type == "function":
        prompt_intro = (
            "You are a cognitive neuroscience expert. The user wants a deep "
            "functional profile of the brain region(s) around "
            f"{coord_str}. The studies below report activation at or near this "
            "coordinate. Synthesize a clear description of:\n"
            "1) Core functions or cognitive processes\n"
            "2) Typical experimental paradigms or tasks\n"
            "3) Known functional networks or connectivity\n"
            "4) Divergent or debated viewpoints in the literature\n\n"
        )
    else:
        prompt_intro = (
            "Please analyze the following neuroimaging studies reporting "
            f"activation at MNI coordinate {coord_str} and provide a concise "
            "yet thorough discussion of its anatomical location and functional "
            "significance.\n\n"
        )

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
) -> str:
    """Generate a prompt for creating images of brain regions.

    The prompt summarizes the region information and instructs an image
    generation model to produce an anatomical, functional, schematic, or
    artistic visualization for the given MNI coordinate.

    Args:
        coordinate: MNI coordinate as [x, y, z] or (x, y, z).
        region_info: Dictionary containing at least a "summary" key and
            optionally an "atlas_labels" mapping of atlas names to labels.
        image_type: Type of image to generate. One of: "anatomical",
            "functional", "schematic", "artistic". Defaults to "anatomical".
        include_atlas_labels: Whether to include atlas labels from
            region_info in the prompt. Defaults to True.

    Returns:
        A detailed prompt string suitable for image generation models.
    """
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
