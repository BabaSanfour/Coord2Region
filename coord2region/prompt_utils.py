from typing import Any, Dict, List, Optional, Tuple, Union


def generate_llm_prompt(
    studies: List[Dict[str, Any]],
    coordinate: Union[List[float], Tuple[float, float, float]],
    prompt_type: str = "summary",
    prompt_template: Optional[str] = None,
) -> str:
    """Generate a detailed prompt for language models based on studies found for a coordinate.

    This function creates a structured prompt that includes study IDs, titles, and abstracts
    formatted for optimal LLM analysis and summarization.

    Parameters
    ----------
    studies : List[Dict[str, Any]]
        List of study metadata dictionaries.
    coordinate : Union[List[float], Tuple[float, float, float]]
        The MNI coordinate [x, y, z] that was searched.
    prompt_type : str, default="summary"
        Type of prompt to generate ("summary", "region_name", "function", etc.).
    prompt_template : str, optional
        Custom prompt template. If provided, it should contain the placeholders
        ``{coord}`` for the coordinate string and ``{studies}`` for the formatted
        study list. When supplied, this template overrides the built-in prompt
        generation logic.

    Returns
    -------
    str
        A detailed prompt for language models, incorporating all relevant study info.
    """
    if not studies:
        return (
            "No neuroimaging studies were found reporting activation at MNI "
            f"coordinate {coordinate}."
        )

    coord_str = f"[{coordinate[0]}, {coordinate[1]}, {coordinate[2]}]"

    # Format study details once for reuse
    studies_section = ""
    for i, study in enumerate(studies, 1):
        studies_section += f"\n--- STUDY {i} ---\n"
        studies_section += f"ID: {study.get('id', 'Unknown ID')}\n"
        studies_section += f"Title: {study.get('title', 'No title available')}\n"
        abstract_text = study.get("abstract", "No abstract available")
        studies_section += f"Abstract: {abstract_text}\n"

    if prompt_template:
        return prompt_template.format(coord=coord_str, studies=studies_section)

    # Build the prompt header with clear instructions
    if prompt_type == "summary":
        prompt_intro = (
            "You are an advanced AI with expertise in neuroanatomy and cognitive "
            "neuroscience. "
            f"The user is interested in understanding the significance of MNI "
            f"coordinate {coord_str}.\n\n"
            "Below is a list of neuroimaging studies that report activation at "
            "this coordinate. Your task is to integrate and synthesize the "
            "knowledge from these studies, focusing on:\n"
            "1) The anatomical structure(s) most commonly associated with this coordinate\n"
            "2) The typical functional roles or processes linked to activation in this region\n"
            "3) The main tasks or experimental conditions in which it was reported\n"
            "4) Patterns, contradictions, or debates in the findings\n\n"
            "Do NOT simply list each study separately. Provide an integrated, cohesive summary.\n"
        )
    elif prompt_type == "region_name":
        prompt_intro = (
            "You are a neuroanatomy expert. The user wants to identify the probable "
            f"anatomical labels for MNI coordinate {coord_str}. The following "
            "studies reported activation around this location. Incorporate "
            "anatomical knowledge and any direct references to brain regions from "
            "these studies. If multiple labels are possible, mention all and "
            "provide rationale and confidence levels.\n\n"
        )
    elif prompt_type == "function":
        prompt_intro = (
            "You are a cognitive neuroscience expert. The user wants a deep "
            "functional profile of the brain region(s) around "
            f"MNI coordinate {coord_str}. The studies below report activation at "
            "or near this coordinate. Synthesize a clear description of:\n"
            "1) Core functions or cognitive processes\n"
            "2) Typical experimental paradigms or tasks\n"
            "3) Known functional networks or connectivity\n"
            "4) Divergent or debated viewpoints in the literature\n\n"
        )
    else:
        # Default to a basic integrated summary
        prompt_intro = (
            "Please analyze the following neuroimaging studies reporting activation at "
            f"MNI coordinate {coord_str} and provide a concise yet thorough "
            "discussion of its anatomical location and functional significance.\n\n"
        )

    # Add study details
    prompt_body = (
        "STUDIES REPORTING ACTIVATION AT MNI COORDINATE " + coord_str + ":\n" + studies_section
    )

    # Final instructions
    prompt_outro = (
        "\nUsing ALL of the information above, produce a single cohesive synthesis. "
        "Avoid bullet-by-bullet summaries of each study. Instead, integrate the findings "
        "across them to describe the region's location, function, and context."
    )

    return prompt_intro + prompt_body + prompt_outro


def generate_region_image_prompt(
    coordinate: Union[List[float], Tuple[float, float, float]],
    region_info: Dict[str, Any],
    image_type: str = "anatomical",
    include_atlas_labels: bool = True,
) -> str:
    """Generate a prompt for creating images of brain regions.

    Parameters
    ----------
    coordinate : Union[List[float], Tuple[float, float, float]]
        MNI coordinate [x, y, z].
    region_info : Dict[str, Any]
        Dictionary returned by :meth:`BrainInsights.get_region_summary` containing at
        least a ``"summary"`` key and optionally ``"atlas_labels"``.
    image_type : str, default="anatomical"
        Type of image to generate (anatomical, functional, schematic, artistic).
    include_atlas_labels : bool, default=True
        Whether to include atlas labels in the region information.

    Returns
    -------
    str
        A detailed prompt for image generation models.
    """
    summary = region_info["summary"]
    first_paragraph = summary.split("\n\n")[0] if "\n\n" in summary else summary

    atlas_context = ""
    if include_atlas_labels and region_info.get("atlas_labels"):
        atlas_context = "According to brain atlases, this region corresponds to: "
        atlas_parts = [
            f"{atlas_name}: {label}" for atlas_name, label in region_info["atlas_labels"].items()
        ]
        atlas_context += ", ".join(atlas_parts) + ". "

    if image_type == "anatomical":
        prompt = f"""Create a detailed anatomical illustration of the brain region at MNI coordinate {coordinate}.
Based on neuroimaging studies, this location corresponds to: {first_paragraph}
{atlas_context}Show a clear, labeled anatomical visualization with the specific coordinate marked.
Include surrounding brain structures for context. Use a professional medical illustration style with
accurate colors and textures of brain tissue."""

    elif image_type == "functional":
        prompt = f"""Create a functional brain activation visualization showing activity at MNI coordinate {coordinate}.
This region corresponds to: {first_paragraph}
{atlas_context}Show the activation as a heat map or colored overlay on a standardized brain template.
Use a scientific visualization style similar to fMRI results in neuroscience publications,
with the activation at the specified coordinate clearly highlighted."""

    elif image_type == "schematic":
        prompt = f"""Create a schematic diagram of brain networks involving the region at MNI coordinate {coordinate}.
This coordinate corresponds to: {first_paragraph}
{atlas_context}Show this region as a node in its relevant brain networks, with connections to other regions.
Use a simplified, clean diagram style with labeled regions and connection lines indicating functional
or structural connectivity. Include a small reference brain to indicate the location."""

    elif image_type == "artistic":
        prompt = f"""Create an artistic visualization of the brain region at MNI coordinate {coordinate}.
This region is: {first_paragraph}
{atlas_context}Create an artistic interpretation that conveys the function of this region through metaphorical
or abstract elements, while still maintaining scientific accuracy in the brain anatomy.
Balance creativity with neuroscientific precision."""

    else:
        prompt = f"""Create a clear visualization of the brain region at MNI coordinate {coordinate}.
Based on neuroimaging studies, this region corresponds to: {first_paragraph}
{atlas_context}Show this region clearly marked on a standard brain template with proper anatomical context."""

    return prompt
