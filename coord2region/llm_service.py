"""LLM service utilities for text and summary generation."""
from typing import Any, Dict, List, Optional, Tuple, Union

from .prompt_utils import generate_llm_prompt
try:  # Optional dependency
    from .ai_model_interface import AIModelInterface
except Exception:  # pragma: no cover - used only when dependencies missing
    AIModelInterface = None  # type: ignore


def generate_summary(
    ai: "AIModelInterface",
    studies: List[Dict[str, Any]],
    coordinate: Union[List[float], Tuple[float, float, float]],
    summary_type: str = "summary",
    model: str = "gemini-2.0-flash",
    atlas_labels: Optional[Dict[str, str]] = None,
    prompt_template: Optional[str] = None,
    max_tokens: int = 1000,
) -> str:
    """Generate a text summary for a coordinate based on studies.

    Parameters
    ----------
    ai : AIModelInterface
        Initialized AI model interface used for text generation.
    studies : List[Dict[str, Any]]
        List of study metadata dictionaries.
    coordinate : Union[List[float], Tuple[float, float, float]]
        MNI coordinate [x, y, z].
    summary_type : str, optional
        Type of summary to generate (``"summary"``, ``"region_name"``, ``"function"``).
    model : str, optional
        Name of the model to use for generation.
    atlas_labels : Dict[str, str], optional
        Mapping of atlas names to region labels to include in the prompt.
    prompt_template : str, optional
        Custom template passed to :func:`generate_llm_prompt`.
    max_tokens : int, optional
        Maximum number of tokens to generate.

    Returns
    -------
    str
        Generated summary text.
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

    # Generate and return the summary using the AI interface
    return ai.generate_text(model=model, prompt=prompt, max_tokens=max_tokens)
