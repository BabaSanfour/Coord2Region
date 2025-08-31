"""Unit tests for coord2region.llm."""

from unittest.mock import MagicMock, patch

from coord2region.llm import (
    IMAGE_PROMPT_TEMPLATES,
    LLM_PROMPT_TEMPLATES,
    generate_llm_prompt,
    generate_region_image_prompt,
    generate_summary,
)


def _sample_studies():
    """Return a minimal list of study dictionaries for testing."""
    return [{"id": "1", "title": "A", "abstract": "B"}]


# ---------------------------------------------------------------------------
# Template exposure tests
# ---------------------------------------------------------------------------


def test_llm_prompt_templates_accessible():
    """LLM prompt templates are exposed for inspection."""
    assert "summary" in LLM_PROMPT_TEMPLATES
    assert "region_name" in LLM_PROMPT_TEMPLATES


def test_image_prompt_templates_accessible():
    """Image prompt templates are exposed for inspection."""
    assert "anatomical" in IMAGE_PROMPT_TEMPLATES
    assert "functional" in IMAGE_PROMPT_TEMPLATES


# ---------------------------------------------------------------------------
# generate_llm_prompt tests
# ---------------------------------------------------------------------------


def test_generate_llm_prompt_no_studies():
    """An informative message is returned when no studies are supplied."""
    msg = generate_llm_prompt([], [1, 2, 3])
    assert "No neuroimaging studies" in msg


def test_generate_llm_prompt_summary():
    """Summary prompts use the corresponding template."""
    prompt = generate_llm_prompt(_sample_studies(), [1, 2, 3])
    expected_intro = LLM_PROMPT_TEMPLATES["summary"].format(coord="[1.00, 2.00, 3.00]")
    assert prompt.startswith(expected_intro)
    assert "ID: 1" in prompt


def test_generate_llm_prompt_region_name():
    """Region-name prompts come from the template dictionary."""
    prompt = generate_llm_prompt(_sample_studies(), [1, 2, 3], prompt_type="region_name")
    expected_intro = LLM_PROMPT_TEMPLATES["region_name"].format(coord="[1.00, 2.00, 3.00]")
    assert prompt.startswith(expected_intro)


def test_generate_llm_prompt_function():
    """Function prompts come from the template dictionary."""
    prompt = generate_llm_prompt(_sample_studies(), [1, 2, 3], prompt_type="function")
    expected_intro = LLM_PROMPT_TEMPLATES["function"].format(coord="[1.00, 2.00, 3.00]")
    assert prompt.startswith(expected_intro)


def test_generate_llm_prompt_unsupported_type():
    """Unsupported prompt types fall back to the default template."""
    prompt = generate_llm_prompt(_sample_studies(), [1, 2, 3], prompt_type="other")
    expected_intro = LLM_PROMPT_TEMPLATES["default"].format(coord="[1.00, 2.00, 3.00]")
    assert prompt.startswith(expected_intro)


def test_generate_llm_prompt_custom_template():
    """Custom templates override built-in formatting."""
    template = "Coordinate: {coord}\n{studies}"
    prompt = generate_llm_prompt(_sample_studies(), [1, 2, 3], prompt_template=template)
    assert prompt.startswith("Coordinate: [1.00, 2.00, 3.00]")
    assert "ID: 1" in prompt


# ---------------------------------------------------------------------------
# generate_region_image_prompt tests
# ---------------------------------------------------------------------------


def test_generate_region_image_prompt_anatomical_with_atlas():
    """Anatomical image prompts include atlas context when available."""
    region_info = {
        "summary": "Paragraph one.\n\nParagraph two.",
        "atlas_labels": {"Atlas": "Label"},
    }
    prompt = generate_region_image_prompt([1, 2, 3], region_info)
    assert "anatomical illustration" in prompt
    assert "Atlas: Label" in prompt


def test_generate_region_image_prompt_functional_no_atlas():
    """Functional image prompts work without atlas labels."""
    region_info = {"summary": "Single paragraph"}
    prompt = generate_region_image_prompt([1, 2, 3], region_info, image_type="functional")
    assert "functional brain activation" in prompt
    assert "According to brain atlases" not in prompt


def test_generate_region_image_prompt_schematic_no_include():
    """Atlas labels can be omitted from schematic prompts."""
    region_info = {
        "summary": "Para.\n\nMore.",
        "atlas_labels": {"Atlas": "Label"},
    }
    prompt = generate_region_image_prompt(
        [1, 2, 3], region_info, image_type="schematic", include_atlas_labels=False
    )
    assert "schematic diagram" in prompt
    assert "Atlas: Label" not in prompt


def test_generate_region_image_prompt_artistic():
    """Artistic prompts balance creativity with accuracy."""
    region_info = {
        "summary": "Summary.\n\nDetails.",
        "atlas_labels": {"Atlas": "Label"},
    }
    prompt = generate_region_image_prompt([1, 2, 3], region_info, image_type="artistic")
    assert "artistic visualization" in prompt


def test_generate_region_image_prompt_unknown_type():
    """Unknown image types fall back to a generic prompt."""
    region_info = {"summary": "Just one paragraph"}
    prompt = generate_region_image_prompt([1, 2, 3], region_info, image_type="other")
    assert "Create a clear visualization" in prompt


def test_generate_region_image_prompt_custom_template():
    """Custom templates override default image prompts."""
    region_info = {"summary": "Single paragraph"}
    template = "Custom image for {coordinate} -> {first_paragraph} || {atlas_context}"
    prompt = generate_region_image_prompt([1, 2, 3], region_info, prompt_template=template)
    assert prompt.startswith("Custom image for [1.00, 2.00, 3.00]")


# ---------------------------------------------------------------------------
# generate_summary tests
# ---------------------------------------------------------------------------


@patch("coord2region.llm.generate_llm_prompt", return_value="PROMPT")
def test_generate_summary_calls_ai(mock_prompt):
    ai = MagicMock()
    ai.generate_text.return_value = "SUMMARY"
    studies = _sample_studies()
    coord = [1, 2, 3]

    result = generate_summary(ai, studies, coord)

    mock_prompt.assert_called_once()
    ai.generate_text.assert_called_once_with(model="gemini-2.0-flash", prompt="PROMPT", max_tokens=1000)
    assert result == "SUMMARY"


@patch("coord2region.llm.generate_llm_prompt")
def test_generate_summary_includes_atlas_labels(mock_prompt):
    base = "Intro\nSTUDIES REPORTING ACTIVATION AT MNI COORDINATE more"
    mock_prompt.return_value = base
    ai = MagicMock()
    ai.generate_text.return_value = "SUMMARY"

    atlas_labels = {"Atlas": "Label"}
    generate_summary(ai, [], [1, 2, 3], atlas_labels=atlas_labels)

    prompt_used = ai.generate_text.call_args.kwargs["prompt"]
    assert "ATLAS LABELS FOR THIS COORDINATE" in prompt_used
    assert "Atlas: Label" in prompt_used

