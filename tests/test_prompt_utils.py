"""Unit tests for coord2region.prompt_utils."""

from coord2region.prompt_utils import (
    IMAGE_PROMPT_TEMPLATES,
    LLM_PROMPT_TEMPLATES,
    generate_llm_prompt,
    generate_region_image_prompt,
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
    """Default summary prompts contain study details and coordinate."""
    prompt = generate_llm_prompt(_sample_studies(), [1, 2, 3])
    assert "[1.00, 2.00, 3.00]" in prompt
    assert "ID: 1" in prompt


def test_generate_llm_prompt_region_name():
    """Region-name prompt mentions anatomical labels."""
    prompt = generate_llm_prompt(
        _sample_studies(), [1, 2, 3], prompt_type="region_name"
    )
    assert "probable anatomical labels" in prompt


def test_generate_llm_prompt_function():
    """Function prompt references cognitive processes."""
    prompt = generate_llm_prompt(
        _sample_studies(), [1, 2, 3], prompt_type="function"
    )
    assert "functional profile" in prompt


def test_generate_llm_prompt_unknown_type():
    """Unknown prompt types fall back to the default template."""
    prompt = generate_llm_prompt(
        _sample_studies(), [1, 2, 3], prompt_type="other"
    )
    assert "Please analyze" in prompt


def test_generate_llm_prompt_custom_template():
    """Custom templates override built-in formatting."""
    template = "Coordinate: {coord}\n{studies}"
    prompt = generate_llm_prompt(
        _sample_studies(), [1, 2, 3], prompt_template=template
    )
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
    prompt = generate_region_image_prompt(
        [1, 2, 3], region_info, image_type="functional"
    )
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
    prompt = generate_region_image_prompt(
        [1, 2, 3], region_info, image_type="artistic"
    )
    assert "artistic visualization" in prompt


def test_generate_region_image_prompt_unknown_type():
    """Unknown image types fall back to a generic prompt."""
    region_info = {"summary": "Just one paragraph"}
    prompt = generate_region_image_prompt(
        [1, 2, 3], region_info, image_type="other"
    )
    assert "Create a clear visualization" in prompt

def test_generate_region_image_prompt_custom_template():
    """Custom templates override default image prompts."""
    region_info = {"summary": "Single paragraph"}
    template = (
        "Custom image for {coordinate} -> {first_paragraph} || {atlas_context}"
    )
    prompt = generate_region_image_prompt(
        [1, 2, 3], region_info, prompt_template=template
    )
    assert prompt.startswith("Custom image for [1.00, 2.00, 3.00]")
