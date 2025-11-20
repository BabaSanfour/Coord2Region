"""Unit tests for coord2region.llm."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from coord2region.llm import (
    IMAGE_PROMPT_TEMPLATES,
    LLM_PROMPT_TEMPLATES,
    generate_llm_prompt,
    generate_region_image_prompt,
    generate_region_image,
    generate_summary,
    generate_batch_summaries,
    generate_summary_async,
    stream_summary,
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
    prompt = generate_llm_prompt(
        _sample_studies(), [1, 2, 3], prompt_type="region_name"
    )
    expected_intro = LLM_PROMPT_TEMPLATES["region_name"].format(
        coord="[1.00, 2.00, 3.00]"
    )
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
    coord = "[1.00, 2.00, 3.00]"
    atlas = "According to brain atlases, this region corresponds to: Atlas: Label. "
    expected = IMAGE_PROMPT_TEMPLATES["anatomical"].format(
        coordinate=coord,
        first_paragraph="Paragraph one.",
        atlas_context=atlas,
        x_coord="1",
        y_coord="2",
        z_coord="3",
        study_context="",
    )
    assert prompt == expected


def test_generate_region_image_prompt_functional_no_atlas():
    """Functional image prompts work without atlas labels."""
    region_info = {"summary": "Single paragraph"}
    prompt = generate_region_image_prompt(
        [1, 2, 3], region_info, image_type="functional"
    )
    coord = "[1.00, 2.00, 3.00]"
    expected = IMAGE_PROMPT_TEMPLATES["functional"].format(
        coordinate=coord,
        first_paragraph="Single paragraph",
        atlas_context="",
        x_coord="1",
        y_coord="2",
        z_coord="3",
        study_context="",
    )
    assert prompt == expected


def test_generate_region_image_prompt_schematic_no_include():
    """Atlas labels can be omitted from schematic prompts."""
    region_info = {
        "summary": "Para.\n\nMore.",
        "atlas_labels": {"Atlas": "Label"},
    }
    prompt = generate_region_image_prompt(
        [1, 2, 3], region_info, image_type="schematic", include_atlas_labels=False
    )
    coord = "[1.00, 2.00, 3.00]"
    expected = IMAGE_PROMPT_TEMPLATES["schematic"].format(
        coordinate=coord,
        first_paragraph="Para.",
        atlas_context="",
        x_coord="1",
        y_coord="2",
        z_coord="3",
        study_context="",
    )
    assert prompt == expected


def test_generate_region_image_prompt_artistic():
    """Artistic prompts balance creativity with accuracy."""
    region_info = {
        "summary": "Summary.\n\nDetails.",
        "atlas_labels": {"Atlas": "Label"},
    }
    prompt = generate_region_image_prompt(
        [1, 2, 3], region_info, image_type="artistic"
    )
    coord = "[1.00, 2.00, 3.00]"
    atlas = "According to brain atlases, this region corresponds to: Atlas: Label. "
    expected = IMAGE_PROMPT_TEMPLATES["artistic"].format(
        coordinate=coord,
        first_paragraph="Summary.",
        atlas_context=atlas,
        x_coord="1",
        y_coord="2",
        z_coord="3",
        study_context="",
    )
    assert prompt == expected


def test_generate_region_image_prompt_unknown_type():
    """Unknown image types fall back to a generic prompt."""
    region_info = {"summary": "Just one paragraph"}
    prompt = generate_region_image_prompt([1, 2, 3], region_info, image_type="other")
    coord = "[1.00, 2.00, 3.00]"
    expected = IMAGE_PROMPT_TEMPLATES["default"].format(
        coordinate=coord,
        first_paragraph="Just one paragraph",
        atlas_context="",
        x_coord="1",
        y_coord="2",
        z_coord="3",
        study_context="",
    )
    assert prompt == expected


def test_generate_region_image_prompt_custom_template():
    """Custom templates override default image prompts."""
    region_info = {"summary": "Single paragraph"}
    template = "Custom image for {coordinate} -> {first_paragraph} || {atlas_context}"
    prompt = generate_region_image_prompt(
        [1, 2, 3], region_info, prompt_template=template
    )
    assert prompt.startswith("Custom image for [1.00, 2.00, 3.00]")


@patch("coord2region.llm.generate_region_image_prompt", return_value="PROMPT")
@patch("coord2region.llm.add_watermark", return_value=b"WM")
def test_generate_region_image_calls_ai(mock_watermark, mock_prompt):
    ai = MagicMock()
    ai.generate_image.return_value = b"IMG"
    region_info = {"summary": "text"}
    coord = [1, 2, 3]
    result = generate_region_image(ai, coord, region_info)
    mock_prompt.assert_called_once()
    ai.generate_image.assert_called_once_with(
        model="stabilityai/stable-diffusion-2", prompt="PROMPT", retries=3
    )
    mock_watermark.assert_called_once_with(b"IMG")
    assert result == b"WM"


# ---------------------------------------------------------------------------
# generate_summary tests
# ---------------------------------------------------------------------------


@patch("coord2region.llm.generate_llm_prompt", return_value="PROMPT")
def test_generate_summary_calls_ai(mock_prompt):
    generate_summary._cache.clear()
    ai = MagicMock()
    ai.generate_text.return_value = "SUMMARY"
    studies = _sample_studies()
    coord = [1, 2, 3]

    result = generate_summary(ai, studies, coord)

    mock_prompt.assert_called_once()
    ai.generate_text.assert_called_once_with(
        model="gemini-2.0-flash", prompt="PROMPT", max_tokens=1000
    )
    assert result == "SUMMARY"


@patch("coord2region.llm.generate_llm_prompt", return_value="PROMPT")
def test_generate_summary_uses_cache(mock_prompt):
    generate_summary._cache.clear()
    ai = MagicMock()
    ai.generate_text.return_value = "SUMMARY"
    studies = _sample_studies()
    coord = [1, 2, 3]

    result1 = generate_summary(ai, studies, coord)
    result2 = generate_summary(ai, studies, coord)

    ai.generate_text.assert_called_once()
    assert result1 == result2 == "SUMMARY"


@patch("coord2region.llm.generate_llm_prompt")
def test_generate_summary_includes_atlas_labels(mock_prompt):
    generate_summary._cache.clear()
    base = "Intro\nSTUDIES REPORTING ACTIVATION AT MNI COORDINATE more"
    mock_prompt.return_value = base
    ai = MagicMock()
    ai.generate_text.return_value = "SUMMARY"

    atlas_labels = {"Atlas": "Label"}
    generate_summary(
        ai, [], [1, 2, 3], atlas_labels=atlas_labels
    )

    prompt_used = ai.generate_text.call_args.kwargs["prompt"]
    assert "ATLAS LABELS FOR THIS COORDINATE" in prompt_used
    assert "Atlas: Label" in prompt_used


@patch("coord2region.llm.generate_llm_prompt", return_value="PROMPT")
def test_generate_summary_async_calls_ai(mock_prompt):
    generate_summary_async._cache.clear()
    ai = MagicMock()
    ai.generate_text_async = AsyncMock(return_value="SUMMARY")
    studies = _sample_studies()
    coord = [1, 2, 3]

    result = asyncio.run(
        generate_summary_async(ai, studies, coord)
    )

    mock_prompt.assert_called_once()
    ai.generate_text_async.assert_awaited_once_with(
        model="gemini-2.0-flash", prompt="PROMPT", max_tokens=1000
    )
    assert result == "SUMMARY"


@patch("coord2region.llm.generate_llm_prompt", return_value="PROMPT")
def test_stream_summary_calls_ai(mock_prompt):
    stream_summary._cache.clear()
    ai = MagicMock()

    def _stream(**kwargs):
        yield "A"
        yield "B"

    ai.stream_generate_text = MagicMock(
        side_effect=lambda **kwargs: _stream()
    )
    studies = _sample_studies()
    coord = [1, 2, 3]

    result = list(stream_summary(ai, studies, coord))

    mock_prompt.assert_called_once()
    ai.stream_generate_text.assert_called_once_with(
        model="gemini-2.0-flash", prompt="PROMPT", max_tokens=1000
    )
    assert result == ["A", "B"]


@patch("coord2region.llm.generate_llm_prompt", side_effect=["P1", "P2"])
def test_generate_batch_summaries_no_batching(mock_prompt):
    ai = MagicMock()
    ai.supports_batching.return_value = False
    ai.generate_text.side_effect = ["S1", "S2"]
    pairs = [([1, 2, 3], _sample_studies()), ([4, 5, 6], _sample_studies())]

    result = generate_batch_summaries(ai, pairs)

    assert result == ["S1", "S2"]
    assert ai.generate_text.call_count == 2


@patch("coord2region.llm.generate_llm_prompt", side_effect=["P1", "P2"])
def test_generate_batch_summaries_with_batching(mock_prompt):
    ai = MagicMock()
    ai.supports_batching.return_value = True
    delimiter = "\n@@@\n"
    ai.generate_text.return_value = f"S1{delimiter}S2"
    pairs = [([1, 2, 3], _sample_studies()), ([4, 5, 6], _sample_studies())]

    result = generate_batch_summaries(ai, pairs)

    assert result == ["S1", "S2"]
    ai.generate_text.assert_called_once()


@pytest.mark.unit
def test_generate_llm_prompt_handles_bad_coord():
    prompt = generate_llm_prompt(_sample_studies(), ["a", "b", "c"])
    assert "['a', 'b', 'c']" in prompt


@pytest.mark.unit
@patch("coord2region.llm.generate_region_image_prompt", return_value="PROMPT")
@patch("coord2region.llm.add_watermark")
def test_generate_region_image_no_watermark(mock_wm, mock_prompt):
    ai = MagicMock()
    ai.generate_image.return_value = b"IMG"
    res = generate_region_image(ai, [1,2,3], {"summary":""}, watermark=False)
    mock_prompt.assert_called_once()
    mock_wm.assert_not_called()
    assert res == b"IMG"


@pytest.mark.unit
def test_stream_summary_cache():
    stream_summary._cache.clear()
    ai = MagicMock()
    ai.stream_generate_text.side_effect = [iter(["A", "B"])]
    studies = _sample_studies()
    coord = [1,2,3]
    first = list(stream_summary(ai, studies, coord))
    second = list(stream_summary(ai, studies, coord))
    assert first == second == ["A", "B"]
    ai.stream_generate_text.assert_called_once()


@pytest.mark.unit
@patch(
    "coord2region.llm.generate_llm_prompt",
    side_effect=["P1", "P2", "P1", "P2"],
)
def test_generate_batch_summaries_cache(mock_prompt):
    generate_batch_summaries._cache.clear()
    ai = MagicMock()
    ai.supports_batching.return_value = True
    delimiter = "\n@@@\n"
    ai.generate_text.return_value = f"S1{delimiter}S2"
    pairs = [([1, 2, 3], _sample_studies()), ([4, 5, 6], _sample_studies())]

    first = generate_batch_summaries(ai, pairs)
    ai.generate_text.assert_called_once()

    ai.generate_text.reset_mock()
    second = generate_batch_summaries(ai, pairs)
    ai.generate_text.assert_not_called()
    assert first == second == ["S1", "S2"]


@pytest.mark.unit
@patch(
    "coord2region.llm.generate_llm_prompt",
    return_value="Intro\nSTUDIES REPORTING ACTIVATION AT MNI COORDINATE ...",
)
def test_generate_summary_async_includes_atlas_labels(mock_prompt):
    generate_summary_async._cache.clear()
    ai = MagicMock()
    ai.generate_text_async = AsyncMock(return_value="SUMMARY")
    asyncio.run(
        generate_summary_async(
            ai,
            [],
            [1, 2, 3],
            atlas_labels={"Atlas": "Label"},
        )
    )

    prompt_used = ai.generate_text_async.await_args.kwargs["prompt"]
    assert "ATLAS LABELS FOR THIS COORDINATE" in prompt_used
    assert "Atlas: Label" in prompt_used
