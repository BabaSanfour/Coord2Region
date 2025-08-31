import sys
import types
from unittest.mock import MagicMock, patch

# Stub optional dependencies for AIModelInterface initialization
sys.modules.setdefault("openai", types.SimpleNamespace())
google_module = types.ModuleType("google")
google_module.genai = types.SimpleNamespace(Client=lambda api_key: None)
sys.modules.setdefault("google", google_module)
sys.modules.setdefault("google.genai", google_module.genai)

from coord2region.llm_service import generate_summary


@patch("coord2region.llm_service.generate_llm_prompt", return_value="PROMPT")
def test_generate_summary_calls_ai(mock_prompt):
    ai = MagicMock()
    ai.generate_text.return_value = "SUMMARY"
    studies = [{"id": "1", "title": "T", "abstract": "A"}]
    coord = [1, 2, 3]

    result = generate_summary(ai, studies, coord)

    mock_prompt.assert_called_once()
    ai.generate_text.assert_called_once_with(model="gemini-2.0-flash", prompt="PROMPT", max_tokens=1000)
    assert result == "SUMMARY"


@patch("coord2region.llm_service.generate_llm_prompt")
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
