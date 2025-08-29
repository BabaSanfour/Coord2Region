import sys
import os
import types
import pytest
from unittest.mock import MagicMock, patch

# Provide stub modules for optional dependencies used during import
sys.modules.setdefault("openai", types.SimpleNamespace())
google_module = types.ModuleType("google")
google_module.genai = types.SimpleNamespace(Client=lambda api_key: None)
sys.modules.setdefault("google", google_module)
sys.modules.setdefault("google.genai", google_module.genai)

from coord2region.brain_insights import BrainInsights


@pytest.mark.unit
def test_get_region_studies_forwards_radius():
    brain = BrainInsights(use_cached_dataset=False, use_atlases=False)
    # Provide a dummy dataset so get_region_studies uses the fast path
    brain.dataset = MagicMock()

    coord = [-30, -22, 50]

    with patch(
        "coord2region.brain_insights.get_studies_for_coordinate", return_value=[]
    ) as mock_get:
        brain.get_region_studies(coord, radius=7)
        mock_get.assert_called_once()
        assert mock_get.call_args.kwargs["radius"] == 7


@pytest.mark.unit
@patch("coord2region.brain_insights.prepare_datasets")
@patch("coord2region.brain_insights.get_studies_for_coordinate", return_value=[{"id": "1"}])
def test_get_region_studies_loads_dataset(mock_get, mock_prepare):
    mock_dataset = MagicMock()
    mock_prepare.return_value = mock_dataset
    brain = BrainInsights(use_cached_dataset=False, use_atlases=False)

    coord = [0, 0, 0]
    result = brain.get_region_studies(coord)

    mock_prepare.assert_called_once()
    mock_get.assert_called_once_with({"Combined": mock_dataset}, coord, radius=0, email=None)
    assert result == [{"id": "1"}]


@pytest.mark.unit
def test_get_region_summary_caches(tmp_path):
    coord = [1, 2, 3]
    with patch("coord2region.brain_insights.AIModelInterface") as mock_ai_cls:
        mock_ai = MagicMock()
        mock_ai.generate_text.return_value = "SUMMARY"
        mock_ai_cls.return_value = mock_ai

        brain = BrainInsights(
            data_dir=str(tmp_path),
            gemini_api_key="key",
            use_cached_dataset=False,
            use_atlases=False,
        )

        brain.get_region_studies = MagicMock(return_value=[{"id": "1", "title": "T", "abstract": "A"}])
        brain.get_enriched_prompt = MagicMock(return_value="PROMPT")

        first = brain.get_region_summary(coord, include_atlas_labels=False)
        assert first["summary"] == "SUMMARY"
        cache_file = os.path.join(
            brain.cache_dir,
            f"summary_{coord[0]}_{coord[1]}_{coord[2]}_summary_gemini-2.0-flash.json",
        )
        assert os.path.exists(cache_file)

        # Second call should hit cache and skip AI/generation
        mock_ai.generate_text.reset_mock()
        brain.get_region_studies.reset_mock()
        brain.get_enriched_prompt.reset_mock()

        second = brain.get_region_summary(coord, include_atlas_labels=False)
        mock_ai.generate_text.assert_not_called()
        brain.get_region_studies.assert_not_called()
        brain.get_enriched_prompt.assert_not_called()
        assert second["summary"] == "SUMMARY"
