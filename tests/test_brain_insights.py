import sys
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
