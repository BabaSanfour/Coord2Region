from unittest.mock import MagicMock, patch

import pytest

from coord2region import ai_helpers


@pytest.mark.unit
def test_load_environment_passes_override_path():
    with patch("coord2region.ai_model_interface.load_env_file") as mock_load:
        ai_helpers.load_environment("custom.env")
    mock_load.assert_called_once_with("custom.env")


@pytest.mark.unit
@patch("coord2region.ai_model_interface.AIModelInterface")
@patch("coord2region.ai_helpers.load_environment")
def test_build_interface_warns_when_no_models(mock_load_env, mock_interface, caplog):
    instance = MagicMock()
    instance.list_available_models.return_value = []
    mock_interface.return_value = instance
    with caplog.at_level("WARNING"):
        ai_helpers.build_interface(enabled_providers=["openai"])
    mock_load_env.assert_called_once()
    mock_interface.assert_called_once_with(enabled_providers=["openai"])
    assert "No AI providers registered" in caplog.text


@pytest.mark.unit
def test_select_model_prefers_explicit_and_candidates(caplog):
    class Stub:
        def __init__(self):
            self.calls = []

        def supports(self, name):
            self.calls.append(name)
            return name == "model-b"

        def list_available_models(self):
            return ["model-b"]

    stub = Stub()
    assert ai_helpers.select_model(stub, ["model-a"], explicit="model-b") == "model-b"
    stub.calls.clear()
    assert ai_helpers.select_model(stub, ["model-b"], explicit="unknown") == "model-b"
    with caplog.at_level("WARNING"):
        stub.calls.clear()
        result = ai_helpers.select_model(stub, ["missing"], explicit="missing")
    assert result is None
    assert "Requested text model" in caplog.text


@pytest.mark.unit
def test_getenv_str_trims_quotes(monkeypatch):
    monkeypatch.delenv("TEST_ENV", raising=False)
    assert ai_helpers.getenv_str("TEST_ENV") is None
    monkeypatch.setenv("TEST_ENV", "  ' value '  ")
    assert ai_helpers.getenv_str("TEST_ENV") == "value"
