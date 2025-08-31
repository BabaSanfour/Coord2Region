import sys
import types

import pytest
from unittest.mock import MagicMock, patch

# Stub external dependencies
openai_stub = types.SimpleNamespace(
    ChatCompletion=types.SimpleNamespace(create=MagicMock()),
    api_key=None,
    api_base=None,
)
sys.modules.setdefault("openai", openai_stub)

google_module = types.ModuleType("google")
google_module.genai = types.SimpleNamespace(Client=MagicMock())
sys.modules.setdefault("google", google_module)
sys.modules.setdefault("google.genai", google_module.genai)

from coord2region.ai_model_interface import AIModelInterface  # noqa: E402
from coord2region.ai_model_interface import ModelProvider  # noqa: E402


@pytest.mark.unit
def test_generate_text_gemini_success():
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = types.SimpleNamespace(
        text="OK"
    )
    with patch("google.genai.Client", return_value=mock_client):
        ai = AIModelInterface(gemini_api_key="key")
        result = ai.generate_text("gemini-2.0-flash", "hi")
    assert result == "OK"
    mock_client.models.generate_content.assert_called_once()


@pytest.mark.unit
def test_generate_text_deepseek_success():
    with patch(
        "openai.ChatCompletion.create",
        return_value={"choices": [{"message": {"content": "hi"}}]},
    ):
        ai = AIModelInterface(openrouter_api_key="key")
        result = ai.generate_text("deepseek-r1", "hello")
    assert result == "hi"


@pytest.mark.unit
def test_generate_text_invalid_model():
    ai = AIModelInterface()
    with pytest.raises(ValueError):
        ai.generate_text("unknown", "test")


@pytest.mark.unit
def test_generate_text_missing_keys():
    openai_stub.api_key = None
    ai = AIModelInterface()
    with pytest.raises(ValueError):
        ai.generate_text("gemini-2.0-flash", "test")
    with pytest.raises(ValueError):
        ai.generate_text("deepseek-r1", "test")


@pytest.mark.unit
def test_generate_text_runtime_error():
    with patch("openai.ChatCompletion.create", side_effect=Exception("boom")):
        ai = AIModelInterface(openrouter_api_key="key")
        with pytest.raises(RuntimeError):
            ai.generate_text("deepseek-r1", "oops")


@pytest.mark.unit
def test_generate_text_retries_transient_failure():
    class FlakyProvider(ModelProvider):
        def __init__(self):
            super().__init__({"m": "m"})
            self.calls = 0

        def generate_text(self, model: str, prompt, max_tokens: int) -> str:
            self.calls += 1
            if self.calls < 2:
                raise RuntimeError("temp")
            return "ok"

    ai = AIModelInterface()
    provider = FlakyProvider()
    ai.register_provider(provider)

    result = ai.generate_text("m", "hi")
    assert result == "ok"
    assert provider.calls == 2


@pytest.mark.unit
@patch("coord2region.ai_model_interface.requests.post")
def test_huggingface_generate_text(mock_post):
    mock_resp = MagicMock()
    mock_resp.json.return_value = [{"generated_text": "hi"}]
    mock_resp.raise_for_status.return_value = None
    mock_post.return_value = mock_resp
    ai = AIModelInterface(huggingface_api_key="key")
    result = ai.generate_text("distilgpt2", "hello", max_tokens=5)
    assert result == "hi"
    mock_post.assert_called_once()


@pytest.mark.unit
@patch("coord2region.ai_model_interface.requests.post")
def test_huggingface_generate_image(mock_post):
    mock_resp = MagicMock()
    mock_resp.content = b"IMG"
    mock_resp.raise_for_status.return_value = None
    mock_post.return_value = mock_resp
    ai = AIModelInterface(huggingface_api_key="key")
    result = ai.generate_image("stabilityai/stable-diffusion-2", "cat")
    assert result == b"IMG"
    mock_post.assert_called_once()
