import sys
import types

import pytest
from unittest.mock import MagicMock, patch
import openai
import asyncio

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
    openai.api_key = None
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


@pytest.mark.unit
def test_supports_batching_flag():
    class DummyProvider(ModelProvider):
        supports_batching = True

        def __init__(self):
            super().__init__({"m": "m"})

        def generate_text(self, model: str, prompt, max_tokens: int) -> str:
            return "ok"

    ai = AIModelInterface()
    provider = DummyProvider()
    ai.register_provider(provider)

    assert ai.supports_batching("m") is True
    provider.supports_batching = False
    assert ai.supports_batching("m") is False


@pytest.mark.unit
def test_register_provider_invalid_name():
    ai = AIModelInterface()
    with pytest.raises(ValueError):
        ai.register_provider("unknown", api_key="k")


@pytest.mark.unit
def test_supports_batching_unknown_model():
    ai = AIModelInterface()
    with pytest.raises(ValueError):
        ai.supports_batching("unknown")


@pytest.mark.unit
def test_generate_text_async_retries():
    class AsyncFlaky(ModelProvider):
        def __init__(self):
            super().__init__({"m": "m"})
            self.calls = 0
        def generate_text(self, model, prompt, max_tokens):
            raise NotImplementedError
        async def generate_text_async(self, model: str, prompt, max_tokens: int) -> str:
            self.calls += 1
            if self.calls < 2:
                raise RuntimeError("boom")
            return "ok"
    ai = AIModelInterface()
    ai.register_provider(AsyncFlaky())
    result = asyncio.run(ai.generate_text_async("m", "hi", retries=2))
    assert result == "ok"


@pytest.mark.unit
def test_generate_image_invalid_model():
    ai = AIModelInterface()
    with pytest.raises(ValueError):
        ai.generate_image("none", "prompt")
