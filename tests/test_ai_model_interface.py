import types
import json
import os
import asyncio

import pytest
from unittest.mock import MagicMock, patch

from coord2region.ai_model_interface import (
    AIModelInterface,
    ModelProvider,
    _parse_model_mapping,
    _load_yaml_environment,
    _retry_async,
    _retry_stream,
    _retry_sync,
    load_env_file,
    huggingface_credentials_present,
    pick_first_supported_model,
    build_generation_summary,
)  # noqa: E402


@pytest.mark.unit
def test_generate_text_gemini_success():
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = types.SimpleNamespace(
        text="OK"
    )
    with patch("coord2region.ai_model_interface.genai") as mock_genai:
        mock_genai.Client.return_value = mock_client
        ai = AIModelInterface(gemini_api_key="key")
        result = ai.generate_text("gemini-2.0-flash", "hi")
    assert result == "OK"
    mock_client.models.generate_content.assert_called_once()


@pytest.mark.unit
def test_generate_text_deepseek_success():
    mock_client = MagicMock()
    mock_client.responses.create.return_value = types.SimpleNamespace(
        output=[types.SimpleNamespace(content=[types.SimpleNamespace(text="hi")])]
    )
    with patch("coord2region.ai_model_interface.OpenAI", return_value=mock_client):
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
    ai = AIModelInterface()
    with pytest.raises(ValueError):
        ai.generate_text("gemini-2.0-flash", "test")
    with pytest.raises(ValueError):
        ai.generate_text("deepseek-r1", "test")


@pytest.mark.unit
def test_generate_text_runtime_error():
    mock_client = MagicMock()
    mock_client.responses.create.side_effect = Exception("boom")
    with patch("coord2region.ai_model_interface.OpenAI", return_value=mock_client):
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
def test_supports_method():
    class DummyProvider(ModelProvider):
        def __init__(self):
            super().__init__({"m": "m"})

        def generate_text(self, model: str, prompt, max_tokens: int) -> str:
            return "ok"

    ai = AIModelInterface()
    provider = DummyProvider()
    ai.register_provider(provider)

    assert ai.supports("m") is True
    assert ai.supports("unknown") is False


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


@pytest.mark.unit
def test_register_provider_disabled():
    class Dummy(ModelProvider):
        def __init__(self):
            super().__init__({"m": "m"})

        def generate_text(self, model: str, prompt, max_tokens: int) -> str:
            return "ok"

    ai = AIModelInterface()
    ai.register_provider(Dummy(), enabled=False)
    with pytest.raises(ValueError):
        ai.generate_text("m", "hi")


@pytest.mark.unit
def test_init_skips_failed_provider(monkeypatch):
    class BrokenProvider(ModelProvider):
        def __init__(self, api_key: str):
            raise RuntimeError("boom")

        def generate_text(self, model: str, prompt, max_tokens: int) -> str:  # pragma: no cover - not used
            return ""  # pragma: no cover

    monkeypatch.setitem(AIModelInterface._PROVIDER_CLASSES, "gemini", BrokenProvider)
    ai = AIModelInterface(gemini_api_key="k")
    assert ai._providers == {}


@pytest.mark.unit
def test_parse_model_mapping_parses_pairs():
    mapping = _parse_model_mapping("alias:model, second : custom , third:")
    assert mapping["alias"] == "model"
    assert mapping["second"] == "custom"
    assert mapping["third"] == "third"


@pytest.mark.unit
def test_load_env_file_sets_values(tmp_path, monkeypatch):
    env_file = tmp_path / "test.env"
    env_file.write_text("NEW_VAR=value\nQUOTED=' spaced '\n", encoding="utf-8")
    monkeypatch.setenv("EXISTING", "keep")
    monkeypatch.setattr(
        "coord2region.ai_model_interface._load_yaml_environment", lambda path: None
    )
    load_env_file(env_file)
    assert os.environ["NEW_VAR"] == "value"
    assert os.environ["QUOTED"] == "' spaced '"
    assert os.environ["EXISTING"] == "keep"


@pytest.mark.unit
def test_huggingface_credentials_present(monkeypatch):
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
    assert huggingface_credentials_present() is False
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "abc")
    assert huggingface_credentials_present() is True


@pytest.mark.unit
def test_pick_first_supported_model():
    class Stub:
        def __init__(self):
            self.models = {"b"}

        def supports(self, model: str) -> bool:
            return model in self.models

    stub = Stub()
    assert pick_first_supported_model(stub, ["a", "b", "c"]) == "b"
    assert pick_first_supported_model(stub, ["x"]) is None


@pytest.mark.unit
def test_build_generation_summary_includes_reasoning():
    summary = build_generation_summary("model", "Hello <think>reasoning", "provider")
    data = json.loads(summary)
    assert data["model"] == "model"
    assert data["provider"] == "provider"
    assert data["has_reasoning"] is True
    assert data["tokens"] > 0


@pytest.mark.unit
def test_stream_generate_text_retries_on_failure():
    class StreamFlaky(ModelProvider):
        def __init__(self):
            super().__init__({"m": "m"})
            self.calls = 0

        def generate_text(self, model: str, prompt, max_tokens: int) -> str:  # pragma: no cover - not used
            raise NotImplementedError

        def stream_generate_text(self, model: str, prompt, max_tokens: int):
            self.calls += 1
            if self.calls < 2:
                raise RuntimeError("boom")
            yield "chunk"

    ai = AIModelInterface()
    ai.register_provider(StreamFlaky())
    chunks = list(ai.stream_generate_text("m", "prompt", retries=2))
    assert chunks == ["chunk"]


@pytest.mark.unit
def test_list_models_and_provider_name():
    class Dummy(ModelProvider):
        def __init__(self):
            super().__init__({"m": "backend"})

        def generate_text(self, model: str, prompt, max_tokens: int) -> str:
            return "ok"

    ai = AIModelInterface()
    provider = Dummy()
    ai.register_provider(provider)
    assert ai.list_available_models() == ["m"]
    assert ai.provider_name("m") == "Dummy"
    assert ai.provider_name("missing") == "UnknownProvider"


@pytest.mark.unit
def test_load_yaml_environment_applies_values(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "environment:\n  KEY_ONE: foo\n  KEY_TWO: 5\n  EMPTY: null\nextra: IGNORED\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("KEY_ONE", raising=False)
    monkeypatch.delenv("KEY_TWO", raising=False)
    _load_yaml_environment(cfg)
    assert os.environ["KEY_ONE"] == "foo"
    assert os.environ["KEY_TWO"] == "5"
    assert "EMPTY" not in os.environ


@pytest.mark.unit
def test_pick_first_supported_model_handles_exception():
    class Stub:
        def __init__(self):
            self.count = 0

        def supports(self, model: str) -> bool:
            self.count += 1
            if self.count == 1:
                raise RuntimeError("boom")
            return model == "yes"

    stub = Stub()
    assert pick_first_supported_model(stub, ["maybe", "yes"]) == "yes"


@pytest.mark.unit
def test_model_provider_async_and_stream_defaults():
    class Basic(ModelProvider):
        def __init__(self):
            super().__init__({"m": "m"})

        def generate_text(self, model: str, prompt, max_tokens: int) -> str:
            return prompt

    provider = Basic()
    assert provider.supports("m")
    assert provider.supports("x") is False
    assert asyncio.run(provider.generate_text_async("m", "text", 5)) == "text"
    assert list(provider.stream_generate_text("m", "chunk", 5)) == ["chunk"]


@pytest.mark.unit
def test_retry_async_helper():
    calls = {"count": 0}

    async def flaky():
        calls["count"] += 1
        if calls["count"] < 2:
            raise RuntimeError("fail")
        return "ok"

    result = asyncio.run(_retry_async(lambda: flaky(), retries=2))
    assert result == "ok"
    assert calls["count"] == 2


@pytest.mark.unit
def test_retry_stream_helper():
    calls = {"count": 0}

    def flaky():
        calls["count"] += 1
        if calls["count"] < 2:
            raise RuntimeError("fail")
        yield from ("a", "b")

    result = list(_retry_stream(flaky, retries=2))
    assert result == ["a", "b"]
    assert calls["count"] == 2


@pytest.mark.unit
def test_retry_sync_helper():
    calls = {"count": 0}

    def flaky():
        calls["count"] += 1
        if calls["count"] < 2:
            raise RuntimeError("fail")
        return "ok"

    result = _retry_sync(flaky, retries=2)
    assert result == "ok"
    assert calls["count"] == 2


@pytest.mark.unit
def test_load_yaml_environment_missing_file(tmp_path):
    missing = tmp_path / "nope.yaml"
    _load_yaml_environment(missing)  # Should not raise


@pytest.mark.unit
def test_load_yaml_environment_invalid_root(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("- item1\n- item2\n", encoding="utf-8")
    monkeypatch.delenv("INVALID_KEY", raising=False)
    _load_yaml_environment(cfg)
    assert "INVALID_KEY" not in os.environ


@pytest.mark.unit
def test_load_yaml_environment_bad_env_section(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("environment:\n  - value\n", encoding="utf-8")
    _load_yaml_environment(cfg)


@pytest.mark.unit
def test_interface_local_provider_config(monkeypatch):
    class StubProvider(ModelProvider):
        def __init__(self, **kwargs):
            super().__init__({"stub": "backend"})
            self.kwargs = kwargs

        def generate_text(self, model: str, prompt, max_tokens: int) -> str:
            return "ok"

    monkeypatch.setitem(
        AIModelInterface._PROVIDER_CLASSES, "local_openai", StubProvider
    )
    ai = AIModelInterface(
        local_openai_base_url="http://localhost",
        local_openai_api_key="secret",
        local_openai_models={"stub": "backend"},
    )
    provider = ai._providers["stub"]
    assert provider.kwargs["base_url"] == "http://localhost"
    assert provider.kwargs["api_key"] == "secret"


@pytest.mark.unit
def test_interface_env_provider_filter(monkeypatch):
    class StubProvider(ModelProvider):
        def __init__(self, **kwargs):
            super().__init__({"stub": "backend"})

        def generate_text(self, model: str, prompt, max_tokens: int) -> str:
            return "ok"

    monkeypatch.setenv("AI_MODEL_PROVIDERS", "local_openai")
    monkeypatch.setitem(
        AIModelInterface._PROVIDER_CLASSES, "local_openai", StubProvider
    )
    ai = AIModelInterface(local_openai_base_url="http://localhost")
    assert "stub" in ai._providers


@pytest.mark.unit
def test_openai_project_key_requires_project():
    with pytest.raises(ValueError):
        AIModelInterface(openai_api_key="sk-proj-123")


@pytest.mark.unit
def test_openai_project_key_with_project(monkeypatch):
    class StubProvider(ModelProvider):
        def __init__(self, **kwargs):
            super().__init__({"gpt-4o": "gpt-4o"})
            self.kwargs = kwargs

        def generate_text(self, model: str, prompt, max_tokens: int) -> str:
            return "ok"

    monkeypatch.setitem(AIModelInterface._PROVIDER_CLASSES, "openai", StubProvider)
    ai = AIModelInterface(openai_api_key="sk-proj-123", openai_project="proj")
    provider = ai._providers["gpt-4o"]
    assert provider.kwargs["project"] == "proj"


@pytest.mark.unit
def test_huggingface_provider_mapping(monkeypatch):
    class StubProvider(ModelProvider):
        def __init__(self, api_key: str, model_providers=None):
            super().__init__({"distilgpt2": "distilgpt2"})
            self.kwargs = {"api_key": api_key, "model_providers": model_providers}

        def generate_text(self, model: str, prompt, max_tokens: int) -> str:
            return "ok"

    monkeypatch.setitem(
        AIModelInterface._PROVIDER_CLASSES, "huggingface", StubProvider
    )
    monkeypatch.setenv("HUGGINGFACE_MODEL_PROVIDERS", "distilgpt2:openai/dummy")
    ai = AIModelInterface(huggingface_api_key="tok")
    provider = ai._providers["distilgpt2"]
    assert provider.kwargs["model_providers"] == {"distilgpt2": "openai/dummy"}


@pytest.mark.unit
def test_huggingface_helpers_and_legacy(monkeypatch):
    provider_cls = AIModelInterface._PROVIDER_CLASSES["huggingface"]
    assert provider_cls._normalize_messages("hello")[0]["content"] == "hello"
    choice = {"message": {"content": [{"text": "hi"}]}}
    assert provider_cls._extract_text(choice) == "hi"

    mock_resp = MagicMock()
    mock_resp.json.return_value = [{"generated_text": "ok"}]
    mock_resp.raise_for_status.return_value = None
    monkeypatch.setattr(
        "coord2region.ai_model_interface.requests.post", lambda *args, **kwargs: mock_resp
    )
    monkeypatch.setattr(
        "coord2region.ai_model_interface.InferenceClient",
        lambda **kwargs: MagicMock(chat=MagicMock(completions=MagicMock())),
    )
    monkeypatch.setattr(
        "coord2region.ai_model_interface.OpenAI", MagicMock(return_value=MagicMock())
    )
    stub = provider_cls(api_key="k", model_providers=None, timeout=5.0)
    result = stub._legacy_generate_text("distilgpt2", "prompt", max_tokens=5)
    assert result == "ok"
