"""AI model interface and provider abstraction."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union


# Optional imports. Each provider checks for the modules it needs and will only
# be instantiated when its requirements are available. This keeps the
# dependency surface small for users who only need a subset of providers.
try:  # pragma: no cover - simple import guard
    import openai  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    openai = None  # type: ignore

try:  # pragma: no cover
    from google import genai  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    genai = None  # type: ignore

try:  # pragma: no cover
    import anthropic  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    anthropic = None  # type: ignore

try:  # pragma: no cover
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore


PromptType = Union[str, List[Dict[str, str]]]


class ModelProvider(ABC):
    """Base class for all model providers."""

    def __init__(self, models: Dict[str, str]):
        self.models = models

    def supports(self, model: str) -> bool:
        return model in self.models

    @abstractmethod
    def generate_text(self, model: str, prompt: PromptType, max_tokens: int) -> str:
        """Generate text from the given model."""


class GeminiProvider(ModelProvider):
    """Provider for Google Gemini models."""

    def __init__(self, api_key: str):
        if genai is None:  # pragma: no cover - handled in tests
            raise ImportError("google-genai is not installed")
        models = {
            "gemini-1.0-pro": "gemini-1.0-pro",
            "gemini-1.5-pro": "gemini-1.5-pro",
            "gemini-2.0-flash": "gemini-2.0-flash",
        }
        super().__init__(models)
        self.client = genai.Client(api_key=api_key)

    def generate_text(self, model: str, prompt: PromptType, max_tokens: int) -> str:  # pragma: no cover - thin wrapper
        if isinstance(prompt, list):
            prompt = " ".join(msg["content"] for msg in prompt if msg.get("role") == "user")
        response = self.client.models.generate_content(model=model, contents=[prompt])
        return response.text


class OpenRouterProvider(ModelProvider):
    """Provider for models available via OpenRouter (e.g., DeepSeek)."""

    def __init__(self, api_key: str):
        if openai is None:  # pragma: no cover
            raise ImportError("openai is not installed")
        models = {
            "deepseek-r1": "deepseek/deepseek-r1:free",
            "deepseek-chat-v3-0324": "deepseek/deepseek-chat-v3-0324:free",
        }
        super().__init__(models)
        openai.api_base = "https://openrouter.ai/api/v1"  # type: ignore[attr-defined]
        openai.api_key = api_key  # type: ignore[attr-defined]

    def generate_text(self, model: str, prompt: PromptType, max_tokens: int) -> str:  # pragma: no cover
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        response = openai.ChatCompletion.create(  # type: ignore[attr-defined]
            model=self.models[model],
            messages=messages,
            max_tokens=max_tokens,
        )
        return response["choices"][0]["message"]["content"]


class OpenAIProvider(ModelProvider):
    """Provider for OpenAI's GPT models."""

    def __init__(self, api_key: str):
        if openai is None:  # pragma: no cover
            raise ImportError("openai is not installed")
        models = {"gpt-4": "gpt-4"}
        super().__init__(models)
        openai.api_key = api_key  # type: ignore[attr-defined]

    def generate_text(self, model: str, prompt: PromptType, max_tokens: int) -> str:  # pragma: no cover
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        response = openai.ChatCompletion.create(  # type: ignore[attr-defined]
            model=self.models[model],
            messages=messages,
            max_tokens=max_tokens,
        )
        return response["choices"][0]["message"]["content"]


class AnthropicProvider(ModelProvider):
    """Provider for Anthropic's Claude models."""

    def __init__(self, api_key: str):
        if anthropic is None:  # pragma: no cover
            raise ImportError("anthropic is not installed")
        models = {
            "claude-3-haiku": "claude-3-haiku-20240307",
            "claude-3-opus": "claude-3-opus-20240229",
        }
        super().__init__(models)
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate_text(self, model: str, prompt: PromptType, max_tokens: int) -> str:  # pragma: no cover
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        response = self.client.messages.create(
            model=self.models[model],
            max_tokens=max_tokens,
            messages=messages,
        )
        if response.content:
            return response.content[0].text
        return ""


class HuggingFaceProvider(ModelProvider):
    """Provider using the HuggingFace Inference API."""

    API_URL = "https://api-inference.huggingface.co/models/{model}"

    def __init__(self, api_key: str):
        if requests is None:  # pragma: no cover
            raise ImportError("requests is required for the HuggingFace provider")
        models = {"distilgpt2": "distilgpt2"}
        super().__init__(models)
        self.api_key = api_key

    def generate_text(self, model: str, prompt: PromptType, max_tokens: int) -> str:  # pragma: no cover
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens}}
        url = self.API_URL.format(model=self.models[model])
        resp = requests.post(url, headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        if isinstance(result, list) and result and "generated_text" in result[0]:
            return result[0]["generated_text"]
        if isinstance(result, dict) and "generated_text" in result:
            return result["generated_text"]
        return str(result)


class AIModelInterface:
    """Register and dispatch to different AI model providers."""

    _PROVIDER_CLASSES = {
        "gemini": GeminiProvider,
        "openrouter": OpenRouterProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "huggingface": HuggingFaceProvider,
    }

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        huggingface_api_key: Optional[str] = None,
        enabled_providers: Optional[List[str]] = None,
    ):
        """Initialise the interface and register available providers."""

        env_providers = os.environ.get("AI_MODEL_PROVIDERS")
        if enabled_providers is None and env_providers:
            enabled_providers = [p.strip() for p in env_providers.split(",") if p.strip()]

        self._providers: Dict[str, ModelProvider] = {}

        provider_kwargs = {
            "gemini": gemini_api_key or os.environ.get("GEMINI_API_KEY"),
            "openrouter": openrouter_api_key or os.environ.get("OPENROUTER_API_KEY"),
            "openai": openai_api_key or os.environ.get("OPENAI_API_KEY"),
            "anthropic": anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY"),
            "huggingface": huggingface_api_key
            or os.environ.get("HUGGINGFACE_API_KEY")
            or os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
        }

        for name, cls in self._PROVIDER_CLASSES.items():
            if enabled_providers is not None and name not in enabled_providers:
                continue
            api_key = provider_kwargs.get(name)
            if not api_key:
                continue
            try:
                provider = cls(api_key)
            except Exception:
                continue
            self.register_provider(provider)

    def register_provider(self, provider: ModelProvider) -> None:
        """Register a provider and its models."""

        for model in provider.models:
            self._providers[model] = provider

    def generate_text(
        self,
        model: str,
        prompt: PromptType,
        max_tokens: int = 1000,
    ) -> str:
        """Generate text using a registered model."""

        provider = self._providers.get(model)
        if provider is None:
            raise ValueError(
                f"Model '{model}' not supported. Available models: {list(self._providers.keys())}"
            )
        try:
            return provider.generate_text(model, prompt, max_tokens=max_tokens)
        except Exception as e:  # pragma: no cover - simple re-raise
            raise RuntimeError(f"Error generating response with {model}: {e}") from e

    def list_available_models(self) -> List[str]:
        """Return the list of registered model names."""

        return list(self._providers.keys())


__all__ = [
    "AIModelInterface",
    "ModelProvider",
]

