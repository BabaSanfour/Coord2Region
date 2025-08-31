"""AI model interface and provider abstraction with retry support.

All provider calls are wrapped with an exponential backoff retry to cope
with transient failures. The retry behaviour can be configured via
``retries`` parameters on the public methods.

The :class:`AIModelInterface` constructor accepts optional API keys for
multiple providers. Notably, the ``openai_api_key`` and
``anthropic_api_key`` parameters (or the ``OPENAI_API_KEY`` and
``ANTHROPIC_API_KEY`` environment variables) enable OpenAI and
Anthropic models respectively.
"""

from __future__ import annotations

import asyncio
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Union


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


def _retry_sync(func, retries: int = 3, base_delay: float = 0.1) -> Any:
    """Retry ``func`` with exponential backoff."""
    delay = base_delay
    for attempt in range(retries):
        try:
            return func()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= 2


async def _retry_async(func, retries: int = 3, base_delay: float = 0.1) -> Any:
    """Asynchronously retry ``func`` with exponential backoff."""
    delay = base_delay
    for attempt in range(retries):
        try:
            return await func()
        except Exception:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(delay)
            delay *= 2


def _retry_stream(func, retries: int = 3, base_delay: float = 0.1) -> Iterator[str]:
    """Retry a streaming function yielding from successive attempts."""

    def generator() -> Iterator[str]:
        delay = base_delay
        for attempt in range(retries):
            try:
                yield from func()
                return
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2

    return generator()


class ModelProvider(ABC):
    """Base class for all model providers.

    See the ``README`` section *Adding a Custom LLM Provider* for
    guidance on implementing subclasses.
    """

    def __init__(self, models: Dict[str, str]):
        self.models = models

    def supports(self, model: str) -> bool:
        """Return ``True`` if the provider exposes the requested model."""
        return model in self.models

    @abstractmethod
    def generate_text(self, model: str, prompt: PromptType, max_tokens: int) -> str:
        """Generate text from the given model."""

    async def generate_text_async(
        self, model: str, prompt: PromptType, max_tokens: int
    ) -> str:
        """Asynchronously generate text.

        Providers that expose native async APIs should override this method.
        The default implementation simply delegates to :meth:`generate_text`
        using ``asyncio.to_thread`` to avoid blocking the event loop.
        """
        return await asyncio.to_thread(self.generate_text, model, prompt, max_tokens)

    def stream_generate_text(
        self, model: str, prompt: PromptType, max_tokens: int
    ) -> Iterator[str]:
        """Yield generated text chunks.

        Providers that support server-side streaming should override this
        method. The base implementation yields the full response in a single
        chunk.
        """
        yield self.generate_text(model, prompt, max_tokens)


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

    def generate_text(
        self, model: str, prompt: PromptType, max_tokens: int
    ) -> str:  # pragma: no cover - thin wrapper
        if isinstance(prompt, list):
            prompt = " ".join(
                msg["content"] for msg in prompt if msg.get("role") == "user"
            )
        response = self.client.models.generate_content(model=model, contents=[prompt])
        return response.text

    async def generate_text_async(
        self, model: str, prompt: PromptType, max_tokens: int
    ) -> str:  # pragma: no cover - thin wrapper
        if hasattr(self.client.models, "generate_content_async"):
            if isinstance(prompt, list):
                prompt = " ".join(
                    msg["content"] for msg in prompt if msg.get("role") == "user"
                )
            response = await self.client.models.generate_content_async(
                model=model, contents=[prompt]
            )
            return response.text
        return await super().generate_text_async(model, prompt, max_tokens)

    def stream_generate_text(
        self, model: str, prompt: PromptType, max_tokens: int
    ) -> Iterator[str]:  # pragma: no cover - thin wrapper
        if isinstance(prompt, list):
            prompt = " ".join(
                msg["content"] for msg in prompt if msg.get("role") == "user"
            )
        stream = self.client.models.generate_content(
            model=model, contents=[prompt], stream=True
        )
        for chunk in stream:
            text = getattr(chunk, "text", None)
            if text:
                yield text


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

    def generate_text(
        self, model: str, prompt: PromptType, max_tokens: int
    ) -> str:  # pragma: no cover
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

    def generate_text(
        self, model: str, prompt: PromptType, max_tokens: int
    ) -> str:  # pragma: no cover
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

    async def generate_text_async(
        self, model: str, prompt: PromptType, max_tokens: int
    ) -> str:  # pragma: no cover - thin wrapper
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        if hasattr(openai.ChatCompletion, "acreate"):
            chat_comp = openai.ChatCompletion  # type: ignore[attr-defined]
            response = await chat_comp.acreate(
                model=self.models[model],
                messages=messages,
                max_tokens=max_tokens,
            )
            return response["choices"][0]["message"]["content"]
        return await super().generate_text_async(model, prompt, max_tokens)

    def stream_generate_text(
        self, model: str, prompt: PromptType, max_tokens: int
    ) -> Iterator[str]:  # pragma: no cover - thin wrapper
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        stream = openai.ChatCompletion.create(  # type: ignore[attr-defined]
            model=self.models[model],
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk["choices"][0]["delta"].get("content")
            if delta:
                yield delta


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

    def generate_text(
        self, model: str, prompt: PromptType, max_tokens: int
    ) -> str:  # pragma: no cover
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
        models = {
            "distilgpt2": "distilgpt2",
            "stabilityai/stable-diffusion-2": "stabilityai/stable-diffusion-2",
        }
        super().__init__(models)
        self.api_key = api_key

    def generate_text(
        self, model: str, prompt: PromptType, max_tokens: int
    ) -> str:  # pragma: no cover
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

    def generate_image(self, model: str, prompt: str) -> bytes:
        """Generate an image using the HuggingFace Inference API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "image/png",
        }
        data = {"inputs": prompt}
        url = self.API_URL.format(model=self.models[model])
        resp = requests.post(url, headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        return resp.content


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
        """Initialise the interface and register available providers.

        The interface accepts optional API keys for different large language
        model providers. The ``openai_api_key`` and ``anthropic_api_key``
        parameters, or their respective ``OPENAI_API_KEY`` and
        ``ANTHROPIC_API_KEY`` environment variables, enable OpenAI and
        Anthropic support.

        Parameters
        ----------
        gemini_api_key : str, optional
            API key for Google Gemini.
        openrouter_api_key : str, optional
            API key for OpenRouter.
        openai_api_key : str, optional
            API key for OpenAI. Defaults to ``OPENAI_API_KEY`` environment
            variable if not provided.
        anthropic_api_key : str, optional
            API key for Anthropic. Defaults to ``ANTHROPIC_API_KEY`` environment
            variable if not provided.
        huggingface_api_key : str, optional
            API key for HuggingFace Inference API. Defaults to
            ``HUGGINGFACE_API_KEY`` or ``HUGGINGFACEHUB_API_TOKEN`` environment
            variables.
        enabled_providers : list[str], optional
            Restrict registration to this subset of providers. By default, all
            providers with available API keys are enabled.
        """
        env_providers = os.environ.get("AI_MODEL_PROVIDERS")
        if enabled_providers is None and env_providers:
            enabled_providers = [
                p.strip() for p in env_providers.split(",") if p.strip()
            ]

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
        """Register a provider and its models.

        The ``README`` section *Adding a Custom LLM Provider* shows how to
        create a provider and register it with this interface.
        """
        for model in provider.models:
            self._providers[model] = provider

    def generate_text(
        self,
        model: str,
        prompt: PromptType,
        max_tokens: int = 1000,
        retries: int = 3,
    ) -> str:
        """Generate text using a registered model with retry.

        Parameters
        ----------
        model, prompt, max_tokens : see :meth:`ModelProvider.generate_text`
        retries : int
            Number of attempts before raising the final error.
        """
        provider = self._providers.get(model)
        if provider is None:
            available = list(self._providers.keys())
            raise ValueError(
                f"Model '{model}' not supported. Available models: {available}"
            )
        try:
            return _retry_sync(
                lambda: provider.generate_text(model, prompt, max_tokens=max_tokens),
                retries=retries,
            )
        except Exception as e:  # pragma: no cover - simple re-raise
            raise RuntimeError(f"Error generating response with {model}: {e}") from e

    async def generate_text_async(
        self,
        model: str,
        prompt: PromptType,
        max_tokens: int = 1000,
        retries: int = 3,
    ) -> str:
        """Asynchronously generate text using a registered model with retry.

        Parameters
        ----------
        model, prompt, max_tokens : see :meth:`ModelProvider.generate_text`
        retries : int
            Number of attempts before raising the final error.
        """
        provider = self._providers.get(model)
        if provider is None:
            available = list(self._providers.keys())
            raise ValueError(
                f"Model '{model}' not supported. Available models: {available}"
            )
        try:
            return await _retry_async(
                lambda: provider.generate_text_async(
                    model, prompt, max_tokens=max_tokens
                ),
                retries=retries,
            )
        except Exception as e:  # pragma: no cover - simple re-raise
            raise RuntimeError(f"Error generating response with {model}: {e}") from e

    def stream_generate_text(
        self,
        model: str,
        prompt: PromptType,
        max_tokens: int = 1000,
        retries: int = 3,
    ) -> Iterator[str]:
        """Stream generated text chunks from a registered model with retry.

        Parameters
        ----------
        model, prompt, max_tokens : see
            :meth:`ModelProvider.stream_generate_text`
        retries : int
            Number of attempts before raising the final error.
        """
        provider = self._providers.get(model)
        if provider is None:
            available = list(self._providers.keys())
            raise ValueError(
                f"Model '{model}' not supported. Available models: {available}"
            )
        try:
            return _retry_stream(
                lambda: provider.stream_generate_text(
                    model, prompt, max_tokens=max_tokens
                ),
                retries=retries,
            )
        except Exception as e:  # pragma: no cover - simple re-raise
            raise RuntimeError(f"Error generating response with {model}: {e}") from e

    def generate_image(
        self,
        model: str,
        prompt: str,
        retries: int = 3,
        **kwargs: Any,
    ) -> bytes:
        """Generate an image using a registered model with retry."""
        provider = self._providers.get(model)
        if provider is None or not hasattr(provider, "generate_image"):
            available = [
                m for m, p in self._providers.items() if hasattr(p, "generate_image")
            ]
            raise ValueError(
                f"Model '{model}' not supported for image generation. "
                f"Available image models: {available}"
            )
        try:
            return _retry_sync(
                lambda: getattr(provider, "generate_image")(model, prompt, **kwargs),
                retries=retries,
            )
        except Exception as e:  # pragma: no cover - simple re-raise
            raise RuntimeError(f"Error generating image with {model}: {e}") from e

    def list_available_models(self) -> List[str]:
        """Return the list of registered model names."""
        return list(self._providers.keys())


__all__ = [
    "AIModelInterface",
    "ModelProvider",
]
