"""AI model interface and provider abstraction with retry support.

All provider calls are wrapped with an exponential backoff retry to cope
with transient failures. The retry behaviour can be configured via
``retries`` parameters on the public methods.

The :class:`AIModelInterface` constructor accepts optional API keys for
multiple providers. Notably, the ``openai_api_key`` and
``anthropic_api_key`` parameters (or the ``OPENAI_API_KEY`` and
``ANTHROPIC_API_KEY`` environment variables) enable OpenAI and
Anthropic models respectively.

Notes
-----
This module requires the ``openai`` (version >=1.0), ``google-genai``,
``anthropic``, ``requests``, ``transformers`` and ``diffusers`` packages.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

from openai import AsyncOpenAI, OpenAI

try:
    from google import genai
except ImportError:  # pragma: no cover - optional dependency
    genai = None
try:
    import anthropic
except ImportError:  # pragma: no cover - optional dependency
    anthropic = None
import requests
from huggingface_hub import InferenceClient
import yaml

try:
    from transformers import pipeline as hf_local_pipeline
except ImportError as exc:  # pragma: no cover - optional dependency
    if "sklearn" in repr(exc).lower():
        raise ImportError(
            "transformers import failed because scikit-learn is missing. "
            "Install scikit-learn with `pip install scikit-learn`."
        ) from exc
    hf_local_pipeline = None
try:
    from diffusers import StableDiffusionPipeline
except ImportError:  # pragma: no cover - optional dependency
    StableDiffusionPipeline = None


PromptType = Union[str, List[Dict[str, str]]]


def _parse_model_mapping(env_value: Optional[str]) -> Dict[str, str]:
    """Parse ``alias:model_id`` pairs from an environment variable."""
    mapping: Dict[str, str] = {}
    if not env_value:
        return mapping
    for raw_item in env_value.split(","):
        alias, _, model_id = raw_item.partition(":")
        alias = alias.strip()
        model_id = model_id.strip()
        if not alias:
            continue
        mapping[alias] = model_id or alias
    return mapping


def _load_yaml_environment(path: Union[str, Path]) -> None:
    """Load environment variables from a YAML configuration file if available."""
    config_path = Path(path)
    if not config_path.exists():
        return
    try:
        parsed = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # pragma: no cover - YAML parse errors are rare
        raise RuntimeError(
            f"Failed to parse configuration file {config_path}: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        return
    env_values = parsed.get("environment", parsed)
    if not isinstance(env_values, dict):
        return
    for key, value in env_values.items():
        if value is None:
            continue
        key_str = str(key).strip()
        if not key_str:
            continue
        # Force override any existing environment variables
        os.environ[key_str] = str(value).strip()


def load_env_file(path: Union[str, Path] = Path(".env")) -> None:
    """Load configuration-managed credentials before falling back to ``.env``.

    Parameters
    ----------
    path : str or Path, optional
        Path to the environment file to load. Defaults to ``.env``.
    """
    _load_yaml_environment(Path("config") / "coord2region-config.yaml")

    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def huggingface_credentials_present() -> bool:
    """Check whether Hugging Face credentials are available.

    Returns
    -------
    bool
        ``True`` if either Hugging Face API key environment variable is set.
    """
    return bool(
        os.environ.get("HUGGINGFACE_API_KEY")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    )


def pick_first_supported_model(
    ai: "AIModelInterface", candidates: Iterable[str]
) -> Optional[str]:
    """Return the first supported model from a list of candidates.

    Parameters
    ----------
    ai : AIModelInterface
        Interface used to query model availability.
    candidates : iterable of str
        Candidate model names evaluated in order of preference.

    Returns
    -------
    str or None
        First supported model name or ``None`` if no match is found.
    """
    for model in candidates:
        try:
            if ai.supports(model):
                return model
        except Exception:
            continue
    return None


def build_generation_summary(model: str, response: str, provider: str) -> str:
    """Return a JSON summary describing a text generation output.

    Parameters
    ----------
    model : str
        Model name used for the generation.
    response : str
        Raw text produced by the model.
    provider : str
        Provider label for the selected model.

    Returns
    -------
    str
        JSON-formatted metadata describing the generation.
    """
    summary = {
        "provider": provider,
        "model": model,
        "has_reasoning": "<think>" in response.lower(),
        "tokens": len(response.split()),
    }
    return json.dumps(summary, indent=2)


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

    Parameters
    ----------
    models : dict
        Mapping of friendly model names to provider-specific identifiers.
    """

    #: Whether the provider natively supports batching multiple prompts in a
    #: single API call. Subclasses can override this to ``True`` when their
    #: backend exposes such functionality.
    supports_batching: bool = False

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
    """Provider for Google Gemini models.

    Parameters
    ----------
    api_key : str
        API key used to authenticate with Google GenAI.
    """

    def __init__(self, api_key: str):  # pragma: no cover - network client setup
        if genai is None:
            raise ImportError(
                "Google Gemini support requires the google-genai package. "
                "Install it via `pip install google-genai`."
            )
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

    def __init__(self, api_key: str):  # pragma: no cover - network client setup
        models = {
            "deepseek-r1": "deepseek/deepseek-r1:free",
            "deepseek-chat-v3-0324": "deepseek/deepseek-chat-v3-0324:free",
        }
        super().__init__(models)
        self.client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        self.async_client = AsyncOpenAI(
            api_key=api_key, base_url="https://openrouter.ai/api/v1"
        )

    def generate_text(
        self, model: str, prompt: PromptType, max_tokens: int
    ) -> str:  # pragma: no cover
        if isinstance(prompt, str):
            prompt_input: PromptType = [{"role": "user", "content": prompt}]
        else:
            prompt_input = prompt
        response = self.client.responses.create(
            model=self.models[model],
            input=prompt_input,
            max_output_tokens=max_tokens,
        )
        return response.output[0].content[0].text

    async def generate_text_async(
        self, model: str, prompt: PromptType, max_tokens: int
    ) -> str:  # pragma: no cover - thin wrapper
        if isinstance(prompt, str):
            prompt_input: PromptType = [{"role": "user", "content": prompt}]
        else:
            prompt_input = prompt
        response = await self.async_client.responses.create(
            model=self.models[model],
            input=prompt_input,
            max_output_tokens=max_tokens,
        )
        return response.output[0].content[0].text

    def stream_generate_text(
        self, model: str, prompt: PromptType, max_tokens: int
    ) -> Iterator[str]:  # pragma: no cover - thin wrapper
        if isinstance(prompt, str):
            prompt_input: PromptType = [{"role": "user", "content": prompt}]
        else:
            prompt_input = prompt
        with self.client.responses.stream(
            model=self.models[model],
            input=prompt_input,
            max_output_tokens=max_tokens,
        ) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    yield event.delta


class GroqProvider(ModelProvider):
    """Provider for Groq-hosted OpenAI-compatible models."""

    def __init__(self, api_key: str):  # pragma: no cover - network client setup
        models = {
            "groq-llama-3.1-70b": "llama-3.1-70b-versatile",
            "groq-llama-3.1-8b": "llama-3.1-8b-instant",
        }
        super().__init__(models)
        client_kwargs = {
            "api_key": api_key,
            "base_url": "https://api.groq.com/openai/v1",
        }
        self.client = OpenAI(**client_kwargs)
        self.async_client = AsyncOpenAI(**client_kwargs)

    def generate_text(
        self, model: str, prompt: PromptType, max_tokens: int
    ) -> str:  # pragma: no cover - thin wrapper
        if isinstance(prompt, str):
            prompt_input: PromptType = [{"role": "user", "content": prompt}]
        else:
            prompt_input = prompt
        response = self.client.responses.create(
            model=self.models[model],
            input=prompt_input,
            max_output_tokens=max_tokens,
        )
        return response.output[0].content[0].text

    async def generate_text_async(
        self, model: str, prompt: PromptType, max_tokens: int
    ) -> str:  # pragma: no cover - thin wrapper
        if isinstance(prompt, str):
            prompt_input: PromptType = [{"role": "user", "content": prompt}]
        else:
            prompt_input = prompt
        response = await self.async_client.responses.create(
            model=self.models[model],
            input=prompt_input,
            max_output_tokens=max_tokens,
        )
        return response.output[0].content[0].text

    def stream_generate_text(
        self, model: str, prompt: PromptType, max_tokens: int
    ) -> Iterator[str]:  # pragma: no cover - thin wrapper
        if isinstance(prompt, str):
            prompt_input: PromptType = [{"role": "user", "content": prompt}]
        else:
            prompt_input = prompt
        with self.client.responses.stream(
            model=self.models[model],
            input=prompt_input,
            max_output_tokens=max_tokens,
        ) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    yield event.delta


class DeepSeekProvider(ModelProvider):
    """Provider for DeepSeek's native API."""

    def __init__(self, api_key: str):  # pragma: no cover - network client setup
        models = {
            "deepseek-reasoner": "deepseek-reasoner",
            "deepseek-chat": "deepseek-chat",
        }
        super().__init__(models)
        client_kwargs = {"api_key": api_key, "base_url": "https://api.deepseek.com/v1"}
        self.client = OpenAI(**client_kwargs)
        self.async_client = AsyncOpenAI(**client_kwargs)

    def generate_text(
        self, model: str, prompt: PromptType, max_tokens: int
    ) -> str:  # pragma: no cover - thin wrapper
        if isinstance(prompt, str):
            prompt_input: PromptType = [{"role": "user", "content": prompt}]
        else:
            prompt_input = prompt
        response = self.client.responses.create(
            model=self.models[model],
            input=prompt_input,
            max_output_tokens=max_tokens,
        )
        return response.output[0].content[0].text

    async def generate_text_async(
        self, model: str, prompt: PromptType, max_tokens: int
    ) -> str:  # pragma: no cover - thin wrapper
        if isinstance(prompt, str):
            prompt_input: PromptType = [{"role": "user", "content": prompt}]
        else:
            prompt_input = prompt
        response = await self.async_client.responses.create(
            model=self.models[model],
            input=prompt_input,
            max_output_tokens=max_tokens,
        )
        return response.output[0].content[0].text

    def stream_generate_text(
        self, model: str, prompt: PromptType, max_tokens: int
    ) -> Iterator[str]:  # pragma: no cover - thin wrapper
        if isinstance(prompt, str):
            prompt_input: PromptType = [{"role": "user", "content": prompt}]
        else:
            prompt_input = prompt
        with self.client.responses.stream(
            model=self.models[model],
            input=prompt_input,
            max_output_tokens=max_tokens,
        ) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    yield event.delta


class TogetherProvider(ModelProvider):
    """Provider for Together AI models (DeepSeek, Llama, Mixtral, etc.)."""

    def __init__(self, api_key: str):  # pragma: no cover - network client setup
        models = {
            "together-deepseek-r1": "deepseek-ai/DeepSeek-R1",
            "together-llama-3.1-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        }
        super().__init__(models)
        client_kwargs = {"api_key": api_key, "base_url": "https://api.together.ai/v1"}
        self.client = OpenAI(**client_kwargs)
        self.async_client = AsyncOpenAI(**client_kwargs)

    def generate_text(
        self, model: str, prompt: PromptType, max_tokens: int
    ) -> str:  # pragma: no cover - thin wrapper
        if isinstance(prompt, str):
            prompt_input: PromptType = [{"role": "user", "content": prompt}]
        else:
            prompt_input = prompt
        response = self.client.responses.create(
            model=self.models[model],
            input=prompt_input,
            max_output_tokens=max_tokens,
        )
        return response.output[0].content[0].text

    async def generate_text_async(
        self, model: str, prompt: PromptType, max_tokens: int
    ) -> str:  # pragma: no cover - thin wrapper
        if isinstance(prompt, str):
            prompt_input: PromptType = [{"role": "user", "content": prompt}]
        else:
            prompt_input = prompt
        response = await self.async_client.responses.create(
            model=self.models[model],
            input=prompt_input,
            max_output_tokens=max_tokens,
        )
        return response.output[0].content[0].text

    def stream_generate_text(
        self, model: str, prompt: PromptType, max_tokens: int
    ) -> Iterator[str]:  # pragma: no cover - thin wrapper
        if isinstance(prompt, str):
            prompt_input: PromptType = [{"role": "user", "content": prompt}]
        else:
            prompt_input = prompt
        with self.client.responses.stream(
            model=self.models[model],
            input=prompt_input,
            max_output_tokens=max_tokens,
        ) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    yield event.delta


class LocalOpenAIProvider(ModelProvider):
    """Provider for self-hosted OpenAI-compatible gateways (vLLM, TGI, Ollama)."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: Optional[str] = None,
        models: Optional[Dict[str, str]] = None,
        default_model: str = "local-reasoning",
    ):  # pragma: no cover - optional local deployment wrapper
        if models is None or not models:
            models = {default_model: default_model}
        super().__init__(models)
        api_key_value = api_key or "EMPTY"
        client_kwargs = {
            "api_key": api_key_value,
            "base_url": base_url.rstrip("/"),
        }
        self.client = OpenAI(**client_kwargs)
        self.async_client = AsyncOpenAI(**client_kwargs)

    def generate_text(
        self, model: str, prompt: PromptType, max_tokens: int
    ) -> str:  # pragma: no cover - thin wrapper
        if isinstance(prompt, str):
            prompt_input: PromptType = [{"role": "user", "content": prompt}]
        else:
            prompt_input = prompt
        response = self.client.responses.create(
            model=self.models[model],
            input=prompt_input,
            max_output_tokens=max_tokens,
        )
        return response.output[0].content[0].text

    async def generate_text_async(
        self, model: str, prompt: PromptType, max_tokens: int
    ) -> str:  # pragma: no cover - thin wrapper
        if isinstance(prompt, str):
            prompt_input: PromptType = [{"role": "user", "content": prompt}]
        else:
            prompt_input = prompt
        response = await self.async_client.responses.create(
            model=self.models[model],
            input=prompt_input,
            max_output_tokens=max_tokens,
        )
        return response.output[0].content[0].text

    def stream_generate_text(
        self, model: str, prompt: PromptType, max_tokens: int
    ) -> Iterator[str]:  # pragma: no cover - thin wrapper
        if isinstance(prompt, str):
            prompt_input: PromptType = [{"role": "user", "content": prompt}]
        else:
            prompt_input = prompt
        with self.client.responses.stream(
            model=self.models[model],
            input=prompt_input,
            max_output_tokens=max_tokens,
        ) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    yield event.delta


class OpenAIProvider(ModelProvider):
    """Provider for OpenAI's GPT models."""

    def __init__(
        self, api_key: str, project: Optional[str] = None
    ):  # pragma: no cover - network client setup
        models = {
            "gpt-4o": "gpt-4o",
            "gpt-4o-mini": "gpt-4o-mini",
            "gpt-4": "gpt-4-turbo-2024-04-09",
            "gpt-image-1": "gpt-4o",  # Uses gpt-4o with image generation tool
            "dall-e-3": "dall-e-3",
            "dall-e-2": "dall-e-2",
        }
        super().__init__(models)
        self._image_models = {"gpt-image-1", "dall-e-3", "dall-e-2"}
        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if project:
            client_kwargs["project"] = project
        self.client = OpenAI(**client_kwargs)
        self.async_client = AsyncOpenAI(**client_kwargs)

    def generate_text(
        self, model: str, prompt: PromptType, max_tokens: int
    ) -> str:  # pragma: no cover
        if isinstance(prompt, str):
            prompt_input: PromptType = [{"role": "user", "content": prompt}]
        else:
            prompt_input = prompt
        response = self.client.responses.create(
            model=self.models[model],
            input=prompt_input,
            max_output_tokens=max_tokens,
        )
        return response.output[0].content[0].text

    async def generate_text_async(
        self, model: str, prompt: PromptType, max_tokens: int
    ) -> str:  # pragma: no cover - thin wrapper
        if isinstance(prompt, str):
            prompt_input: PromptType = [{"role": "user", "content": prompt}]
        else:
            prompt_input = prompt
        response = await self.async_client.responses.create(
            model=self.models[model],
            input=prompt_input,
            max_output_tokens=max_tokens,
        )
        return response.output[0].content[0].text

    def stream_generate_text(
        self, model: str, prompt: PromptType, max_tokens: int
    ) -> Iterator[str]:  # pragma: no cover - thin wrapper
        if isinstance(prompt, str):
            prompt_input: PromptType = [{"role": "user", "content": prompt}]
        else:
            prompt_input = prompt
        with self.client.responses.stream(
            model=self.models[model],
            input=prompt_input,
            max_output_tokens=max_tokens,
        ) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    yield event.delta

    def generate_image(
        self, model: str, prompt: str, **kwargs: Any
    ) -> bytes:  # pragma: no cover - network image wrapper
        if model not in self._image_models:
            raise ValueError(f"Model '{model}' is not an image model")

        # gpt-image-1 uses the Responses API with image generation tool
        if model == "gpt-image-1":
            # Build the tool parameters from kwargs
            tool_params = {"type": "image_generation"}
            if "quality" in kwargs:
                tool_params["quality"] = kwargs["quality"]
            if "size" in kwargs:
                tool_params["size"] = kwargs["size"]
            if "background" in kwargs:
                tool_params["background"] = kwargs["background"]

            try:
                response = self.client.responses.create(
                    model=self.models[model],  # This should be gpt-4o or similar
                    input=prompt,
                    tools=[tool_params],
                )

                # Extract the image data from the response
                image_data = None
                for output in response.output:
                    if output.type == "image_generation_call":
                        image_data = output.result
                        break

                if image_data:
                    # The result is already base64 encoded
                    return base64.b64decode(image_data)
                else:
                    raise ValueError("No image generated in response")

            except AttributeError:
                # Fallback for older OpenAI SDK versions that don't have responses API
                raise NotImplementedError(
                    "gpt-image-1 requires the Responses API which is not available"
                    " in your OpenAI SDK version. "
                    "Please update to the latest OpenAI SDK or use"
                    " dall-e-2/dall-e-3 instead."
                )

        # DALL-E models use the Images API
        elif self.models[model] in ["dall-e-3", "dall-e-2"]:
            # Remove unsupported kwargs for images.generate
            image_kwargs = {}
            if "size" in kwargs:
                image_kwargs["size"] = kwargs["size"]
            if "quality" in kwargs and self.models[model] == "dall-e-3":
                image_kwargs["quality"] = kwargs["quality"]
            if "n" in kwargs:
                image_kwargs["n"] = kwargs["n"]

            if self.models[model] == "dall-e-3":
                # DALL-E 3 supports b64_json response format
                resp = self.client.images.generate(
                    model=self.models[model],
                    prompt=prompt,
                    response_format="b64_json",
                    **image_kwargs,
                )
                data = resp.data[0].b64_json
                return base64.b64decode(data)
            else:
                # DALL-E 2 uses URL format
                resp = self.client.images.generate(
                    model=self.models[model], prompt=prompt, **image_kwargs
                )
                # Get the URL and download the image
                image_url = resp.data[0].url
                response = requests.get(image_url)
                response.raise_for_status()
                return response.content


class AnthropicProvider(ModelProvider):
    """Provider for Anthropic's Claude models.

    Parameters
    ----------
    api_key : str
        API key used to authenticate with Anthropic.
    """

    def __init__(self, api_key: str):  # pragma: no cover - network client setup
        if anthropic is None:
            raise ImportError(
                "Anthropic support requires the anthropic package. "
                "Install it via `pip install anthropic`."
            )
        models = {
            "claude-3-haiku": "claude-3-haiku-20240307",
            "claude-3-opus": "claude-3-opus-20240229",
            "claude-image": "claude-3-opus-20240229",
        }
        super().__init__(models)
        self._image_models = {"claude-image"}
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

    def generate_image(
        self, model: str, prompt: str, **kwargs: Any
    ) -> bytes:  # pragma: no cover - network image wrapper
        if model not in self._image_models:
            raise ValueError(f"Model '{model}' is not an image model")
        resp = self.client.images.generate(model=self.models[model], prompt=prompt)
        data = resp.data[0].b64_json  # type: ignore[attr-defined]
        return base64.b64decode(data)


class HuggingFaceProvider(ModelProvider):
    """Provider using the HuggingFace Inference Hub."""

    API_URL = "https://api-inference.huggingface.co/models/{model}"

    def __init__(
        self,
        api_key: str,
        *,
        model_providers: Optional[Dict[str, str]] = None,
        timeout: float = 60.0,
    ):  # pragma: no cover - network client setup
        models = {
            "distilgpt2": "distilgpt2",
            "deepseek-r1-distill-qwen-14b": (
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
            ),
            "deepseek-r1": "deepseek-ai/DeepSeek-R1",
            "gpt-oss-120b": "openai/gpt-oss-120b",
            "llama-3.3-70b-instruct": ("meta-llama/Llama-3.3-70B-Instruct"),
            "stabilityai/stable-diffusion-2": ("stabilityai/stable-diffusion-2"),
            "stabilityai/stable-diffusion-3.5-large": (
                "stabilityai/stable-diffusion-3.5-large"
            ),
            "stabilityai/stable-diffusion-xl-base-1.0": (
                "stabilityai/stable-diffusion-xl-base-1.0"
            ),
        }

        super().__init__(models)
        self.api_key = api_key
        self.model_providers = model_providers or {}
        self._timeout = timeout
        self._provider_clients: Dict[Optional[str], InferenceClient] = {
            None: InferenceClient(token=api_key, timeout=timeout)
        }
        self._router_client = OpenAI(
            api_key=api_key, base_url="https://router.huggingface.co/v1"
        )
        self._router_provider_names = {
            "together",
            "sambanova",
            "novita",
            "fireworks",
            "replicate",
            "groq",
            "cerebras",
            "featherless",
            "hyperbolic",
        }

    @staticmethod
    def _normalize_messages(prompt: PromptType) -> List[Dict[str, str]]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        return prompt

    @staticmethod
    def _extract_text(choice: Any) -> str:
        message = getattr(choice, "message", None)
        if message is None and isinstance(choice, dict):
            message = choice.get("message")
        if message is None:
            text = getattr(choice, "text", None)
            if text is None and isinstance(choice, dict):
                text = choice.get("text")
            return text or ""
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(item.get("text", ""))
                else:
                    parts.append(str(item))
            return "".join(parts)
        if content is not None:
            return str(content)
        return ""

    def _get_client(self, provider: Optional[str]) -> InferenceClient:
        if provider not in self._provider_clients:
            self._provider_clients[provider] = InferenceClient(
                token=self.api_key, timeout=self._timeout, provider=provider
            )
        return self._provider_clients[provider]

    def _legacy_generate_text(
        self, model: str, prompt: PromptType, max_tokens: int
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if isinstance(prompt, list):
            user_chunks = [p["content"] for p in prompt if p.get("role") == "user"]
            system_chunks = [p["content"] for p in prompt if p.get("role") == "system"]
            combined_parts: List[str] = []
            if system_chunks:
                combined_parts.append("\n".join(system_chunks))
            if user_chunks:
                combined_parts.append("\n".join(user_chunks))
            prompt_input: Union[str, PromptType] = "\n\n".join(combined_parts)
        else:
            prompt_input = prompt
        data = {
            "inputs": prompt_input,
            "parameters": {
                "max_new_tokens": max_tokens,
                "return_full_text": False,
            },
            "options": {"wait_for_model": True},
        }
        url = self.API_URL.format(model=self.models[model])
        resp = requests.post(url, headers=headers, json=data, timeout=self._timeout)
        resp.raise_for_status()
        result = resp.json()
        if isinstance(result, list) and result:
            generated = result[0]
            if isinstance(generated, dict):
                return generated.get("generated_text", str(generated))
            return str(generated)
        if isinstance(result, dict):
            text = result.get("generated_text")
            if text:
                return text
        return str(result)

    def generate_text(
        self, model: str, prompt: PromptType, max_tokens: int
    ) -> str:  # pragma: no cover - network wrapper
        messages = self._normalize_messages(prompt)
        provider = self.model_providers.get(model)
        if provider and provider.lower() in self._router_provider_names:
            try:
                completion = self._router_client.chat.completions.create(
                    model=f"{self.models[model]}:{provider}",
                    messages=messages,
                    max_tokens=max_tokens,
                )
                choice = completion.choices[0] if completion.choices else None
                if choice is not None:
                    message = getattr(choice, "message", None)
                    if message is None and isinstance(choice, dict):
                        message = choice.get("message")
                    if message is not None:
                        content = getattr(message, "content", None)
                        if content is None and isinstance(message, dict):
                            content = message.get("content")
                        if isinstance(content, list):
                            return "".join(str(item) for item in content)
                        if content is not None:
                            return str(content)
                return ""
            except Exception:
                # Fall back to inference client below
                pass
        try:
            client = self._get_client(provider)
            completion = client.chat.completions.create(
                model=self.models[model],
                messages=messages,
                max_tokens=max_tokens,
            )
            if getattr(completion, "choices", None):
                return self._extract_text(completion.choices[0])
            return ""
        except Exception:
            return self._legacy_generate_text(model, prompt, max_tokens)

    def generate_image(
        self, model: str, prompt: str
    ) -> bytes:  # pragma: no cover - network wrapper
        """Generate an image using the HuggingFace Inference API."""
        provider = self.model_providers.get(model)
        try:
            client = self._get_client(provider)
            image = client.text_to_image(prompt, model=self.models[model])
            if hasattr(image, "save"):
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                return buf.getvalue()
            if isinstance(image, bytes):
                return image
        except Exception:
            # Fall back to classic binary endpoint.
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "image/png",
            }
            data = {"inputs": prompt}
            url = self.API_URL.format(model=self.models[model])
            resp = requests.post(url, headers=headers, json=data, timeout=self._timeout)
            resp.raise_for_status()
            return resp.content


class HuggingFaceLocalProvider(ModelProvider):
    """Provider that runs HuggingFace models locally.

    Uses ``transformers`` for text and ``diffusers`` for images. Both text and
    image generation can be configured independently by specifying
    ``text_model`` and/or ``image_model`` when registering the provider. The
    heavy model weights are loaded on first use.

    Parameters
    ----------
    text_model : str, optional
        Local text generation model name.
    image_model : str, optional
        Local image generation model name.
    """

    def __init__(
        self,
        *,
        text_model: Optional[str] = None,
        image_model: Optional[str] = None,
    ):  # pragma: no cover - heavy local dependency
        models: Dict[str, str] = {}
        if text_model:
            models[text_model] = text_model
        if image_model:
            models[image_model] = image_model
        if not models:
            raise ValueError("At least one of text_model or image_model must be set")
        if hf_local_pipeline is None:
            raise ImportError(
                "Local HuggingFace inference requires the transformers package. "
                "Install it via `pip install transformers`."
            )
        if image_model and StableDiffusionPipeline is None:
            raise ImportError(
                "Local HuggingFace image generation requires the diffusers package. "
                "Install it via `pip install diffusers`."
            )
        super().__init__(models)
        self._text_model = text_model
        self._image_model = image_model
        self._text_generator = None
        self._image_pipeline = None

    def _ensure_text_pipeline(self) -> None:
        if self._text_generator is None:
            self._text_generator = hf_local_pipeline(
                "text-generation", model=self._text_model
            )

    def _ensure_image_pipeline(self) -> None:
        if self._image_pipeline is None:
            self._image_pipeline = StableDiffusionPipeline.from_pretrained(
                self._image_model
            )

    def generate_text(
        self, model: str, prompt: PromptType, max_tokens: int
    ) -> str:  # pragma: no cover - optional heavy dependency
        if model != self._text_model or not self._text_model:
            raise ValueError(f"Model '{model}' is not configured for text generation")
        if isinstance(prompt, list):
            prompt = " ".join(
                msg["content"] for msg in prompt if msg.get("role") == "user"
            )
        self._ensure_text_pipeline()
        result = self._text_generator(prompt, max_new_tokens=max_tokens)
        return result[0]["generated_text"]

    def generate_image(self, model: str, prompt: str) -> bytes:  # pragma: no cover
        if model != self._image_model or not self._image_model:
            raise ValueError(f"Model '{model}' is not configured for image generation")
        self._ensure_image_pipeline()
        image = self._image_pipeline(prompt).images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()


class AIModelInterface:
    """Register and dispatch to different AI model providers."""

    _PROVIDER_CLASSES = {
        "gemini": GeminiProvider,
        "openrouter": OpenRouterProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "huggingface": HuggingFaceProvider,
        "huggingface_local": HuggingFaceLocalProvider,
        "groq": GroqProvider,
        "deepseek": DeepSeekProvider,
        "together": TogetherProvider,
        "local_openai": LocalOpenAIProvider,
    }

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_project: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        huggingface_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        deepseek_api_key: Optional[str] = None,
        together_api_key: Optional[str] = None,
        local_openai_base_url: Optional[str] = None,
        local_openai_api_key: Optional[str] = None,
        local_openai_models: Optional[Dict[str, str]] = None,
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
        groq_api_key : str, optional
            API key for Groq Cloud. Defaults to ``GROQ_API_KEY`` environment
            variable.
        deepseek_api_key : str, optional
            API key for DeepSeek's native API. Defaults to ``DEEPSEEK_API_KEY``.
        together_api_key : str, optional
            API key for Together AI. Defaults to ``TOGETHER_API_KEY``.
        local_openai_base_url : str, optional
            Base URL for a self-hosted OpenAI-compatible server (vLLM, TGI,
            Ollama). Defaults to ``AI_BASE_URL`` environment variable.
        local_openai_api_key : str, optional
            API key for the self-hosted OpenAI-compatible server. Defaults to
            ``AI_API_KEY`` environment variable.
        local_openai_models : dict, optional
            Mapping of public model aliases to backend IDs for the local
            provider. Defaults to parsing ``AI_MODELS`` environment variable
            (``alias:model`` comma-separated pairs).
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

        provider_configs: Dict[str, Dict[str, Any]] = {}

        gemini_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if gemini_key:
            provider_configs["gemini"] = {"api_key": gemini_key}

        openrouter_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        if openrouter_key:
            provider_configs["openrouter"] = {"api_key": openrouter_key}

        openai_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        openai_project_value = openai_project or os.environ.get("OPENAI_PROJECT")
        if openai_key:
            if openai_key.startswith("sk-proj-") and not openai_project_value:
                raise ValueError(
                    "OPENAI_API_KEY appears to be a project-scoped key but no "
                    "OPENAI_PROJECT was provided."
                    "Set the project ID via the openai_project argument or the "
                    "OPENAI_PROJECT environment variable."
                )
            openai_cfg: Dict[str, Any] = {"api_key": openai_key}
            if openai_project_value:
                openai_cfg["project"] = openai_project_value
            provider_configs["openai"] = openai_cfg

        anthropic_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_key:
            provider_configs["anthropic"] = {"api_key": anthropic_key}

        huggingface_key = (
            huggingface_api_key
            or os.environ.get("HUGGINGFACE_API_KEY")
            or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        )
        if huggingface_key:
            hf_provider_map = _parse_model_mapping(
                os.environ.get("HUGGINGFACE_MODEL_PROVIDERS")
            )
            hf_config: Dict[str, Any] = {"api_key": huggingface_key}
            if hf_provider_map:
                hf_config["model_providers"] = hf_provider_map
            provider_configs["huggingface"] = hf_config

        groq_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        if groq_key:
            provider_configs["groq"] = {"api_key": groq_key}

        deepseek_key = deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY")
        if deepseek_key:
            provider_configs["deepseek"] = {"api_key": deepseek_key}

        together_key = together_api_key or os.environ.get("TOGETHER_API_KEY")
        if together_key:
            provider_configs["together"] = {"api_key": together_key}

        local_base_url = local_openai_base_url or os.environ.get("AI_BASE_URL")
        if local_base_url:
            local_config: Dict[str, Any] = {"base_url": local_base_url}
            local_key = local_openai_api_key or os.environ.get("AI_API_KEY")
            if local_key:
                local_config["api_key"] = local_key
            local_models = local_openai_models or _parse_model_mapping(
                os.environ.get("AI_MODELS")
            )
            if local_models:
                local_config["models"] = local_models
            provider_configs["local_openai"] = local_config

        for name in self._PROVIDER_CLASSES:
            if enabled_providers is not None and name not in enabled_providers:
                continue
            config = provider_configs.get(name)
            if not config:
                continue
            try:
                self.register_provider(name, **config)
            except Exception:
                continue

    def register_provider(
        self,
        provider: Union[ModelProvider, str],
        *,
        enabled: bool = True,
        **config: Any,
    ) -> None:
        """Register a provider and its models.

        Parameters
        ----------
        provider : ModelProvider | str
            Either an instantiated provider or the name of a provider defined in
            :attr:`_PROVIDER_CLASSES`.
        enabled : bool, optional
            When ``False`` the provider is skipped.
        **config : dict, optional
            Configuration forwarded to the provider constructor when ``provider``
            is given as a string.
        """
        if not enabled:
            return
        if isinstance(provider, str):
            cls = self._PROVIDER_CLASSES.get(provider)
            if cls is None:
                raise ValueError(f"Unknown provider '{provider}'")
            provider_obj = cls(**config)
        else:
            provider_obj = provider
        for model in provider_obj.models:
            self._providers[model] = provider_obj

    def supports(self, model: str) -> bool:
        """Return whether ``model`` is registered with any provider."""
        return model in self._providers

    def supports_batching(self, model: str) -> bool:
        """Return whether the provider for ``model`` supports batching."""
        provider = self._providers.get(model)
        if provider is None:
            available = list(self._providers.keys())
            raise ValueError(
                f"Model '{model}' not supported. Available models: {available}"
            )
        return getattr(provider, "supports_batching", False)

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

    def provider_name(self, model: str) -> str:
        """Return the provider class name registered for ``model``."""
        provider = self._providers.get(model)
        return type(provider).__name__ if provider is not None else "UnknownProvider"


__all__ = [
    "AIModelInterface",
    "ModelProvider",
    "PromptType",
    "load_env_file",
    "huggingface_credentials_present",
    "pick_first_supported_model",
    "build_generation_summary",
]
