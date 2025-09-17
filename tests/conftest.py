"""Test configuration and lightweight dependency stubs."""

from __future__ import annotations

import sys
import types


def _ensure_google_genai_stub() -> None:
    if "google.genai" in sys.modules:
        return
    google_mod = sys.modules.get("google")
    if google_mod is None:
        google_mod = types.ModuleType("google")
        google_mod.__path__ = []  # mark as package
        sys.modules["google"] = google_mod

    class _Models:
        def generate_content(self, *args, **kwargs):
            if kwargs.get("stream"):
                return []
            return types.SimpleNamespace(text="")

        async def generate_content_async(self, *args, **kwargs):
            return types.SimpleNamespace(text="")

    class _Client:
        def __init__(self, *args, **kwargs):
            self.models = _Models()

    genai_mod = types.ModuleType("google.genai")
    genai_mod.Client = _Client
    google_mod.genai = genai_mod
    sys.modules["google.genai"] = genai_mod


def _ensure_openai_stub() -> None:
    if "openai" in sys.modules:
        return

    class _SyncResponses:
        def create(self, *args, **kwargs):
            return types.SimpleNamespace(
                output=[
                    types.SimpleNamespace(
                        content=[types.SimpleNamespace(text="")]
                    )
                ]
            )

        def stream(self, *args, **kwargs):
            return []

    class _Images:
        def generate(self, *args, **kwargs):
            return types.SimpleNamespace(
                data=[types.SimpleNamespace(b64_json="")]
            )

    class _OpenAI:
        def __init__(self, *args, **kwargs):
            self.responses = _SyncResponses()
            self.images = _Images()

    class _AsyncResponses:
        async def create(self, *args, **kwargs):
            return types.SimpleNamespace(
                output=[
                    types.SimpleNamespace(
                        content=[types.SimpleNamespace(text="")]
                    )
                ]
            )

    class _AsyncOpenAI(_OpenAI):
        def __init__(self, *args, **kwargs):
            self.responses = _AsyncResponses()
            self.images = _Images()

    openai_mod = types.ModuleType("openai")
    openai_mod.OpenAI = _OpenAI
    openai_mod.AsyncOpenAI = _AsyncOpenAI
    sys.modules["openai"] = openai_mod


def _ensure_anthropic_stub() -> None:
    if "anthropic" in sys.modules:
        return

    class _Responses:
        def create(self, *args, **kwargs):
            return types.SimpleNamespace(
                output=[
                    types.SimpleNamespace(
                        content=[types.SimpleNamespace(text="")]
                    )
                ]
            )

        def stream(self, *args, **kwargs):
            return []

    class _Anthropic:
        def __init__(self, *args, **kwargs):
            self.responses = _Responses()

    anthropic_mod = types.ModuleType("anthropic")
    anthropic_mod.Anthropic = _Anthropic
    sys.modules["anthropic"] = anthropic_mod


def _ensure_transformers_stub() -> None:
    if "transformers" in sys.modules:
        return

    def pipeline(*args, **kwargs):  # noqa: D401 - simple stub
        class _Pipe:
            def __call__(self, *call_args, **call_kwargs):
                return [
                    {"generated_text": ""}
                ]

        return _Pipe()

    transformers_mod = types.ModuleType("transformers")
    transformers_mod.pipeline = pipeline
    sys.modules["transformers"] = transformers_mod


def _ensure_diffusers_stub() -> None:
    if "diffusers" in sys.modules:
        return

    class _Result:
        def __init__(self):
            self.images = [types.SimpleNamespace(save=lambda *a, **k: None)]

    class _StableDiffusionPipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):  # noqa: D401 - simple stub
            return cls()

        def __call__(self, *args, **kwargs):  # noqa: D401 - simple stub
            return _Result()

    diffusers_mod = types.ModuleType("diffusers")
    diffusers_mod.StableDiffusionPipeline = _StableDiffusionPipeline
    sys.modules["diffusers"] = diffusers_mod


_ensure_google_genai_stub()
_ensure_openai_stub()
_ensure_anthropic_stub()
_ensure_transformers_stub()
_ensure_diffusers_stub()
