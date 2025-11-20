"""Helpers for selecting AI models and loading environment variables.

This module consolidates the common glue used by examples so they can remain
minimal. It provides:

- ``load_environment``: read ``.env`` (if present) into ``os.environ``.
- ``build_interface``: construct an ``AIModelInterface`` with available providers.
- ``select_model``: choose a supported model from a candidate list with an optional
        override.
- ``getenv_str``: fetch and trim a string environment variable.

Examples import these helpers instead of duplicating logic.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence, TYPE_CHECKING

# Avoid importing heavy/optional dependencies at module import time.
# Import within functions to keep examples importable without provider SDKs.
if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .ai_model_interface import AIModelInterface


# Candidate model lists used by examples. Order conveys preference.
TEXT_MODEL_CANDIDATES: Sequence[str] = (
    "o4",
    "o4-mini",
    "o3-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o-mini",
    "deepseek-r1",
    "deepseek-reasoner",
    "deepseek-chat-v3-0324",
    "deepseek-chat",
    "gemini-2.0-flash",
    "claude-3-opus",
    "claude-3-haiku",
    "groq-llama-3.1-70b",
    "groq-llama-3.1-8b",
    "together-deepseek-r1",
    "together-llama-3.1-70b",
    "llama-3.3-70b-instruct",
    "gpt-oss-120b",
    "distilgpt2",
)

IMAGE_MODEL_CANDIDATES: Sequence[str] = (
    "claude-image",
    "gpt-image-1",
    "dall-e-3",
    "dall-e-2",
    "stabilityai/stable-diffusion-3.5-large",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/stable-diffusion-2",
)


def load_environment(env_path: Optional[str] = None) -> None:
    """Load environment variables from configuration files.

    Parameters
    ----------
    env_path : str, optional
        Override path to the ``.env`` file. Defaults to ``".env"`` if not provided.
    """
    from .ai_model_interface import load_env_file

    load_env_file(env_path or ".env")


def build_interface(
    *, enabled_providers: Optional[Sequence[str]] = None
) -> "AIModelInterface":
    """Initialise :class:`AIModelInterface` with available providers.

    Parameters
    ----------
    enabled_providers : sequence of str, optional
        Explicit list of provider names to register. If omitted, all detected
        providers are enabled.

    Returns
    -------
    AIModelInterface
        Interface capable of dispatching requests to the configured providers.
    """
    load_environment()
    providers = list(enabled_providers) if enabled_providers is not None else None
    from .ai_model_interface import AIModelInterface

    ai = AIModelInterface(enabled_providers=providers)
    if not ai.list_available_models():
        logging.warning(
            "No AI providers registered. Ensure API keys are present in "
            "the environment."
        )
    return ai


def select_model(
    ai: "AIModelInterface",
    candidates: Sequence[str],
    *,
    explicit: Optional[str] = None,
    kind: str = "text",
) -> Optional[str]:
    """Return the first supported model, honouring an explicit override.

    Parameters
    ----------
    ai : AIModelInterface
        Interface used to query available models.
    candidates : sequence of str
        Preferred model aliases evaluated in order.
    explicit : str, optional
        Explicit model request taking precedence if supported.
    kind : str, optional
        Human-friendly label for the capability being selected (e.g., ``"text"``).

    Returns
    -------
    str or None
        Supported model name or ``None`` when no candidate is available.
    """
    if explicit:
        request = explicit.strip()
        if request:
            if ai.supports(request):
                logging.info("Using requested %s model: %s", kind, request)
                return request
            logging.warning(
                "Requested %s model '%s' is not available. Falling back to candidates.",
                kind,
                request,
            )
    from .ai_model_interface import pick_first_supported_model

    model = pick_first_supported_model(ai, candidates)
    if model:
        logging.info("Selected %s model: %s", kind, model)
    else:
        available = ", ".join(ai.list_available_models()) or "none"
        logging.warning("No supported %s models. Available models: %s", kind, available)
    return model


def getenv_str(name: str) -> Optional[str]:
    """Return the trimmed value of an environment variable.

    Parameters
    ----------
    name : str
        Environment variable to read and sanitize.

    Returns
    -------
    str or None
        Trimmed string value with surrounding quotes removed, or ``None`` if
        not set or empty.
    """
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1].strip()
    return value or None


__all__ = [
    "TEXT_MODEL_CANDIDATES",
    "IMAGE_MODEL_CANDIDATES",
    "load_environment",
    "build_interface",
    "select_model",
    "getenv_str",
]
