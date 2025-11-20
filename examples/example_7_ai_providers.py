"""Demonstrate provider selection, retries, and caching
=====================================================

This example uses the OpenAI Responses API via the ``openai`` client
(``openai>=1``). Ensure the environment has a valid OpenAI API key and the
newer library installed before running.
"""

import os

from openai import APIError, AuthenticationError

from coord2region.ai_model_interface import AIModelInterface
from coord2region.llm import generate_summary


def main() -> None:
    """Run the provider demo if credentials are available."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("OPENAI_API_KEY not set; skipping example")
        return

    ai = AIModelInterface(
        openai_api_key=key,
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        enabled_providers=["openai", "gemini"],
    )

    # Authentication failures are caught so Sphinx-Gallery marks the example
    # as skipped instead of failing.
    try:
        print("Available models:", ai.list_available_models())

        # Generate text using a specific model. The interface will retry
        # transient failures with exponential backoff.
        response = ai.generate_text(
            model="gpt-4o-mini", prompt="Hello from Coord2Region!"
        )
    except AuthenticationError:
        print("OpenAI authentication failed; skipping example")
        return
    except APIError as err:
        if err.status_code == 401:
            print("OpenAI authentication failed (HTTP 401); skipping example")
            return
        raise
    print(response)

    # Generate a summary with caching. A second call with the same arguments
    # will return instantly from the in-memory cache.
    studies = [{"id": "1", "title": "A", "abstract": "B"}]
    coord = [0, 0, 0]
    summary = generate_summary(ai, studies, coord, cache_size=2)
    summary_again = generate_summary(ai, studies, coord, cache_size=2)  # cache hit
    assert summary == summary_again


if __name__ == "__main__":
    main()

