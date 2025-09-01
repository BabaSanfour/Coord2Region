"""Demonstrate provider selection, retries, and caching.

This example now relies on the updated ``openai`` client (``openai>=1``)
under the hood. Ensure the environment has a valid OpenAI API key and the
newer library installed before running.
"""

from coord2region.ai_model_interface import AIModelInterface
from coord2region.llm import generate_summary


def main() -> None:
    # Supply API keys through environment variables or directly here. This
    # example shows explicit constructor arguments for clarity. Replace the
    # placeholders with real keys when running locally.
    ai = AIModelInterface(
        openai_api_key="sk-...",
        gemini_api_key="...",
        enabled_providers=["openai", "gemini"],
    )

    print("Available models:", ai.list_available_models())

    # Generate text using a specific model. The interface will retry transient
    # failures with exponential backoff.
    response = ai.generate_text(model="gpt-4", prompt="Hello from Coord2Region!")
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

