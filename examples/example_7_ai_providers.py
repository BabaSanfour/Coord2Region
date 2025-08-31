"""Demonstrate selecting among different LLM providers."""

from coord2region.ai_model_interface import AIModelInterface


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

    # Generate text using a specific model
    response = ai.generate_text(model="gpt-4", prompt="Hello from Coord2Region!")
    print(response)


if __name__ == "__main__":
    main()

