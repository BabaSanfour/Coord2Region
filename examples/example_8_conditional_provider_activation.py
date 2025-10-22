"""Activate providers based on available API keys
===============================================

This example shows how to conditionally use OpenAI or Anthropic depending on
which credentials are configured.
"""

import os

from coord2region.ai_model_interface import AIModelInterface


def main() -> None:
    """Run a demo that selects providers based on available API keys."""
    ai = AIModelInterface(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    if ai.supports("gpt-4o-mini"):
        print(ai.generate_text("gpt-4o-mini", "Hello from OpenAI"))
    elif ai.supports("claude-3-opus"):
        print(ai.generate_text("claude-3-opus", "Hello from Anthropic"))
    else:
        print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY to enable providers.")


if __name__ == "__main__":
    main()
