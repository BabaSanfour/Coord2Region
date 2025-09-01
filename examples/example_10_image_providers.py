"""Generate images using OpenAI or Anthropic depending on available keys
======================================================================
"""

import os

from coord2region.ai_model_interface import AIModelInterface


def main() -> None:
    ai = AIModelInterface()
    ai.register_provider(
        "openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        enabled=bool(os.getenv("OPENAI_API_KEY")),
    )
    ai.register_provider(
        "anthropic",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        enabled=bool(os.getenv("ANTHROPIC_API_KEY")),
    )

    if ai.supports("gpt-image-1"):
        img = ai.generate_image("gpt-image-1", "A colourful brain illustration")
        with open("openai_image.png", "wb") as f:
            f.write(img)
        print("Saved image from OpenAI as openai_image.png")
    elif ai.supports("claude-image"):
        img = ai.generate_image("claude-image", "A colourful brain illustration")
        with open("claude_image.png", "wb") as f:
            f.write(img)
        print("Saved image from Anthropic as claude_image.png")
    else:
        print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY to enable image providers.")


if __name__ == "__main__":
    main()
