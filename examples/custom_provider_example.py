"""Minimal example demonstrating a custom LLM provider.

The provider simply echoes the supplied prompt and is registered
with :class:`coord2region.ai_model_interface.AIModelInterface`.
"""

from coord2region.ai_model_interface import AIModelInterface, ModelProvider


class EchoProvider(ModelProvider):
    """Provider that returns the prompt verbatim."""

    def __init__(self) -> None:
        super().__init__({"echo-1": "echo-1"})

    def generate_text(self, model: str, prompt, max_tokens: int) -> str:
        if isinstance(prompt, str):
            return prompt
        return " ".join(m["content"] for m in prompt)


def main() -> None:
    ai = AIModelInterface()
    ai.register_provider(EchoProvider())
    print(ai.generate_text("echo-1", "Hello from EchoProvider"))


if __name__ == "__main__":
    main()
