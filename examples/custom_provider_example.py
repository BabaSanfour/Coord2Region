"""Custom provider: minimal implementation and usage
=====================================================

This example shows how to write a small custom provider compatible with
the :class:`coord2region.ai_model_interface.AIModelInterface`.

The provider below simply echoes the supplied prompt. In real usage you would
wrap an external API (e.g., your organization's internal model endpoint).

What you'll see here:
- Define a :class:`~coord2region.ai_model_interface.ModelProvider` subclass
- Register it with :class:`~coord2region.ai_model_interface.AIModelInterface`
- Call sync, async and streaming generation methods
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
    # Basic generation
    print(ai.generate_text("echo-1", "Hello from EchoProvider"))

    # Streaming (yields chunks as they arrive)
    print("Streaming:", end=" ")
    for chunk in ai.stream_generate_text("echo-1", "streaming demo", max_tokens=8):
        print(chunk, end="")
    print()

    # Async version
    import asyncio

    async def _demo_async():
        res = await ai.generate_text_async("echo-1", "Hello async", max_tokens=8)
        print("Async:", res)

    asyncio.run(_demo_async())


if __name__ == "__main__":
    main()
