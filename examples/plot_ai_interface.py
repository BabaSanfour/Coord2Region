"""
AI interface
============

This example demonstrates the :class:`coord2region.ai_model_interface.AIModelInterface`
and the :class:`coord2region.ai_model_interface.ModelProvider` abstraction.

We implement a tiny provider that simply echoes prompts back to the caller to
show how to:

- Register a provider and its model IDs
- Generate text, stream text, and use the async API
- List available models from all registered providers

In real usage you would register built-in providers by name (e.g.,
``AIModelInterface().register_provider("openai", api_key=...)``) and then call
``generate_text`` with the model alias you care about.
"""

# %%
# Import the interface and base provider class
from coord2region.ai_model_interface import AIModelInterface, ModelProvider
import asyncio


# %%
# Define a minimal provider that echoes prompts
class EchoProvider(ModelProvider):
    """A trivial provider that returns the prompt.

    This mirrors the interface of real providers (OpenAI, Gemini, etc.) but
    keeps the logic simple so the example runs quickly in the docs build.
    """

    def __init__(self):
        # Map public model name -> internal identifier
        super().__init__({"echo": "echo"})

    def generate_text(self, model, prompt, max_tokens):
        # Providers may accept either a raw string or a list of {role, content}
        if isinstance(prompt, list):
            prompt = " ".join(p["content"] for p in prompt)
        return f"Echo: {prompt}"


# %%
# Register the provider and generate a response
aio = AIModelInterface()
aio.register_provider(EchoProvider())

# List available models
print("Available models:", aio.list_available_models())

# Basic text generation
print(aio.generate_text("echo", "Hello brain"))

# Stream generation (yields chunks)
print("Streaming:", end=" ")
for chunk in aio.stream_generate_text("echo", "stream me", max_tokens=10):
    print(chunk, end="")
print()

# Async generation
async def _demo_async():
    result = await aio.generate_text_async("echo", "Hello async", max_tokens=10)
    print("Async:", result)

asyncio.run(_demo_async())
