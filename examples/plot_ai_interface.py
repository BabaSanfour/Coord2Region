"""
AI interface
============

Demonstrates the :class:`coord2region.ai_model_interface.AIModelInterface` by
registering a trivial provider that simply echoes whatever prompt it receives.
"""

# %%
# Import the interface and base provider class
from coord2region.ai_model_interface import AIModelInterface, ModelProvider


# %%
# Define a minimal provider that echoes prompts
class EchoProvider(ModelProvider):
    def __init__(self):
        super().__init__({"echo": "echo"})

    def generate_text(self, model, prompt, max_tokens):
        if isinstance(prompt, list):
            prompt = " ".join(p["content"] for p in prompt)
        return f"Echo: {prompt}"


# %%
# Register the provider and generate a response
aio = AIModelInterface()
aio.register_provider(EchoProvider())

print(aio.generate_text("echo", "Hello brain"))
