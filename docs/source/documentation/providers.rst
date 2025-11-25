AI Provider Configuration
=========================

**Coord2Region** integrates with a variety of Generative AI providers to enrich
anatomical data with semantic summaries and illustrations.



The package uses a flexible **Environment Variable** strategy: it checks for
keys in your shell environment, a local ``.env`` file, or a private YAML configuration.

Quick Start: Interactive Setup
------------------------------

The easiest way to get started is to use the included configuration wizard.
This script will prompt you for the API keys you wish to enable and save them
securely to ``config/coord2region-config.yaml`` (which is automatically git-ignored).

.. code-block:: bash

   python scripts/configure_coord2region.py



Supported Providers
-------------------

Coord2Region supports both commercial cloud providers and open-weight community models.

.. list-table::
   :widths: 15 25 40 20
   :header-rows: 1

   * - Provider
     - Environment Variable
     - Capabilities
     - Pricing
   * - **OpenAI**
     - ``OPENAI_API_KEY``
     - Flagship reasoning (o-series), text (GPT-4), and image (DALLE/GPT-Image).
     - `OpenAI Pricing <https://openai.com/api/pricing/>`_
   * - **Anthropic**
     - ``ANTHROPIC_API_KEY``
     - Claude 3.x models for high-quality reasoning and text generation.
     - `Anthropic <https://www.anthropic.com/product>`_
   * - **Google Gemini**
     - ``GEMINI_API_KEY``
     - Native integration with Gemini 1.5 Pro/Flash via ``google-genai``.
     - `Gemini API <https://ai.google.dev/>`_
   * - **Hugging Face**
     - ``HUGGINGFACE_API_KEY``
     - Gateway to thousands of open models via Inference API and Routers.
     - `HF Inference <https://huggingface.co/inference-api>`_
   * - **DeepSeek**
     - ``DEEPSEEK_API_KEY``
     - Direct access to DeepSeek reasoning models with structured output support.
     - `DeepSeek Docs <https://api-docs.deepseek.com/>`_
   * - **OpenRouter**
     - ``OPENROUTER_API_KEY``
     - Aggregator for DeepSeek, Llama, and others (often includes free tiers).
     - `OpenRouter <https://openrouter.ai/models>`_
   * - **Groq**
     - ``GROQ_API_KEY``
     - Extremely high-speed inference for Llama/Gemma/Qwen.
     - `Groq Cloud <https://console.groq.com/>`_

.. note::
   
   **Disabling Providers:** To disable a provider, simply unset its environment 
   variable or remove it from your configuration file. Coord2Region dynamically 
   adjusts available features based on active keys.

OpenAI (Text + Image)
---------------------

OpenAI is the default backend for many tutorials.

1. **Get a Key:** Sign up at `platform.openai.com <https://platform.openai.com>`_.
2. **Billing:** Ensure you have added credits; the API is not free.
3. **Configure:** Set ``OPENAI_API_KEY``.

**Supported Models**

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Alias
     - Context
     - Notes
   * - ``o4`` / ``o4-mini``
     - 200k
     - **Reasoning.** Latest flagship models with tool-use awareness.
   * - ``gpt-4o``
     - 128k
     - **Multimodal.** High capability text and vision.
   * - ``gpt-4o-mini``
     - 128k
     - **Efficient.** The default model for Coord2Region summaries.
   * - ``gpt-image-1``
     - N/A
     - **Image.** DALLE-based generation (max 1024x1024).

.. code-block:: python
   :caption: Example: Generating a summary with OpenAI

   from coord2region.llm import generate_summary
   
   # Uses o4-mini for reasoning-heavy tasks
   summary = generate_summary(ai, studies, coord, model="o4-mini")


Hugging Face (Router & Inference)
---------------------------------

The Hugging Face integration allows you to route requests to specific managed providers (like Together AI, SambaNova, or Fal AI) using a single HF Token.

**1. Setup**

Get a **Read** token from your `Hugging Face Settings <https://huggingface.co/settings/tokens>`_ and set it as ``HUGGINGFACE_API_KEY``.

**2. Router Configuration**

You must map specific models to their backend providers using the ``HUGGINGFACE_MODEL_PROVIDERS`` environment variable. The format is a comma-separated list of ``alias:provider``.

.. code-block:: bash

    HUGGINGFACE_MODEL_PROVIDERS=\
      gpt-oss-120b:sambanova,\
      deepseek-r1:together,\
      llama-3.3-70b-instruct:together,\
      stabilityai/stable-diffusion-3.5-large:fal-ai

**Recommended Mappings**

| Alias | Provider ID | Description |
| :--- | :--- | :--- |
| ``gpt-oss-120b`` | ``sambanova`` | **SambaNova.** Often offers a free developer tier. |
| ``deepseek-r1`` | ``together`` | **Together AI.** High-performance hosting for large reasoning models. |
| ``stable-diffusion-3.5`` | ``fal-ai`` | **Fal AI.** Specialized image generation backend (Paid). |

.. warning::
   **Gated Models:** For models like Llama 3.3 or Stable Diffusion 3.5, you must visit the model card on Hugging Face and accept the license terms before your API token will work.


Local & Self-Hosted
-------------------

You can point Coord2Region to your own inference server (vLLM, Ollama, TGI) by overriding the base URL. This mimics the OpenAI API structure.

**Configuration Variables**

- ``AI_BASE_URL``: Override the endpoint (e.g., ``http://localhost:8000/v1``).
- ``AI_API_KEY``: (Optional) A pass-through token if your gateway requires auth.
- ``AI_MODELS``: Register your local models so Coord2Region knows they exist.
  
  *Format:* ``alias:model_id`` pairs.

**Example: Connecting to Ollama**

If you are running Ollama locally:

.. code-block:: bash

    export AI_BASE_URL="http://localhost:11434/v1"
    export AI_API_KEY="ollama"  # dummy key often required by clients
    export AI_MODELS="local-llama:llama3.1,local-mistral:mistral"