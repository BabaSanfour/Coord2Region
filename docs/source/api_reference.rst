:html_theme.sidebar_secondary.remove:

.. _api_reference:

API Reference
=============

Welcome to the technical reference for **Coord2Region**.

While the CLI is great for quick recipes, the Python API is designed for **experienced users**. It allows you to integrate anatomical mapping, literature search, and AI summarization directly into your existing scripts, notebooks, and data processing pipelines.



Why use the Python API?
-----------------------

* **Batch Processing:** Loop through thousands of coordinates efficiently using the :mod:`~coord2region.pipeline`.
* **Custom Workflows:** Use the :class:`~coord2region.fetching.AtlasFetcher` to download atlases for your own use, even outside of Coord2Region.
* **AI Integration:** Import :mod:`~coord2region.llm` to add semantic summaries to your own `pandas` DataFrames or analysis results.

Quick Access
------------

Most users will only need a few core components. Use this cheat sheet to find the right tool for your task:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Goal
     - Component
   * - **Run Everything**
     - :func:`~coord2region.pipeline.run_pipeline`
       The high-level orchestrator. It takes coordinates/regions and returns a complete result object with labels, studies, and summaries.
   * - **Map Coordinates**
     - :class:`~coord2region.coord2region.AtlasMapper`
       The mathematical engine. Use this if you just want to know "What region is at (30, -20, 50)?" without triggering web requests.
   * - **Manage Data**
     - :class:`~coord2region.fetching.AtlasFetcher`
       Handles downloading, caching, and loading brain templates (Nilearn/MNE).
   * - **Query Literature**
     - :func:`~coord2region.coord2study.search_studies`
       Interface with Neurosynth and NeuroQuery to find studies near your coordinates.
   * - **Generate Summaries**
     - :func:`~coord2region.llm.generate_summary`
       Base function to send retrieved studies to an LLM (OpenAI, Gemini) and get back a human-readable summary.
   * - **Configure Settings**
     - :class:`~coord2region.config.Coord2RegionConfig`
       Manages API keys, output directories, and default parameters via YAML or environment variables.



The package is organized into logical sub-systems. You can browse the full tree in the **left navigation sidebar**, but here is a guide to what lives where:

* **`coord2region.pipeline`**: The "glue" code that stitches modules together.
* **`coord2region.fetching`**: Data input. Handles interactions with `nilearn` and `mne` datasets.
* **`coord2region.ai_model_interface`**: External APIs. Contains the logic for querying Neurosynth and NeuroQuery.
* **`coord2region.llm`**: Generative AI. Functions for prompting OpenAI, Gemini, and Hugging Face models.
* **`coord2region.utils`**: Low-level helpers for file I/O, validation, and logging.

.. tip::
   **Pro Tip:** If you are looking for the exact arguments for the CLI, they map 1-to-1 with the arguments in :class:`~coord2region.config.Coord2RegionConfig`.

.. toctree::
   :hidden:
   :maxdepth: 4

   autoapi/coord2region/index