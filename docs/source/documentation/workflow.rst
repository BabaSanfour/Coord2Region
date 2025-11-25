.. _documentation-workflow:

Concepts & Workflow
===================

Every Coord2Region run follows the same lifecycle, whether you are running a 
single coordinate in the CLI or a batch of thousands in Python. 



1. Gather Inputs
----------------

The pipeline accepts two primary input types:

* **Coordinates**: MNI or Talairach coordinates (x, y, z).
* **Region Names**: Anatomical labels (e.g., "Left Amygdala") which are first 
  mapped to centroids using a reference atlas.

2. Configure Providers
----------------------

Coord2Region is modular. You must decide which external services to enable:

* **Atlases**: Local anatomical lookups (e.g., Harvard-Oxford). See :doc:`../atlases`.
* **Literature**: Meta-analytic databases (Neurosynth/NeuroQuery) for finding studies.
* **AI Providers**: LLMs (OpenAI, Gemini) for semantic summaries and images. See :doc:`../providers`.

*Tip: Run ``python scripts/configure_coord2region.py`` to generate a persistent YAML config.*

3. Choose an Interface
----------------------

Select the tool that matches your technical comfort level:

* **Web Builder**: Design pipelines visually and export the config. See :doc:`web_interface`.
* **CLI**: The standard way to run reproducible recipes. See :doc:`cli_tools`.
* **Python API**: For direct integration into scripts or notebooks. See :doc:`../pipeline`.

4. Inspect Artefacts
--------------------

By default, results land in ``coord2region-output/``.

* **Structured Data**: JSON and CSV files containing the raw mappings and study lists.
* **Reports**: PDF or Markdown summaries of the findings.
* **Media**: Generated brain overlays or AI illustrations.
* **Provenance**: A YAML copy of the exact configuration used to generate the run.