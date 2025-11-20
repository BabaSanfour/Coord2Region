Coord2Region
============

.. raw:: html

   <div class="logo-banner" style="text-align:center;margin:1.5rem 0;">
     <picture>
       <source media="(prefers-color-scheme: dark)" srcset="../static/images/logo_darkmode.png">
       <img src="../static/images/logo.png" alt="Coord2Region logo" style="max-width:420px;width:90%;height:auto;">
     </picture>
   </div>

**Coord2Region** is your bridge from coordinates to context: atlas regions, related studies, optional AI summaries, and reproducible outputs. It bundles NiMARE, Nilearn, and MNE under a unified CLI + Python interface, plus a companion web builder and cloud runner.

Overview
--------

.. figure:: ../static/images/workflow.jpg
   :alt: Coord2Region workflow overview
   :align: center
   :width: 90%

   High-level workflow from inputs to outputs.

Web interface previews
----------------------

.. |ui1| image:: ../static/images/web-interface-ui-builder1.png
   :alt: Config Builder – inputs and atlas
   :width: 31%

.. |ui2| image:: ../static/images/web-interface-ui-builder2.png
   :alt: Config Builder – outputs and providers
   :width: 31%

.. |ui3| image:: ../static/images/web-interface-ui-runner.png
   :alt: Runner preview
   :width: 31%

|ui1| |ui2| |ui3|

Quick Start (Choose Your Path)
------------------------------

1. **Install the package** (Python 3.10+):

   .. code-block:: bash

      python -m venv .venv
      source .venv/bin/activate
      pip install coord2region

2. **Configure credentials.** Run :mod:`scripts.configure_coord2region` once to create a private ``config/coord2region-config.yaml`` (atlas directories and API keys). Alternatively, set environment variables such as ``OPENAI_API_KEY`` or ``GEMINI_API_KEY``.

3. **Run a CLI recipe.**

   .. code-block:: bash

      # Atlas labels only
      coord2region coords-to-atlas 30 -22 50 --atlas harvard-oxford

      # Labels + studies + summary (API key required)
      coord2region coords-to-summary 30 -22 50 --atlas harvard-oxford --model gemini-2.0-flash

      # Region name → coordinates + insights bundle
      coord2region region-to-insights "Left Amygdala" --atlas harvard-oxford

   All commands emit YAML/JSON/CSV artefacts inside ``coord2region-output/`` by default.

4. **Switch to Python (optional).**

   .. code-block:: python

      from coord2region import AtlasFetcher, AtlasMapper, AIModelInterface, generate_summary

      atlas = AtlasFetcher().fetch_atlas("harvard-oxford")
      mapper = AtlasMapper("harvard-oxford", atlas["vol"], atlas["hdr"], atlas["labels"])
      print(mapper.mni_to_region_name([30, -22, 50]))

      ai = AIModelInterface(huggingface_api_key="YOUR_KEY")
      studies = []  # fetch with coord2region.coord2study helpers
      print(generate_summary(ai, studies, [30, -22, 50]))

5. **Pick your control surface.**

   - **Builder.** The React/Vite builder at https://babasanfour.github.io/Coord2Region/ mirrors the CLI schema, offers presets, and keeps YAML/CLI previews in sync.
   - **Cloud Runner.** Already packaged a config? Submit it through the hosted runner, stream logs, and download YAML/JSON/CSV/images without touching a local environment.

Explore More
------------

- Tutorials and runnable examples are in the gallery below.
- The user guide covers the end-to-end :mod:`coord2region.pipeline`.
- The API reference lists public classes and functions.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   README

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   pipeline
   atlases

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials

.. toctree::
   :maxdepth: 1
   :caption: Examples

   auto_examples/index

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   autoapi/index

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide

   developer_guide

.. toctree::
   :maxdepth: 1
   :caption: Roadmap

   roadmap
