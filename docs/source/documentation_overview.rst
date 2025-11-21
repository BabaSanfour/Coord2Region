Documentation Overview
======================

Use this page as the section navigator for Coord2Region. Every heading below mirrors the layout in the main sidebar so you can jump straight to the resource that matches your current task.

.. contents:: Section Navigation
   :local:
   :depth: 1

Tutorials
---------

Start with the :doc:`tutorials` collection to see Coord2Region in action. Each walkthrough mixes atlas lookups, literature retrieval, and AI-ready summaries so you can copy/paste only the bits you need.

Examples
--------

The :doc:`auto_examples/index` gallery contains short, focused recipes. Use them to answer "how do I…?" questions quickly—each example bundles runnable code cells, CLI invocations, and the resulting artefacts.

Glossary
--------

Key terms you will meet across the docs:

- **Atlas Mapper** – the object that turns 3D coordinates into atlas labels and vice versa.
- **Provider** – an integration that fetches studies, runs AI models, or talks to hosted services.
- **Recipe** – a CLI or YAML configuration describing what Coord2Region should produce.
- **Builder** – the web UI that mirrors CLI/YAML schemas and exports ready-to-run configs.

Implementation details
----------------------

See :doc:`providers` for the underlying integrations—AI models, NiMARE/Nilearn support, and how data providers interoperate with Coord2Region pipelines.

Design philosophy
-----------------

The :doc:`roadmap` shares how we prioritize reliability, provenance, and low-friction developer ergonomics. It is the best entry point if you want to understand why features land the way they do.

Example datasets
----------------

:doc:`atlases` documents every bundled atlas, how to add your own directories, and what metadata flows into Coord2Region outputs.

Command-line tools
------------------

The :doc:`README` and :doc:`pipeline` pages cover the ``coord2region`` CLI, the YAML schema it consumes, and how command outputs line up with the Python API.

Migrating from other analysis software
--------------------------------------

Coming from Nilearn, NiMARE, or MNE-Python already? Coord2Region wraps their primitives so you can reuse familiar atlases, masker logic, and study selection while leaning on a unified CLI/API.

The typical M/EEG workflow
--------------------------

The :doc:`pipeline` guide follows the same steps we use in CoCo Lab: gather coordinates, map them to atlas regions, link the relevant papers, and export structured artefacts for downstream notebooks or dashboards.

How to cite Coord2Region
------------------------

Grab ready-to-use CITATION.cff details plus BibTeX/APA snippets from :ref:`cite-coord2region`. Please cite Coord2Region (and any atlases or providers you use) in publications and presentations.

Papers citing Coord2Region
--------------------------

We maintain a running list of talks, posters, and manuscripts inside :ref:`contributors`. Submit a pull request or open an issue whenever you publish a Coord2Region-powered result.
