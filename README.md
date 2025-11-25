# Coord2Region


[![Codecov](https://img.shields.io/codecov/c/github/BabaSanfour/Coord2Region)](https://codecov.io/gh/BabaSanfour/Coord2Region)
[![Tests](https://img.shields.io/github/actions/workflow/status/BabaSanfour/Coord2Region/python-tests.yml?branch=main&label=tests)](https://github.com/BabaSanfour/Coord2Region/actions/workflows/python-tests.yml)
[![Documentation Status](https://readthedocs.org/projects/coord2region/badge/?version=latest)](https://coord2region.readthedocs.io/en/latest/)
[![Preprint](https://img.shields.io/badge/Preprint-Zenodo-orange)](https://zenodo.org/records/15048848)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/BabaSanfour/Coord2Region)

<div align="left" style="display:flex;gap:1rem;align-items:center;">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/source/_static/images/logo_darkmode.png">
    <img alt="Coord2Region logo" src="docs/source/_static/images/logo.png" width="300">
  </picture>
  <p style="margin:0;">
    <strong>Coord2Region</strong> maps brain coordinates (or atlas region names) to anatomical labels, nearby studies, LLM summaries, and optional AI-generated images. It combines NiMARE, Nilearn, and MNE under a single CLI/Python API and ships with a companion web interface for configuration authoring.
  </p>
</div>

## Why Coord2Region?

- **Atlas + studies in one stop.** Fetch atlases, convert MNI ↔ Talairach, and query datasets such as Neurosynth, NeuroQuery, and NiMARE without wiring them up yourself.
- **Optional AI enrichments.** Provide API keys once (OpenAI, Gemini, Hugging Face, etc.) and the same workflow can emit human-friendly summaries or illustrative images.
- **Reproducible outputs.** Every command can emit YAML, JSON, and CSV artefacts so collaborators can re-run the exact pipeline.
- **Browser builder.** The React/Vite builder mirrors the CLI schema so first-time users can generate configs and commands without installing Python up front.

## Quick Start

1. **Install the package** (Python 3.10+):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install coord2region
   ```

2. **Configure credentials and defaults.** Run the helper once to create a private `config/coord2region-config.yaml`. It covers atlas fetch directories as well as AI provider API keys (all optional).

   ```bash
   python scripts/configure_coord2region.py
   ```

   Prefer environment variables? Set `OPENAI_API_KEY`, `GEMINI_API_KEY`, `HUGGINGFACE_API_KEY`, etc. instead of generating the YAML.

3. **Run a CLI recipe.**

   ```bash
   # Atlas labels only
   coord2region coords-to-atlas 30 -22 50 --atlas harvard-oxford

   # Labels + studies + LLM summary (requires API key)
   coord2region coords-to-summary 30 -22 50 --atlas harvard-oxford --model gemini-2.0-flash

   # Region name workflow
   coord2region region-to-insights "Left Amygdala" --atlas harvard-oxford
   ```

4. **Explore the builder.** Visit the [Config Builder](https://babasanfour.github.io/Coord2Region/builder/) to generate YAML/CLI commands interactively. Import/export configs to stay in sync with local runs.

5. **Jump into Python (optional).**

   ```python
   from coord2region import AtlasFetcher, AtlasMapper, AIModelInterface, generate_summary

   atlas = AtlasFetcher().fetch_atlas("harvard-oxford")
   mapper = AtlasMapper("harvard-oxford", atlas["vol"], atlas["hdr"], atlas["labels"])
   print(mapper.mni_to_region_name([30, -22, 50]))

   ai = AIModelInterface(huggingface_api_key="YOUR_KEY")
   studies = []  # populate via coord2region.coord2study helpers
   print(generate_summary(ai, studies, [30, -22, 50]))
   ```

## CLI recipes at a glance

| Goal | Command |
| --- | --- |
| Labels only | `coord2region coords-to-atlas 30 -22 50 --atlas harvard-oxford` |
| Labels + studies | `coord2region coords-to-study 30 -22 50 --atlas harvard-oxford --radius-mm 10` |
| Labels + studies + summaries | `coord2region coords-to-summary 30 -22 50 --atlas harvard-oxford --model gemini-2.0-flash` |
| Add nilearn anatomical figures | `coord2region coords-to-insights 30 -22 50 --image-backend nilearn` |
| Region → coordinates + insights | `coord2region region-to-insights "Left Amygdala" --atlas harvard-oxford` |

All commands emit YAML/JSON/CSV outputs under `coord2region-output/` by default. Use `--result-dir` to customise the export path.

## Web interface

The web interface mirrors the CLI schema and lives at [babasanfour.github.io/Coord2Region](https://babasanfour.github.io/Coord2Region/). It provides:

- Guided forms for inputs (coordinates or region names), atlas selection, study radius, summaries, and image options.
- Live YAML + CLI previews you can copy or download.
- Presets to learn common workflows (single peak lookup, region → coords, multi-peak insights).
- Import/export so you can iterate on a config in the browser and run the CLI locally.

| ![Config Builder – inputs and atlas](docs/source/_static/images/web-interface-ui-builder1.png) | ![Config Builder – outputs and providers](docs/source/_static/images/web-interface-ui-builder2.png) | ![Runner preview](docs/source/_static/images/web-interface-ui-runner.png) |
| :--: | :--: | :--: |
| Builder (inputs & atlas) | Builder (outputs & providers) | Runner |

To preview or hack on the web stack locally, follow [`web-interface/README.md`](web-interface/README.md) (Vite dev server + Jekyll shell + Playwright tests).

## Further reading

- [Documentation](https://coord2region.readthedocs.io/en/latest/) – user guide, pipeline walkthrough, API reference, tutorials.
- [Examples gallery](examples/)
- [Web interface overview](web-interface/README.md)
- [License][license] · [Contributing][contributing] · [Code of Conduct][code_of_conduct] · [Security Policy][security]
- [Preprint](https://zenodo.org/records/15048848)

## API workflow
A compact overview of the Coord2Region pipeline: shows how inputs (coordinates or region names) are mapped to atlas labels, linked to study results, optionally enriched by AI summaries/images, and exported as reproducible artifacts.

<div align="center">
    <img src="docs/source/_static/images/workflow.jpg" alt="Coord2Region workflow" width="800">
</div>

[license]: LICENSE
[contributing]: CONTRIBUTING.md
[code_of_conduct]: CODE_OF_CONDUCT.md
[security]: SECURITY.md
