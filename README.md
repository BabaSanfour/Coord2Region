# Coord2Region

[![Codecov](https://img.shields.io/codecov/c/github/BabaSanfour/Coord2Region)](https://codecov.io/gh/BabaSanfour/Coord2Region)
[![Tests](https://img.shields.io/github/actions/workflow/status/BabaSanfour/Coord2Region/python-tests.yml?branch=main&label=tests)](https://github.com/BabaSanfour/Coord2Region/actions/workflows/python-tests.yml)
[![Documentation Status](https://readthedocs.org/projects/coord2region/badge/?version=latest)](https://coord2region.readthedocs.io/en/latest/)
[![Preprint](https://img.shields.io/badge/Preprint-Zenodo-orange)](https://zenodo.org/records/15048848)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)

**Coord2Region** maps 3D brain coordinates to anatomical regions, retrieves related studies, and uses large language models to summarize findings or generate images.

## Features

- Automatic anatomical labeling across multiple atlases
- LLM-powered summaries of nearby literature
- Coordinate-to-study lookups via Neurosynth, NeuroQuery, etc.
- AI-generated region images
- Command-line and Python interfaces

## Workflow

![Coord2Region workflow](https://raw.githubusercontent.com/BabaSanfour/Coord2Region/main/docs/_static/images/workflow.jpg)

## Web interface (previews)

| ![Config Builder – inputs and atlas](https://raw.githubusercontent.com/BabaSanfour/Coord2Region/main/docs/_static/images/web-interface-ui-builder1.png) | ![Config Builder – outputs and providers](https://raw.githubusercontent.com/BabaSanfour/Coord2Region/main/docs/_static/images/web-interface-ui-builder2.png) | ![Runner preview](https://raw.githubusercontent.com/BabaSanfour/Coord2Region/main/docs/_static/images/web-interface-ui-runner.png) |
| :--: | :--: | :--: |
| Builder (inputs & atlas) | Builder (outputs & providers) | Runner |

## Web interface

The interactive configuration builder is published at
[https://babasanfour.github.io/Coord2Region/](https://babasanfour.github.io/Coord2Region/). A dedicated GitHub Actions workflow
builds the schema, compiles the Vite bundle, runs the Playwright UI checks, and
deploys the Jekyll site to GitHub Pages whenever `main` is updated. To preview
the site locally, install the `web-interface/` dependencies and run `npm run dev`
alongside `bundle exec jekyll serve --livereload` (see `web-interface/README.md`
for the full walkthrough).

## Installation

Requires Python 3.10 or later. We recommend installing in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install coord2region
```

To work on Coord2Region itself, install the optional dependencies:

```bash
pip install '.[dev]'    # linting and tests
pip install '.[docs]'   # documentation build
```

On shells like zsh, keep the extras spec in quotes to avoid glob expansion errors.

Set environment variables like `OPENAI_API_KEY` or `GEMINI_API_KEY` to enable LLM-based features.

## Example

```bash
coord2region coords-to-atlas 30 -22 50 --atlas harvard-oxford
```

Other use cases:

- `coord2region coords-to-study 30 -22 50` → atlas labels and related studies
- `coord2region coords-to-summary 30 -22 50` → labels, studies and an AI summary
- `coord2region coords-to-image 30 -22 50 --image-backend nilearn` → labels, studies and a rendered image
- `coord2region coords-to-insights 30 -22 50 --atlas harvard-oxford` → full report with labels, studies, summary and image

Region-driven workflows:

- `coord2region region-to-coords "Left Amygdala" --atlas harvard-oxford` → retrieve the atlas coordinate
- `coord2region region-to-insights "Left Amygdala" --atlas harvard-oxford` → coordinates, studies, summary and image for that region

Full usage instructions and API details are available in the [documentation](https://coord2region.readthedocs.io/en/latest/).

## Links

- [Documentation](https://coord2region.readthedocs.io/en/latest/)
- [License][license]
- [Contributing][contributing]
- [Code of Conduct][code_of_conduct]
- [Security Policy][security]
- [Preprint](https://zenodo.org/records/15048848)

[license]: LICENSE
[contributing]: CONTRIBUTING.md
[code_of_conduct]: CODE_OF_CONDUCT.md
[security]: SECURITY.md
