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

- `coord2region coords-to-summary 30 -22 50` → text summary from related studies
- `coord2region coords-to-image 30 -22 50` → AI-generated region image

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
