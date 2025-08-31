# Coord2Region

[![Codecov](https://img.shields.io/codecov/c/github/BabaSanfour/Coord2Region)](https://codecov.io/gh/BabaSanfour/Coord2Region)
[![Test Status](https://img.shields.io/github/actions/workflow/status/BabaSanfour/Coord2Region/python-tests.yml?branch=main&label=tests)](https://github.com/BabaSanfour/Coord2Region/actions?query=workflow%3Apython-tests)
[![Documentation Status](https://readthedocs.org/projects/coord2region/badge/?version=latest)](https://coord2region.readthedocs.io/en/latest/?badge=latest)
[![GitHub Repository](https://img.shields.io/badge/Source%20Code-BabaSanfour%2FCoord2Region-blue)](https://github.com/BabaSanfour/Coord2Region)
[![Preprint](https://img.shields.io/badge/Preprint-Zenodo-orange)](https://zenodo.org/records/15048848)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD3Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

**Coord2Region** is an open-source Python package designed to simplify neuroimage workflows by automatically mapping 3D coordinates in brain space (e.g. MNI or Talairach) to anatomical brain regions across multiple widely used atlases. This package supports both single-coordinate and high-throughput batch queries. Furthermore, it integrates smoothly with external meta-analytic resources through the Neuroimaging Meta-Analysis Research Environment (NiMARE), which includes Neurosynth and NeuroQuery databases. By streamlining methods to associate 3D brain coordinates with a range of anatomical and functional labels, Coord2Region promotes robust and reproducible research practices in neuroimaging.


## Features

- **Automated Anatomical Mapping:** Quickly retrieve region names for any given MNI coordinate.
- **Batch Processing:** Efficiently map large sets of coordinates using vectorized operations.
- **Multiple Atlas Support:** Compare and cross-validate results using different atlases (e.g., Harvard-Oxford, Juelich, Schaefer, Yeo).
- **Meta-analytic Integration:** Interface with Neurosynth and NeuroQuery for enriched functional interpretations.
- **Extensible and Modular:** Easily add new atlases or extend functionality.

## Installation

### Prerequisites

- Python 3.8 or higher
- Nilearn 0.11 or newer

### Installing for End Users

You can install coord2region in your own virtual environment using either conda or Python's venv:

#### Using Conda
```bash
conda create -n coord2region python=3.8
conda activate coord2region
pip install -r requirements.txt
```

#### Using venv
```bash
python -m venv env
# On Linux/Mac:
source env/bin/activate
# On Windows:
env\Scripts\activate
pip install -r requirements.txt
```

Alternatively, clone the repository and install using pip:
```bash
git clone https://github.com/BabaSanfour/Coord2Region.git
cd coord2region
pip install .
```

### Installing for Development

If you wish to contribute or modify the package, set up a development environment:
1. Create and activate your virtual environment (see above).
2. Install the runtime requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Install additional development requirements:
   ```bash
   pip install -r requirements-dev.txt
   ```
4. (Optional) Run tests with:
   ```bash
   pytest
   ```

## Quickstart Pipeline

The high-level pipeline combines atlas lookup, study queries and optional AI
generation. It can return multiple output types in one call and optionally
export the results.

```python
from coord2region.pipeline import run_pipeline

results = run_pipeline(
    inputs=[[30, -22, 50]],
    input_type="coords",
    outputs=["region_labels", "summaries", "images"],
    output_format="pdf",
    output_path="quickstart.pdf",
)

print(results[0].summary)
print("Image saved to", results[0].image)
```

See `examples/example_pipeline.py` for a complete script demonstrating mixed
outputs and PDF export.

The ``image_backend`` parameter controls how images are generated. Set it to
``"nilearn"`` to create a simple brain overlay without relying on an AI model,
or ``"both"`` to obtain outputs from both approaches:

```python
results = run_pipeline(
    inputs=[[30, -22, 50]],
    input_type="coords",
    outputs=["images"],
    image_backend="nilearn",
)
print("Nilearn image saved to", results[0].images["nilearn"])
```

AI-generated images are watermarked by default with the text
``"AI approximation for illustrative purposes"``. To produce an image without
the watermark, call :func:`coord2region.llm.generate_region_image` with
``watermark=False``.

## Data directory

All downloads and generated files are stored in a configurable base directory.
By default this is ``~/coord2region`` but a different location can be supplied
via the ``data_dir`` parameter used throughout the library.  The helper
``coord2region.utils.resolve_data_dir`` expands the user supplied path and
creates several standard subdirectories:

- ``cached_data/`` – cached NiMARE datasets and intermediate files.
- ``generated_images/`` – images produced by :func:`pipeline.run_pipeline`.
- ``results/`` – exported outputs such as JSON or PDF reports.

Relative paths passed to functions like ``output_path`` are resolved against the
base directory, allowing simple names such as ``"results/report.pdf"`` to be
placed within the coord2region workspace.

## Environment Variables

Some features rely on external AI services. Provide the following optional
environment variables to enable them:

- `GEMINI_API_KEY` – Google's Generative AI (Gemini) models.
- `OPENROUTER_API_KEY` – OpenRouter access for DeepSeek models.
- `OPENAI_API_KEY` – OpenAI's GPT models (e.g. GPT‑4).
- `ANTHROPIC_API_KEY` – Anthropic's Claude models.
- `HUGGINGFACE_API_KEY` or `HUGGINGFACEHUB_API_TOKEN` – HuggingFace Inference API.
- `AI_MODEL_PROVIDERS` – Comma separated list of providers to enable. If unset,
  all providers with available API keys are registered.
- `DALLE_API_KEY` – OpenAI's DALL·E image generation service.
- `STABILITY_API_KEY` – Stability AI image generation service.

Set these variables in your shell before running examples or tests, for example:

```bash
export GEMINI_API_KEY="..."
export OPENROUTER_API_KEY="..."
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

### Selecting AI providers

The :class:`coord2region.ai_model_interface.AIModelInterface` can dispatch to
multiple backends. Providers are enabled when their API key is supplied and can
be limited with the ``AI_MODEL_PROVIDERS`` environment variable or the
``enabled_providers`` constructor argument.

```python
from coord2region.ai_model_interface import AIModelInterface

# Enable only OpenAI and Gemini providers
ai = AIModelInterface(
    openai_api_key="sk-...", gemini_api_key="...", enabled_providers=["openai", "gemini"]
)
print(ai.list_available_models())
response = ai.generate_text(model="gpt-4", prompt="Hello")
```

### Adding a Custom LLM Provider

To support additional large language models, create a subclass of
`ModelProvider` and register it with
`AIModelInterface.register_provider`. The provider should define a
dictionary of model names and implement `generate_text`.

```python
from coord2region.ai_model_interface import AIModelInterface, ModelProvider

class EchoProvider(ModelProvider):
    """Minimal provider that echoes the prompt."""

    def __init__(self):
        super().__init__({"echo-1": "echo-1"})

    def generate_text(self, model, prompt, max_tokens):
        if isinstance(prompt, str):
            return prompt
        return " ".join(m["content"] for m in prompt)

ai = AIModelInterface()
ai.register_provider(EchoProvider())
print(ai.generate_text("echo-1", "Hello"))
```

See `examples/custom_provider_example.py` for a complete runnable example.

## Usage

Below is a simple example of how to use coord2region in your Python code:

```python
from coord2region.coord2region import AtlasMapper, BatchAtlasMapper, MultiAtlasMapper
from coord2region.fetching import AtlasFetcher

# Fetch an atlas (e.g., Harvard-Oxford)
af = AtlasFetcher()
atlas_data = af.fetch_atlas('harvard-oxford')

# Create an AtlasMapper
mapper = AtlasMapper(
    name='harvard-oxford',
    vol=atlas_data['vol'],
    hdr=atlas_data['hdr'],
    labels=atlas_data['labels']
)

# Map a single MNI coordinate to its region name
mni_coord = [30, -22, 50]
region_name = mapper.mni_to_region_name(mni_coord)
print(f"The region for coordinate {mni_coord} is {region_name}")

# For batch processing:
batch_mapper = BatchAtlasMapper(mapper)
region_names = batch_mapper.batch_mni_to_region_name([[30, -22, 50], [10, 20, 30]])
print("Batch mapped regions:", region_names)
```

To fetch meta-analytic datasets and query studies for a coordinate:

```python
from coord2region.coord2study import prepare_datasets, get_studies_for_coordinate

# Load a cached deduplicated dataset (stored in ``~/coord2region/cached_data`` by
# default) or fetch and create one if missing. Pass ``data_dir`` to override the
# storage location.
dataset = prepare_datasets(sources=["nidm_pain"])
studies = get_studies_for_coordinate({"Combined": dataset}, coord=[30, -22, 50])
print(f"Found {len(studies)} studies")
```

### Running the high-level pipeline

The :func:`coord2region.pipeline.run_pipeline` function ties together atlas
lookup, study queries and optional AI generation. Results can be exported in
several formats using the ``output_format`` argument:

```python
from coord2region.pipeline import run_pipeline

run_pipeline(
    inputs=[[30, -22, 50]],
    input_type="coords",
    outputs=["region_labels"],
    output_format="csv",
    output_path="results.csv",
)
```

Supported formats include ``"json"``, ``"pickle"``, ``"csv"``, ``"pdf"`` and
``"directory"``.

For more examples and detailed API usage, please refer to the code in `combined_python_code.txt` and the examples in the repository.

### Adding NiMARE datasets and refreshing the cache

The `fetch_datasets` utility supports multiple NiMARE-compatible sources such as
`"neurosynth"`, `"neuroquery"` and `"nidm_pain"`. To add a new dataset, extend
`fetch_datasets` with the required download and conversion logic and expose it
with a unique key. Users can then request it explicitly:

```python
from coord2region.coord2study import fetch_datasets, prepare_datasets
fetch_datasets("coord2region", sources=["neurosynth", "my_new_dataset"])
```

Deduplicated datasets are cached in `<data_dir>/cached_data/deduplicated_dataset.pkl.gz`
(by default `~/coord2region/cached_data/deduplicated_dataset.pkl.gz`). When new
datasets are added or existing ones updated, remove this file or call
`prepare_datasets` with the desired `sources` to rebuild the cache and keep it
current.

## Documentation, Examples and Paper

For a comprehensive overview and detailed description of the package design, please see the 
**coord2region** [documentation](https://coord2region.readthedocs.io/en/latest/) and [Preprint](https://zenodo.org/records/15048848)

## Contributing

Contributions are welcome! If you would like to contribute:

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add YourFeature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

Please ensure that you have installed the development dependencies and that all tests pass before submitting a pull request.

## License

This project is open-source. Please refer to the LICENSE file for details.

## Citation

If you use coord2region in your research, please cite the associated paper.
