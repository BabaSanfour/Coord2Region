Pipeline
========

The :func:`coord2region.pipeline.run_pipeline` helper ties together atlas
lookups, study queries and optional generative AI to provide an end-to-end
workflow.  It accepts different kinds of inputs and can produce multiple output
types in a single call.

Supported use cases
-------------------

* **Coordinates to region labels** – map MNI/Talairach coordinates to atlas
  region names.
* **Coordinates to summaries** – fetch studies near a coordinate and generate a
  language‑model summary.
* **Coordinates to images** – create an illustrative image for the queried
  coordinate.
* **Region names to coordinates** – retrieve representative MNI coordinates for
  a given anatomical label (use the ``mni_coordinates`` output).
* **Batch processing and exporting** – process many items at once and export the
  results as JSON, pickle, CSV, PDF or a directory of files.

Python usage
------------

A minimal pipeline combining atlas labels, a textual summary and an illustrative
image while saving the results to a PDF file:

.. code-block:: python

    from coord2region.pipeline import run_pipeline

    results = run_pipeline(
        inputs=[[30, -22, 50]],
        input_type="coords",
        outputs=["region_labels", "summaries", "images"],
        output_format="pdf",
        output_path="results.pdf",
        config={"use_cached_dataset": False},
    )

    print(results[0].summary)
    print("Image saved to", results[0].image)

Each :class:`coord2region.pipeline.PipelineResult` also exposes a ``warnings``
list. When a region name cannot be mapped to any configured atlas, the pipeline
records an explanatory warning instead of failing, allowing you to surface the
issue to end users.

AI-generated images are watermarked by default with the text
``"AI approximation for illustrative purposes"``. To produce an image without
the watermark, call :func:`coord2region.llm.generate_region_image` with
``watermark=False``.

Command-line interface
----------------------

The ``coord2region`` command exposes common pipeline workflows. Coordinates can
be provided in multiple forms: as a single string (``30,-22,50`` or ``"30 -22 50"``)
or as three separate numbers (``30 -22 50``). You can also pass a CSV/XLSX file
via ``--coords-file``.

.. note::
   The examples below are shell commands. Run them in your terminal (bash/zsh/PowerShell),
   not inside a Python session. For example, to summarize a coordinate:

   ``coord2region coords-to-summary 30 -22 50``

.. code-block:: bash

    # Generate a text summary for a coordinate (all styles equivalent)
    coord2region coords-to-summary 30,-22,50
    coord2region coords-to-summary "30 -22 50"
    coord2region coords-to-summary 30 -22 50

    # Map a coordinate to specific atlas labels
    coord2region coords-to-atlas 30 -22 50 --atlas harvard-oxford

    # Use multiple atlases (repeat --atlas or comma-separate)
    coord2region coords-to-atlas 30 -22 50 --atlas harvard-oxford --atlas juelich
    coord2region coords-to-atlas 30 -22 50 --atlas harvard-oxford,juelich,aal

    # Produce an image for a coordinate with chosen backend
    coord2region coords-to-image 30 -22 50 --image-backend nilearn
    coord2region coords-to-image 30 -22 50 --image-backend both

    # Load many coordinates from a file
    coord2region coords-to-atlas --coords-file path/to/coords.csv --output-format csv --output-path results.csv

    # Convert a region name to an example coordinate (field 'mni_coordinates')
    coord2region region-to-coords "Amygdala"

Common options:

- ``--atlas``: Select atlas name(s). Repeat the flag or pass a comma-separated list.
  Defaults to ``harvard-oxford,juelich,aal``.
- ``--coords-file``: Load coordinates from CSV/XLSX (first three columns are used).
- ``--output-format`` and ``--output-path``: Export results as JSON, pickle,
  CSV, PDF or a directory of files.
- ``--data-dir``: Base directory for caches, generated images and results.
- ``--image-backend`` (``coords-to-image``): Choose between ``ai``, ``nilearn``
  or ``both``.
- Provider keys (if using AI summaries/images): ``--gemini-api-key``,
  ``--openrouter-api-key``, ``--openai-api-key``, ``--anthropic-api-key``,
  ``--huggingface-api-key``.

Configuration files
-------------------

Complex runs can be described in YAML and executed with ``--config``:

.. code-block:: yaml

    inputs:
      - [30, -22, 50]
    input_type: coords
    outputs: [region_labels, summaries, images]
    output_format: pdf
    output_path: results.pdf

.. code-block:: bash

    coord2region --config my_pipeline.yml

Ensure that any required AI provider API keys (e.g. ``OPENAI_API_KEY`` or
``GEMINI_API_KEY``) are set in the environment to enable summary or image
generation.
