Pipeline
========

The :func:`coord2region.pipeline.run_pipeline` helper ties together atlas
lookups, study queries and optional generative AI to provide an end-to-end
workflow.  It accepts different kinds of inputs and can produce multiple output
types in a single call.

.. figure:: ../static/images/workflow.jpg
  :alt: Pipeline workflow
  :align: center
  :width: 90%

  Coord2Region pipeline overview.

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
        output_name="results.pdf",
        config={"use_cached_dataset": False},
    )

    print(results[0].summary)
    print("Image saved to", results[0].image)

Each :class:`coord2region.pipeline.PipelineResult` also exposes a ``warnings``
list. When a region name cannot be mapped to any configured atlas, the pipeline
records an explanatory warning instead of failing, allowing you to surface the
issue to end users.

The ``output_name`` argument is a simple filename or directory name without
path separators. It is automatically created inside the ``results`` subfolder
of the working directory.

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
   not inside a Python session. For example, to retrieve studies and labels:

   ``coord2region coords-to-study 30 -22 50``

.. code-block:: bash

    # Map a coordinate to specific atlas labels
    coord2region coords-to-atlas 30 -22 50 --atlas harvard-oxford

    # Retrieve studies (includes atlas labels)
    coord2region coords-to-study 30 -22 50 --atlas harvard-oxford

    # Generate a text summary (labels + studies)
    coord2region coords-to-summary 30 -22 50 --atlas harvard-oxford

    # Produce an image for a coordinate with the nilearn backend
    coord2region coords-to-image 30 -22 50 --image-backend nilearn

    # Create a full report with summary and image outputs
    coord2region coords-to-insights 30 -22 50 --atlas harvard-oxford --image-backend nilearn

    # Load many coordinates from a file
    coord2region coords-to-study --coords-file path/to/coords.csv --output-format csv --output-name results.csv

    # Convert a region name to insights (atlas must be explicit)
    coord2region region-to-insights "Left Amygdala" --atlas harvard-oxford

Common options:

- ``--atlas``: Select atlas name(s). Repeat the flag or pass a comma-separated list.
  Region-based commands require exactly one atlas.
- ``--coords-file``: Load coordinates from CSV/XLSX (first three columns are used).
- ``--output-format`` and ``--output-name``: Export results as JSON, pickle,
  CSV, PDF or a directory. The name is created inside the working directory.
- ``--working-directory``: Base directory for caches, generated images and results.
- ``--sources`` and ``--email-for-abstracts``: Control dataset selection and
  contact email when searching studies. Available on commands that retrieve studies.
- ``--image-backend`` (image-producing commands): Choose between ``ai``,
  ``nilearn`` or ``both``. The CLI defaults to ``nilearn`` for offline usage.
- Provider keys for AI features: ``--gemini-api-key``, ``--openrouter-api-key``,
  ``--openai-api-key``, ``--anthropic-api-key`` and ``--huggingface-api-key``.

Configuration files
-------------------

Complex runs can be described in YAML and executed with ``--config``:

.. code-block:: yaml

    inputs:
      - [30, -22, 50]
    input_type: coords
    outputs: [region_labels, raw_studies, summaries, images]
    output_format: pdf
    output_name: results.pdf

.. code-block:: bash

    coord2region --config my_pipeline.yml

Ensure that any required AI provider API keys (e.g. ``OPENAI_API_KEY`` or
``GEMINI_API_KEY``) are set in the environment to enable summary or image
generation.
