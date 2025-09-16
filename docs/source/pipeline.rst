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
* **Region names to coordinates** – retrieve representative coordinates for a
  given anatomical label.
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

AI-generated images are watermarked by default with the text
``"AI approximation for illustrative purposes"``. To produce an image without
the watermark, call :func:`coord2region.llm.generate_region_image` with
``watermark=False``.

Command-line interface
----------------------

The ``coord2region`` command exposes common pipeline workflows:

.. code-block:: bash

    # Generate a text summary for a coordinate
    coord2region coords-to-summary 30,-22,50

    # Map a coordinate to atlas labels
    coord2region coords-to-atlas 30,-22,50

    # Produce an image for a coordinate
    coord2region coords-to-image 30,-22,50

    # Convert a region name to an example coordinate
    coord2region region-to-coords "Amygdala"

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
