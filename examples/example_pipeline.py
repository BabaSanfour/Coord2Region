"""Mixed output pipeline example
==============================

This script demonstrates how to generate a text summary and an illustrative
image for a coordinate while saving all results to a PDF file. The dataset is
cached between runs so repeated executions don't reprocess or re-download the
data.

Make sure to set the appropriate API keys (e.g. ``OPENAI_API_KEY`` or
``GEMINI_API_KEY``) so that the language model and image generation providers
are available.
"""

import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from coord2region.pipeline import run_pipeline
# The example coordinate ([30, -22, 50]) falls in the right precentral gyrus (primary motor cortex) in MNI space.
# For more on MNI coordinates, see: https://en.wikipedia.org/wiki/Talairach_coordinates#MNI_template
# Coordinate of interest
coord = [[30, -22, 50]]

try:
    results = run_pipeline(
        inputs=coord,
        input_type="coords",
        outputs=["region_labels", "summaries", "images"],
        output_format="pdf",
        output_name="example_pipeline.pdf",
        # Reuse the cached dataset to avoid repeated downloads or processing
        config={"use_cached_dataset": True},
    )

    logger.info("Summary:\n%s", results[0].summary)
    logger.info("Image saved to: %s", results[0].image)
except Exception as err:  # pylint: disable=broad-except
    logger.error("An error occurred while running the pipeline: %s", err)
    if os.getenv("ENV") == "development":
        raise
