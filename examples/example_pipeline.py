"""Mixed output pipeline example.

This script demonstrates how to generate a text summary and an illustrative
image for a coordinate while saving all results to a PDF file. The dataset is
cached between runs so repeated executions don't reprocess or re-download the
data.

Make sure to set the appropriate API keys (e.g. ``OPENAI_API_KEY`` or
``GEMINI_API_KEY``) so that the language model and image generation providers
are available.
"""

from coord2region.pipeline import run_pipeline

# Coordinate of interest
coord = [[30, -22, 50]]

results = run_pipeline(
    inputs=coord,
    input_type="coords",
    outputs=["region_labels", "summaries", "images"],
    output_format="pdf",
    output_path="example_pipeline.pdf",
    # Reuse the cached dataset to avoid repeated downloads or processing
    brain_insights_kwargs={"use_cached_dataset": True},
)

print("Summary:\n", results[0].summary)
print("Image saved to:", results[0].image)
