"""Demonstrate exporting pipeline results to various formats
==========================================================

This example queries atlas labels for a single coordinate and saves the output
as CSV, PDF, and a directory of JSON files.
"""

# %%
from coord2region.pipeline import run_pipeline

coord = [[30, -22, 50]]

# Save results to CSV
run_pipeline(
    inputs=coord,
    input_type="coords",
    outputs=["region_labels"],
    output_format="csv",
    output_name="results.csv",
    config={"use_cached_dataset": False},
)

# Save results to PDF
run_pipeline(
    inputs=coord,
    input_type="coords",
    outputs=["region_labels"],
    output_format="pdf",
    output_name="results.pdf",
    config={"use_cached_dataset": False},
)

# Save results to a directory with JSON and image copies
run_pipeline(
    inputs=coord,
    input_type="coords",
    outputs=["region_labels"],
    output_format="directory",
    output_name="results_dir",
    config={"use_cached_dataset": False},
)
