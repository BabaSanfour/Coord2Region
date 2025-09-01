"""
High level pipeline
===================

The :func:`coord2region.pipeline.run_pipeline` function combines atlas lookup,
study search and optional AI summaries. This example runs the pipeline on a
single coordinate and returns region labels.
"""

# %%
# Import the pipeline helper
from coord2region.pipeline import run_pipeline

# %%
# Run the pipeline on a coordinate
results = run_pipeline(
    inputs=[[0, -52, 26]],
    input_type="coords",
    outputs=["region_labels"],
)

# %%
# Show the resulting labels
print(results[0].region_labels)
