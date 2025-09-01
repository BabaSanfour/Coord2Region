"""
fMRI coordinate-to-region workflow
==================================

Map an fMRI activation coordinate to atlas labels using the high-level
pipeline. Study lookup is optional and omitted here to keep the example
lightweight.
"""

# %%
# Run the pipeline on a single coordinate
from coord2region.pipeline import run_pipeline

coord = [[-12, -60, 54]]
results = run_pipeline(
    inputs=coord,
    input_type="coords",
    outputs=["region_labels"],
)

# %%
# Display the resulting labels
res = results[0]
print("Labels:", res.region_labels)

# %%
# To also retrieve related studies, include ``"raw_studies"`` in ``outputs``
# and ensure the required NiMARE dataset is available locally.
