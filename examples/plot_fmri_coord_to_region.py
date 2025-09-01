"""
fMRI coordinate-to-region workflow
==================================

Map an fMRI activation coordinate to atlas labels using the high-level
pipeline. Study lookup is optional and omitted here to keep the example
lightweight.
"""

from pathlib import Path

from coord2region.fetching import AtlasFetcher
from coord2region.pipeline import run_pipeline

# %%
# Check that the Harvard-Oxford atlas is available locally
fetcher = AtlasFetcher()
atlas_file = (
    Path(fetcher.nilearn_data)
    / "harvard-oxford"
    / "HarvardOxford-cort-maxprob-thr25-2mm.nii.gz"
)

if atlas_file.exists():
    # %%
    # Run the pipeline on a single coordinate
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
else:
    print("Harvard-Oxford atlas not found; skipping fMRI example.")

# %%
# To also retrieve related studies, include ``"raw_studies"`` in ``outputs``
# and ensure the required NiMARE dataset is available locally.
