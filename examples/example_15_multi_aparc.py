"""Query multiple FreeSurfer surface atlases with ``MultiAtlasMapper``
=====================================================================

This example mirrors the multi-atlas tutorial but uses three
FreeSurfer-based surface parcellations via MNE:

- ``aparc``
- ``aparc.a2009s``
- ``aparc.a2005s``

We batch-map MNI coordinates to region names across these atlases.
"""

# %%
# 1. Imports
from coord2region.fetching import AtlasFetcher
from coord2region.coord2region import MultiAtlasMapper
import mne
from pathlib import Path

# %%
# 2. Define atlases and initialize ``MultiAtlasMapper``
# Explicitly prefer the MNE backend for FreeSurfer atlases.
af = AtlasFetcher()
# Try to locate a subjects_dir for FreeSurfer (optional but recommended)
subjects_dir = mne.get_config("SUBJECTS_DIR", None)
if subjects_dir is None:
    try:
        subjects_dir = str(Path(mne.datasets.sample.data_path(download=False)) / "subjects")
    except Exception:
        subjects_dir = None
common = {"prefer": "mne", "subject": "fsaverage"}
if subjects_dir is not None:
    common = {**common, "subjects_dir": subjects_dir}

atlases = {
    "aparc": {**common},
    "aparc.a2009s": {**common},
    "aparc.a2005s": {**common},
}
multi = MultiAtlasMapper(data_dir=af.data_dir, atlases=atlases)
print(f"Loaded {len(multi.mappers)} surface atlases.")

# %%
# 3. Batch map MNI coordinates to region names across the atlases
mni_coords = [
    [-42, -22, 10],
    [40, -25, 15],
    [-10, -70, 20],
]
results = multi.batch_mni_to_region_names(mni_coords)
for atlas_name, names in results.items():
    print(f"\nAtlas: {atlas_name}")
    for coord, name in zip(mni_coords, names):
        print(f"  MNI {coord} → {name}")

# %%
# 4. Summary
# - Initialized multiple FreeSurfer surface atlases
# - Compared region-name lookups for the same coordinates across parcellations
