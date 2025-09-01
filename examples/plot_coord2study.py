"""
Coordinate to study lookup
==========================

Find studies reporting a coordinate with :mod:`coord2region.coord2study`. To
keep the example lightweight, the Neurosynth dataset is queried only if it is
already present locally.
"""

# %%
# Attempt to load a cached Neurosynth dataset
from pathlib import Path
from coord2region.coord2study import fetch_datasets, get_studies_for_coordinate

data_dir = Path("~/.coord2region_examples").expanduser()
if (data_dir / "neurosynth").exists():
    datasets = fetch_datasets(data_dir=str(data_dir), sources=["neurosynth"])
else:
    datasets = {}
    print("Neurosynth dataset not found; skipping download in docs build.")

# %%
# Query studies if a dataset is available
if datasets:
    studies = get_studies_for_coordinate(datasets, [0, -52, 26], radius=5)
    for study in studies[:3]:
        print(study.get("id"), study.get("title"))
