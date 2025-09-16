"""
Coordinate to study lookup
==========================

Find studies reporting a coordinate with :mod:`coord2region.coord2study`. To
keep the example lightweight, the Neurosynth dataset is queried only if it is
already present locally.

Data download
-------------

This example uses NiMARE-compatible datasets (e.g., Neurosynth, NeuroQuery).
It does not download automatically during the docs build. To fetch datasets
beforehand into the same cache location used below:

.. code-block:: python

    from coord2region.coord2study import fetch_datasets
    # Choose a cache directory and the sources you want
    data_dir = "~/.coord2region_examples"
    fetch_datasets(data_dir=data_dir, sources=["neurosynth"])  # or ["neuroquery"], etc.

You can pass multiple sources (e.g., ["neurosynth", "neuroquery"]). The first
run downloads and converts datasets; subsequent runs reuse the cache.
"""

# %%
# Attempt to load a cached Neurosynth dataset
from pathlib import Path
import os
from coord2region.coord2study import fetch_datasets, get_studies_for_coordinate
from coord2region.coord2study import fetch_datasets
# Choose a cache directory and the sources you want
data_dir = "~/.coord2region_examples"
fetch_datasets(data_dir=data_dir, sources=["neurosynth"])  # or ["neuroquery"], etc.

data_dir = Path("~/coord2region").expanduser()
if (data_dir / "neurosynth").exists():
    datasets = fetch_datasets(data_dir=str(data_dir), sources=["neurosynth"])
else:
    datasets = {}
    print("Neurosynth dataset not found; skipping download in docs build.")

# %%
# Query studies if a dataset is available
email = "coord@example.com"
if datasets:
    # Optionally set your email (for PubMed/Entrez courtesy) via ENV:
    #   export ENTREZ_EMAIL="you@example.com"
    studies = get_studies_for_coordinate(datasets, [0, -52, 26], radius=5, email=email)
    for study in studies[:3]:
        print(study.get("id"), study.get("title"))
