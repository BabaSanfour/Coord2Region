"""
Coordinate to study lookup
==========================

Find studies reporting a coordinate with :mod:`coord2region.coord2study`. To
keep the example lightweight, the Neurosynth dataset is queried only if it is
already present locally.

Data download
-------------

This example uses NiMARE-compatible datasets (e.g., Neurosynth, NeuroQuery).
To keep the docs build lightweight it does not trigger downloads. To fetch
datasets beforehand into the same cache location used below, run:

.. code-block:: python

    from coord2region.coord2study import fetch_datasets
    # Choose a cache directory and the sources you want
    data_dir = "~/.coord2region_examples"
    fetch_datasets(data_dir=data_dir, sources=["neurosynth"])  # or ["neuroquery"], etc.

You can pass multiple sources (e.g., ["neurosynth", "neuroquery"]). The first
run downloads and converts datasets; subsequent runs reuse the cache.
"""

# %%
# Attempt to load a cached Neurosynth dataset without downloading
from pathlib import Path
from coord2region.coord2study import fetch_datasets, get_studies_for_coordinate

# Choose a cache directory; this should match wherever you pre-fetched data
data_dir = Path("~/.coord2region_examples").expanduser()

if (data_dir / "neurosynth").exists():
    # Reuse cached datasets (no network activity if already present)
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
    # Query a coordinate (MNI) within a small search radius
    studies = get_studies_for_coordinate(datasets, [0, -52, 26], radius=5, email=email)
    print(f"Found {len(studies)} studies near [0, -52, 26] (showing up to 3):")
    for study in studies[:3]:
        print(study.get("id"), study.get("title"))
