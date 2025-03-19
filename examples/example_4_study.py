"""
==========================================================
Using NiMARE Datasets for Neuroscience Meta-Analysis
==========================================================

This tutorial demonstrates how to use `coord2study` to query **neuroimaging meta-analysis datasets** 
for studies related to given **MNI coordinates**.

We will:
- Fetch the **NIDM-Pain dataset** using `fetch_datasets`
- Query **studies related to a given brain coordinate**
- Extract **study metadata** such as title and abstract (if available)

**Note:** To speed up execution, we limit this example to **NIDM-Pain only**, but you can include other datasets 
by setting `neurosynth=True` and `neuroquery=True` in `fetch_datasets`.
"""

# %%
# 1. Import Required Libraries
# We start by importing the necessary libraries.

import os
from coord2region.coord2study import fetch_datasets, get_studies_for_coordinate

# %%
# 2. Fetch the NIDM-Pain Dataset
# 
# We use `fetch_datasets` to download the **NIDM-Pain dataset**, which contains **neuroimaging meta-analysis studies**.

home_dir = os.path.expanduser("~")
data_dir = os.path.join(home_dir, 'coord2region') # Use package directory for data storage
os.makedirs(data_dir, exist_ok=True)

datasets = fetch_datasets(data_dir=data_dir, neurosynth=False, neuroquery=False)  # Only use NIDM-Pain

print(f"Loaded datasets: {list(datasets.keys())}")

# %%
# 3. Query Studies for an MNI Coordinate
# 
# We specify an **MNI coordinate** to find studies reporting activation at that location.

mni_coord = [48,-38, -24]  # Example coordinate in MNI space

study_results = get_studies_for_coordinate(datasets, coord=mni_coord)

# Display results
print(f"\nFound {len(study_results)} studies for MNI coordinate {mni_coord}:\n")
for study in study_results[:5]:  # Show only first 5 studies for brevity
    print(f"Study ID: {study['id']}")
    print(f"Source: {study['source']}")
    print(f"Title: {study.get('title', 'No title available')}")
    print("-" * 40)

# %%
# 4. Extract and Display Study Metadata
# 
# If available, we can retrieve additional metadata **such as abstracts** using **PubMed**.

for study in study_results[:3]:  # Limit to first 3 studies
    print(f"Study ID: {study['id']}")
    print(f"Title: {study.get('title', 'No title available')}")
    if "abstract" in study:
        print(f"Abstract: {study['abstract'][:300]}...")  # Show only first 300 characters
    print("=" * 60)

# %%
# 5. Summary
#
# In this tutorial, we:
# - Loaded the **NIDM-Pain** dataset using `fetch_datasets`
# - Queried **studies reporting activation** at a given MNI coordinate
# - Extracted **study titles and abstracts** from the results
#
# This functionality is useful for **meta-analysis research**, allowing users to explore
# which brain regions are consistently activated across multiple studies.
