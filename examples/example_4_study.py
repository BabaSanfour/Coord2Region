"""Query neuroimaging meta-analysis datasets using ``coord2study``
================================================================

Demonstrates fetching NiMARE datasets and finding studies related to
specific MNI coordinates.
"""

# %%
# 1. Import Required Libraries
# We start by importing the necessary libraries.

from coord2region.coord2study import prepare_datasets, get_studies_for_coordinate
from coord2region.paths import get_working_directory

# %%
# 2. Prepare the NIDM-Pain Dataset
#
# We use `prepare_datasets` to load a cached deduplicated dataset or
# download and create one if necessary. For speed, we only include the
# NIDM-Pain dataset in this example.

working_dir = get_working_directory("coord2region_full")
dataset = prepare_datasets(data_dir=working_dir, sources=["nidm_pain","neurosynth","neuroquery"])
datasets = {"Combined": dataset}

print(f"Loaded dataset with {len(dataset.ids)} studies")

# %%
# 3. Query Studies for an MNI Coordinate
# 
# We specify an **MNI coordinate** to find studies reporting activation at that location.

mni_coord = [48,-38, -24]  # Example coordinate in MNI space

study_results = get_studies_for_coordinate(datasets, coord=mni_coord,radius=2, email="snesmaeili@gmail.com")

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
# - Prepared the **NIDM-Pain** dataset using `prepare_datasets`
# - Queried **studies reporting activation** at a given MNI coordinate
# - Extracted **study titles and abstracts** from the results
#
# This functionality is useful for **meta-analysis research**, allowing users to explore
# which brain regions are consistently activated across multiple studies.
