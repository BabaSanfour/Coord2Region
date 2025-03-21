"""
=====================================================
End-to End Pipeline: Multi-Atlas Coordinate Querying
=====================================================
This example shows a straightfoward end-to-end pipeline for querying brain regions and studies for a set of MNI coordinates.
The pipeline:
- Generates a toy CSV file with 10 random brain MNI coordinates.
- Loads 4 atlases (Harvard-Oxford, Schaefer, yeo, juelich) using `AtlasFetcher`.
- Retrieves the corresponding brain regions in each atlas.
- Saves the coordinate-to-region mappings as a CSV file.
- Queries studies that reported these coordinates and saves them as JSON files.
"""
# %%
import os
import json
import numpy as np
import pandas as pd
from coord2region.fetching import AtlasFetcher
from coord2region.coord2region import MultiAtlasMapper
from coord2region.coord2study import fetch_datasets, get_studies_for_coordinate
# %%
# Step 1: Generate 10 random MNI coordinates
# -----------------------------------------
np.random.seed(42)
mni_coordinates = np.random.randint(-60, 60, size=(10, 3))  # Generate 10 random (x, y, z) points
mni_list = mni_coordinates.tolist()
# Save coordinates to a CSV file
csv_filename = "toy_coordinates.csv"
df = pd.DataFrame(mni_list, columns=["x", "y", "z"])
df.to_csv(csv_filename, index=False)
print(f"Saved toy coordinates to {csv_filename}")
# df = pd.read_csv("coordinates.csv"); instead of generating random coordinates
# %%
# Step 2: Fetch and load multiple atlases
# -----------------------------------------
af = AtlasFetcher()
atlases = {
    "harvard-oxford": {},
    "schaefer": {},
    "yeo": {},
    "juelich": {},
}
# %%
# Step 3: Create MultiAtlasMapper
# -----------------------------------------
multi_mapper = MultiAtlasMapper(data_dir=af.data_dir, atlases=atlases)
# %%
# Step 4: Map each coordinate to regions in all four atlases
# ----------------------------------------------------------
results = []
for coord in mni_list:
    regions = multi_mapper.batch_mni_to_region_names([coord])
    result = {"x": coord[0], "y": coord[1], "z": coord[2]}
    for atlas, region_list in regions.items():
        result[atlas] = region_list[0] if region_list else "Unknown"
    results.append(result)
# Save results to a CSV file
mapped_csv_filename = "mapped_coordinates.csv"
df_mapped = pd.DataFrame(results)
df_mapped.to_csv(mapped_csv_filename, index=False)
print(f"Saved mapped coordinates to {mapped_csv_filename}")
# %%
# Step 5: Fetch NiMARE datasets (only NIDM-Pain for speed)
# --------------------------------------------------------
dataset_dir = af.data_dir
# neurosynth and neuroquey take a long time to download for the sake of this example we will only use NIDM-Pain
datasets = fetch_datasets(dataset_dir, neurosynth=False, neuroquery=False)
# %%
# Step 6: Query studies for each coordinate and save to JSON
# ----------------------------------------------------------
output_dir = "studies_per_coordinate"
os.makedirs(output_dir, exist_ok=True)
for coord in mni_list:
    studies = get_studies_for_coordinate(datasets, coord)
    coord_filename = f"{output_dir}/studies_{coord[0]}_{coord[1]}_{coord[2]}.json"
    with open(coord_filename, "w") as f:
        json.dump(studies, f, indent=4)
    print(f"Saved studies for coordinate {coord} to {coord_filename}")
# %%
# Summary:
# - We generated 10 random MNI coordinates and saved them to `toy_coordinates.csv`.
# - We mapped each coordinate to 4 brain atlases and saved the results to `mapped_coordinates.csv`.
# - We querie