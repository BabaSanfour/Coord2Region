"""
========================================
Batch Processing with BatchAtlasMapper
========================================

This example demonstrates how to use `BatchAtlasMapper` for **batch processing**
of MNI coordinates, voxel indices, and region names.

We will:
- Fetch the Harvard-Oxford atlas using `AtlasFetcher`
- Create an `AtlasMapper`
- Use `BatchAtlasMapper` to efficiently convert **multiple** coordinates at once
"""

# %%
# ## 1. Import Required Libraries
# We start by importing the necessary libraries.

import numpy as np
from coord2region.fetching import AtlasFetcher
from coord2region.coord2region import AtlasMapper, BatchAtlasMapper

# %%
# ## 2. Fetch the Harvard-Oxford Atlas
# We use `AtlasFetcher` to download the atlas into `atlas_data/`.

atlas_name = "harvard-oxford"
af = AtlasFetcher(data_dir="atlas_data")  # Download into "atlas_data" directory
atlas = af.fetch_atlas(atlas_name)

# %%
# ## 3. Load the Atlas into an `AtlasMapper`
# We now initialize an `AtlasMapper` using the fetched atlas data.

atlas_mapper = AtlasMapper(
    name=atlas_name,
    vol=atlas["vol"],   # 3D atlas volume (NumPy array)
    hdr=atlas["hdr"],   # Affine transformation matrix (4x4)
    labels=atlas["labels"],  # Region labels
)

print(f"Atlas '{atlas_name}' initialized with {len(atlas['labels'])} regions.")

# %%
# ## 4. Create a `BatchAtlasMapper`
# The `BatchAtlasMapper` allows efficient batch processing of coordinates.

batch_mapper = BatchAtlasMapper(atlas_mapper)
print("BatchAtlasMapper initialized.")

# %%
# ## 5. Convert Multiple MNI Coordinates to Region Names
# We can convert multiple MNI coordinates at once to **brain region names**.

mni_coords = [
    [-20, 30, 40],   # Example MNI coordinate
    [40, -20, 30],   # Another coordinate
    [-10, 50, -20]   # A third coordinate
]

region_names = batch_mapper.batch_mni_to_region_name(mni_coords)
for i, (coord, region) in enumerate(zip(mni_coords, region_names)):
    print(f"Coordinate {i+1} {coord} is in region: {region}")

# %%
# ## 6. Convert Region Names to MNI Coordinates
# Given multiple **region names**, we can retrieve all **MNI coordinates** in each region.

region_queries = ["Frontal Pole", "Insular Cortex", "Superior Frontal Gyrus"]
mni_results = batch_mapper.batch_region_name_to_mni(region_queries)

for region, coords in zip(region_queries, mni_results):
    print(f"Region '{region}' has {len(coords)} MNI coordinates.")

# %%
# ## 7. Convert Multiple MNI Coordinates to Voxel Indices
# We can efficiently convert MNI coordinates to voxel space.

voxel_coords = batch_mapper.batch_mni_to_voxel(mni_coords)
for i, (mni, voxel) in enumerate(zip(mni_coords, voxel_coords)):
    print(f"MNI coordinate {mni} maps to voxel index {voxel}")

# %%
# ## 8. Convert Multiple Voxel Indices to MNI Coordinates
# We can also convert **voxel indices back to MNI coordinates**.

mni_from_voxels = batch_mapper.batch_voxel_to_mni(voxel_coords)
for i, (voxel, mni) in enumerate(zip(voxel_coords, mni_from_voxels)):
    print(f"Voxel index {voxel} maps to MNI coordinate {mni}")

# %%
# ## 9. Convert Region Indexes to MNI Coordinates
# We can retrieve **all MNI coordinates** associated with **multiple region indices**.

region_indices = [1, 10, 20]  # Example region indices
mni_from_indices = batch_mapper.batch_region_index_to_mni(region_indices)

for idx, mni_coords in zip(region_indices, mni_from_indices):
    print(f"Region index {idx} has {len(mni_coords)} MNI coordinates.")

# %%
# ## 10. Summary
#
# In this tutorial, we:
# - Downloaded the Harvard-Oxford atlas using `AtlasFetcher`
# - Created an `AtlasMapper`
# - Used `BatchAtlasMapper` to convert multiple:
#   - MNI coordinates to **region names**
#   - Region names to **MNI coordinates**
#   - MNI coordinates to **voxels**
#   - Voxels back to **MNI coordinates**
#
# This allows efficient **batch processing** for neuroimaging workflows.
