"""Fetch the Harvard-Oxford atlas and explore it with ``AtlasMapper``
=====================================================================

The example downloads the atlas using ``AtlasFetcher`` and shows how to
convert between region names, indices, and MNI coordinates.
"""

# %%
# 1. Import Required Libraries
# We start by importing the necessary libraries.

from coord2region.fetching import AtlasFetcher
from coord2region.coord2region import AtlasMapper

# %%
# 2. Fetch the Harvard-Oxford Atlas
# We use `AtlasFetcher` to download the atlas into default package data_dir.

atlas_name = "harvard-oxford"
af = AtlasFetcher()  # Download into default package directory; you can specify a custom directory data_dir="path/to/dir"
atlas = af.fetch_atlas(atlas_name)

# The fetched atlas contains:
print(f"Atlas keys: {list(atlas.keys())}")

# %%
# 3. Load the Atlas into an `AtlasMapper`
# We now initialize an `AtlasMapper` using the fetched atlas data.

atlas_mapper = AtlasMapper(
    name=atlas_name,
    vol=atlas["vol"],   # 3D atlas volume (NumPy array)
    hdr=atlas["hdr"],   # Affine transformation matrix (4x4)
    labels=atlas["labels"],  # Region labels
)

print(f"Atlas '{atlas_name}' initialized with {len(atlas['labels'])} regions.")

# %%
# 4. List Available Regions
# The `list_all_regions()` method returns the names of all regions in the atlas.

all_regions = atlas_mapper.list_all_regions()
print(f"Available regions:\n{all_regions[:10]}...")  # Print first 10 regions

# %%
# 5. Convert a Region Index to a Name
# Let's retrieve the region name for a **specific index**.

region_idx = 10  # Example index
region_name = atlas_mapper.region_name_from_index(region_idx)

print(f"Region Index {region_idx} corresponds to: {region_name}")

# %%
# 6. Convert a Region Name to an Index
# We can also find the **index of a region by its name**.

region_query = "Frontal Pole"  # Change this to any region name
region_index = atlas_mapper.region_index_from_name(region_query)

print(f"Region '{region_query}' corresponds to index: {region_index}")

# %%
# 7. Convert MNI Coordinates to Voxel Space
# We use `mni_to_voxel()` to find the **voxel location** of an MNI coordinate.
# You could also use `mni_to_vertices()` for surface atlases or `convert_to_mni()` that supports both.
mni_coord = [-20, 30, 40]  # Example MNI coordinate
voxel_index = atlas_mapper.mni_to_voxel(mni_coord) 

print(f"MNI coordinate {mni_coord} maps to voxel index {voxel_index}")

# %%
# 8. Convert MNI Coordinates to Region Name
# Now, let's find which **region** corresponds to an MNI coordinate.

region_from_mni = atlas_mapper.mni_to_region_name(mni_coord)
print(f"MNI coordinate {mni_coord} is in region: {region_from_mni}")

# %%
# 9. Infer Hemisphere from Region Name (1.0)
# Finally, we can infer the **hemisphere** of a region if its name follows a standard format.
region_name = "Frontal Pole" 
hemisphere = atlas_mapper.infer_hemisphere(region_name)

print(f"Region '{region_name}' belongs to hemisphere: {hemisphere}")

# %% 10. Infer Hemisphere from Region Name (1.0)
# As you have seen the **Frontal Pole** does not belong to any **hemisphere**. This is because 
# some atlases do not provide hemisphere information for regions.
# Let's try another region that has hemisphere information.
region_name = "Lat_Fis-post-rh"  # This label belongs to the Destrieux atlas
region_index = atlas_mapper.region_index_from_name(region_name)

if region_index == "Unknown":
    print(f"Region '{region_name}' is not part of the {atlas_name} atlas.")
else:
    hemisphere = atlas_mapper.infer_hemisphere(region_name)
    print(f"Region '{region_name}' belongs to hemisphere: {hemisphere}")

# %%
# 11. Summary
#
# In this tutorial, we:
# - Downloaded the Harvard-Oxford atlas using `AtlasFetcher`
# - Loaded it into `AtlasMapper`
# - Converted between MNI coordinates, voxel indices, and region names
# - Listed available regions
# - Inferred the hemisphere of a given region
#
# This allows efficient **brain region mapping** for neuroimaging analysis.
