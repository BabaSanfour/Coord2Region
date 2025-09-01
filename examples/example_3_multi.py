"""Use ``MultiAtlasMapper`` to query multiple atlases simultaneously
==================================================================

The example fetches several atlases, converts coordinates to regions, and
retrieves region coordinates across parcellations.
"""

# %%
# 1. Import Required Libraries
# We start by importing the necessary libraries.

from coord2region.fetching import AtlasFetcher
from coord2region.coord2region import MultiAtlasMapper

# %%
# 2. Fetch Multiple Atlases
# We use `AtlasFetcher` to download the Harvard-Oxford and Schaefer atlases.

atlas_names = ["harvard-oxford", "schaefer"]
af = AtlasFetcher()  # Download into default package directory; you can specify a custom directory data_dir="path/to/dir"

# Dictionary specifying fetch arguments for each atlas
atlases = {
    "harvard-oxford": {},  # Specify Harvard-Oxford sub-atlas
    "schaefer": {} # use default from nilearn
}

# %%
# 3. Create a `MultiAtlasMapper`
# This initializes multiple atlases and allows simultaneous querying.

multi_mapper = MultiAtlasMapper(data_dir=af.data_dir, atlases=atlases) # make sure to use the same data_dir as the fetcher

print(f"MultiAtlasMapper initialized with {len(multi_mapper.mappers)} atlases.")

# %%
# 4. Convert MNI Coordinates to Region Names Across Atlases
# Let's see which **brain regions** the same **MNI coordinates** belong to in each atlas.

mni_coords = [
    [-20, 30, 40],   # Example MNI coordinate
    [40, -20, 30],   # Another coordinate
    [-10, 50, -20]   # A third coordinate
]

region_names_per_atlas = multi_mapper.batch_mni_to_region_names(mni_coords)

for atlas_name, region_names in region_names_per_atlas.items():
    print(f"\nAtlas: {atlas_name}")
    for coord, region in zip(mni_coords, region_names):
        print(f"  MNI {coord} → Region: {region}")

# %%
# 5. Convert Region Names to MNI Coordinates Across Atlases
# Given a **region name**, we retrieve all **MNI coordinates** corresponding to that region in each atlas.
# We will plot the first two MNI coordinates for brevity.
region_queries = ["Frontal Pole", "Insular Cortex", "Superior Frontal Gyrus"]

mni_results = multi_mapper.batch_region_name_to_mni(region_queries)

for atlas_name, coords_list in mni_results.items():
    print(f"\nAtlas: {atlas_name}")
    for region, coords in zip(region_queries, coords_list):
        print(f"  Region '{region}' → {len(coords)} MNI coordinates found.")
        if len(coords) > 2:
            print(f"  Example MNI coordinates: {coords[:2]}...")

# %%
# 6. Summary
#
# In this tutorial, we:
# - Downloaded the Harvard-Oxford and Schaefer atlases using `AtlasFetcher`
# - Created a `MultiAtlasMapper`
# - Converted **MNI coordinates** to **region names** in both atlases
# - Retrieved **MNI coordinates** for a given region across multiple atlases
#
# This enables **multi-atlas region mapping**, making it ideal for **comparative neuroimaging studies**.
