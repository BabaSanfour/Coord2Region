"""Batch processing with ``BatchAtlasMapper`` using the ``aparc`` atlas
======================================================================

This example mirrors the Harvard-Oxford batch tutorial but uses a
surface-based FreeSurfer atlas (``aparc``) via MNE. It demonstrates
mapping multiple MNI coordinates to region names and converting
region names to all vertex MNI coordinates.

Note that ``aparc`` is a surface atlas. There is no voxel space here;
operations are defined in terms of mesh vertices and their MNI positions.
"""

# %%
# Step 1: Imports
from coord2region.fetching import AtlasFetcher
from coord2region.coord2region import AtlasMapper, BatchAtlasMapper

# %%
# Step 2: Fetch the ``aparc`` (surface) atlas from MNE/FreeSurfer
atlas_name = "aparc"
af = AtlasFetcher()
atlas = af.fetch_atlas(atlas_name, prefer="mne", subject="fsaverage")

# %%
# Step 3: Create an ``AtlasMapper``
atlas_mapper = AtlasMapper(
    name=atlas_name,
    vol=atlas["vol"],
    hdr=atlas["hdr"],
    labels=atlas.get("labels"),
    indexes=atlas.get("indexes"),
    regions=atlas.get("regions"),
    subject="fsaverage",
)
print(f"Atlas '{atlas_name}' initialized (surface).")

# %%
# Step 4: Wrap in a ``BatchAtlasMapper``
batch_mapper = BatchAtlasMapper(atlas_mapper)
print("BatchAtlasMapper ready.")

# %%
# Step 5: Batch convert MNI coordinates → region names
mni_coords = [
    [-42, -22, 10],
    [40, -25, 15],
    [-10, -70, 20],
]
region_names = batch_mapper.batch_mni_to_region_name(mni_coords)
for coord, region in zip(mni_coords, region_names):
    print(f"MNI {coord} → {region}")

# %%
# Step 6: Batch convert region names → all MNI coordinates (per region)
# Use a few aparc-style labels (suffix indicates hemisphere: -lh / -rh).
region_queries = [
    "superiortemporal-lh",
    "insula-rh",
    "superiorfrontal-lh",
]
region_mni_sets = batch_mapper.batch_region_name_to_mni(region_queries)
for region, coords in zip(region_queries, region_mni_sets):
    print(f"Region '{region}' → {len(coords)} vertices")
    if len(coords) > 0:
        print(f"  example MNI: {coords[0]}")

# %%
# Step 7: Summary
# - Fetched the surface-based aparc atlas
# - Batch-mapped MNI coordinates to region names
# - Converted region names to the full set of vertex MNI coordinates

