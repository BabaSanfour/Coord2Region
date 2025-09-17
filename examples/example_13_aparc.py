"""Fetch the FreeSurfer ``aparc`` atlas and explore it with ``AtlasMapper``
=============================================================================

This example demonstrates how to work with a surface-based atlas
from MNE/FreeSurfer (``aparc``). It mirrors the Harvard-Oxford examples
but uses a cortical parcellation defined on the surface mesh.

Notes
-----
- ``aparc`` is a surface atlas; region membership is defined by vertex IDs
  on the cortical mesh rather than by voxels in a 3D volume.
- If the FreeSurfer ``fsaverage`` subject is not available locally, MNE will
  fetch it on demand when building the surface mapping, which may require
  internet access.
"""

# %%
# 1. Import Required Libraries
from coord2region.fetching import AtlasFetcher
from coord2region.coord2region import AtlasMapper

# %%
# 2. Fetch the ``aparc`` Atlas (surface-based)
# We explicitly prefer the MNE backend for FreeSurfer atlases.
atlas_name = "aparc"
af = AtlasFetcher()
atlas = af.fetch_atlas(atlas_name, prefer="mne", subject="fsaverage")

# %%
# 3. Load the Atlas into an ``AtlasMapper``
# For surface atlases, pass the ``regions`` mapping so region_name → vertices
# queries work as expected. Subject defaults to ``fsaverage``.
atlas_mapper = AtlasMapper(
    name=atlas_name,
    vol=atlas["vol"],         # [lh_vertices, rh_vertices]
    hdr=atlas["hdr"],         # None for surface atlases
    labels=atlas.get("labels"),
    indexes=atlas.get("indexes"),
    regions=atlas.get("regions"),  # region → vertex IDs
    subject="fsaverage",
)

print(f"Atlas '{atlas_name}' (surface) initialized with "
      f"{len(atlas_mapper.list_all_regions())} regions.")

# %%
# 4. List Available Regions
regions = atlas_mapper.list_all_regions()
print(f"First 10 regions: {regions[:10]} ...")

# %%
# 5. Convert an MNI Coordinate to a Region Name
# The mapping is based on the nearest mesh vertex on the surface.
mni_coord = [-42.0, -22.0, 10.0]
region_from_mni = atlas_mapper.mni_to_region_name(mni_coord)
print(f"MNI {mni_coord} → region: {region_from_mni}")

# %%
# 6. Convert a Region Name to MNI Coordinates
# For surface atlases, this returns the MNI coordinates of all vertices in the
# region. We show the number of vertices and the first coordinate for brevity.
query_region = (
    region_from_mni if isinstance(region_from_mni, str) and region_from_mni != "Unknown" else regions[0]
)
coords_in_region = atlas_mapper.region_name_to_mni(query_region)
print(
    f"Region '{query_region}' has {len(coords_in_region)} vertices;"
    f" example MNI: {coords_in_region[0] if len(coords_in_region) else '[]'}"
)

# %%
# 7. Round-trip Vertex Conversion (vertex ↔ MNI)
# Demonstrate converting the nearest vertex back to an MNI coordinate.
nearest_vertex = int(atlas_mapper.mni_to_vertex(mni_coord))
lh_set = set(map(int, atlas_mapper.vol[0]))
hemi = 0 if nearest_vertex in lh_set else 1
nearest_mni = atlas_mapper.convert_to_mni([nearest_vertex], hemi=hemi)
print(
    f"Nearest vertex: {nearest_vertex} (hemi={hemi}); back to MNI: {nearest_mni}"
)

# %%
# 8. Summary
# - Loaded a surface-based atlas (``aparc``)
# - Mapped MNI coordinates to region names
# - Converted region names to the set of vertex MNI coordinates
# - Demonstrated vertex ↔ MNI round-trip
