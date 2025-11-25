"""Basic usage of :class:`AtlasMapper`
================================================================

This example converts between coordinate systems and queries brain regions.
See also the API docs for
:class:`coord2region.coord2region.AtlasMapper`.
"""

# %%
# 1. Importing Required Libraries
# First, we import the necessary libraries, including NumPy and `AtlasMapper`.

import numpy as np
from coord2region import AtlasMapper

# %%
# 2. Creating a Simple Example Atlas
# To demonstrate `AtlasMapper`, we create a small synthetic 3D atlas.
# The atlas is a 10x10x10 NumPy array where each voxel represents a different region.
# We also define a corresponding **affine transformation matrix** for MNI-space mapping.

atlas_size = (10, 10, 10)
atlas_data = np.zeros(atlas_size, dtype=int)

# Define some arbitrary region indices
atlas_data[2:5, 2:5, 2:5] = 1  # Region 1
atlas_data[6:9, 6:9, 6:9] = 2  # Region 2

# Define an affine transformation (identity matrix for simplicity)
affine = np.eye(4)

# Define labels and region indices
labels = { "1": "Frontal Cortex", "2": "Occipital Cortex" }
indices = [1, 2]

# %%
# 3. Initializing the `AtlasMapper`
# Now, we create an `AtlasMapper` instance using the synthetic atlas data.

atlas = AtlasMapper(
    name="ExampleAtlas",
    vol=atlas_data,
    hdr=affine,
    labels=labels,
    indexes=indices,
    system="mni"
)

print(f"Atlas Name: {atlas.name}")
print(f"Atlas Shape: {atlas.shape}")
print(f"Atlas Coordinate System: {atlas.system}")
print(f"Atlas Coordinate System: {atlas.atlas_type}") # The AtlasMapper can detect the type of atlas

# %%
# 4. Converting MNI Coordinates to Voxel Indices
# We use `convert_to_source()` to convert an MNI coordinate (x, y, z) into voxel space.
# This method works for both volumetric and surface atlases. `mni_to_voxel()` function works for volumetric atlases 
# and  `mni_to_vertex()` works for surface atlases.
# Since our affine matrix is the identity matrix, this is a simple rounding operation.

mni_coord = [3, 3, 3]
voxel_idx = atlas.convert_to_source(mni_coord)

print(f"MNI Coordinate {mni_coord} maps to Voxel Index {voxel_idx}")

# %%
# 5. Retrieving a Region Name from a Voxel Index
# We use `mni_to_region_name()` to determine which brain region a voxel belongs to.

region_name = atlas.mni_to_region_name(mni_coord)
print(f"Voxel {voxel_idx} belongs to region: {region_name}")

# %%
# 6. Listing All Available Regions
# The `list_all_regions()` method returns all defined region names.

all_regions = atlas.list_all_regions()
print(f"Available Regions: {all_regions}")

# %%
# 7. Converting a Region Index to MNI Coordinates
# The `region_index_to_mni()` method returns all MNI coordinates corresponding to a given region.

region_idx = 1
region_mni_coords = atlas.region_index_to_mni(region_idx)

print(f"Region Index {region_idx} has {len(region_mni_coords)} coordinates in MNI space.")

# %%
# 8. Converting Voxel Indices Back to MNI
# We use `convert_to_mni()` to reverse the voxel-to-MNI conversion. The method works for both volumetric and surface atlases.
# `voxel_to_mni()` works for volumetric atlases and `vertext_to_mni()` works for surface atlases. 

mni_from_voxel = atlas.convert_to_mni([voxel_idx])
print(f"Voxel {voxel_idx} maps back to MNI coordinates: {mni_from_voxel}")

# %%
# 9. Handling Surface Atlases
# We can also use `AtlasMapper` for **surface-based** atlases.
# In this case, we define a list of vertices instead of a volumetric atlas.
# You can test 'convert_to_source()' and 'convert_to_mni()' methods with surface atlases.
# Or you can use 'mni_to_vertex()' and 'vertex_to_mni()' methods.

surface_vertices = np.array([100, 200, 300])  # Example vertex indices
atlas_surface = AtlasMapper(
    name="SurfaceAtlas",
    vol=surface_vertices.tolist(),  # Volumetric data is replaced with a list of vertices
    hdr=None,  # No affine matrix for surface atlases
    labels={ "100": "Visual Cortex", "200": "Motor Cortex", "300": "Somatosensory Cortex" },
    indexes=[100, 200, 300],
    system="fsaverage"
)

print(f"Surface Atlas Regions: {atlas_surface.list_all_regions()}")

# %%
# 10. Summary
# In this tutorial, we explored:
#
# - Initializing an `AtlasMapper` for volumetric and surface atlases
# - Converting between MNI and voxel coordinates
# - Querying brain region names and indices
# - Listing available regions
#
# This provides a foundation for working with neuroimaging atlases in Python.
