"""
Fetching atlases
================

Showcase the :class:`coord2region.fetching.AtlasFetcher` by listing a few
available atlases and downloading one of them. Also demonstrate how surface
atlases (e.g., FreeSurfer ``aparc``) provide a mapping from region names to
mesh vertices.
"""

# %%
# Create the fetcher and list atlases
from coord2region.fetching import AtlasFetcher
import numpy as np

fetcher = AtlasFetcher()
print("Available atlases:", fetcher.list_available_atlases()[:5])

# %%
# Download the AAL atlas and inspect its labels
atlas_data = fetcher.fetch_atlas("aal")
labels = atlas_data.get("labels", [])
print(f"Fetched AAL atlas with {len(labels)} labels")

# %%
# Surface example: FreeSurfer ``aparc`` (via MNE)
# ------------------------------------------------
# For FreeSurfer surface atlases (``aparc``, ``aparc.a2009s``, ...), the fetcher
# returns per-vertex information for the fsaverage source space (oct6). In this
# case:
# - ``atlas["vol"]`` is a list: [lh_vertices, rh_vertices]
# - ``atlas["labels"]`` has one entry per labeled vertex (left then right)
# - ``atlas["indexes"]`` gives the corresponding vertex IDs in the source space
# - ``atlas["regions"]`` maps region name → array of vertex IDs
#
# We demonstrate how to get all vertices for a given region label.
# Tip: volumetric atlases work in voxel/MNI spaces, while surface atlases
# work in vertex space on a cortical mesh (per hemisphere).
try:
    aparc = fetcher.fetch_atlas("aparc", prefer="mne", subject="fsaverage")

    lh_verts, rh_verts = aparc["vol"]
    print(
        f"Fetched aparc (surface): lh={len(lh_verts)} vertices, rh={len(rh_verts)} vertices"
    )
    print(
        f"Labeled vertices: {len(aparc['labels'])} (left hemisphere first, then right)"
    )

    # find the vertices of source space belonging to a given label
    label = "precentral-lh"
    mask = aparc["labels"] == label
    idx_positions = np.where(mask)[0]
    vertex_ids = aparc["indexes"][mask]
    print(
        f"Label '{label}': {len(idx_positions)} vertices (example IDs: {vertex_ids[:5] if len(vertex_ids) else []})"
    )

    # Alternative: use the regions mapping directly (region → vertex IDs)
    vertex_ids_via_regions = aparc["regions"].get(label, np.array([], dtype=int))
    print(
        f"Via regions['{label}']: {len(vertex_ids_via_regions)} vertices"
    )

    # Note on hemisphere ordering for aparc atlases:
    # the first N entries in ``labels``/``indexes`` correspond to the left
    # hemisphere, followed by the right hemisphere.
except Exception as e:
    # If FreeSurfer data is not available and cannot be fetched, skip gracefully.
    print(f"Skipping 'aparc' surface demo (reason: {e})")
