"""
MEG source localization
=======================

Localize auditory MEG activity from MNE's ``sample`` dataset and map the peak
activation to an anatomical label using Coord2Region. This example uses an
MNE/FreeSurfer surface-based atlas (``aparc``) to demonstrate surface
label mapping.

Data download
-------------

This example requires the MNE "sample" dataset (including the FreeSurfer
subjects directory). The script intentionally avoids downloading data
automatically. To fetch the dataset beforehand, run the following in a Python
session:

.. code-block:: python

    import mne
    # Downloads the dataset to the default MNE data folder
    mne.datasets.sample.data_path()

You can control the download location using the environment variable
``MNE_DATA`` or by passing a ``path=...`` argument to ``data_path()``. The
example will then discover the dataset and use ``<data_path>/subjects`` as the
``subjects_dir``.

Notes on vertices vs. MNI coordinates
-------------------------------------
Surface atlases store region membership as sets of vertex IDs (integers on the
mesh), while peak locations are typically expressed in MNI coordinates
(floating-point positions). Therefore, comparing a peak MNI coordinate to a
label's vertices will never match. Instead, check whether the peak vertex
identifier (``peak_vertno``) is included in the region's vertex list, e.g.::

    peak_vertno in atlas['regions']['superiortemporal-lh']  # -> True

By design, ``mapper.mni_to_region_name(coord)`` determines the nearest region
label to the provided MNI coordinate. The exact vertex membership lives under
``atlas['regions'][label]``.
"""

# %%
# Load the sample dataset
from pathlib import Path

import mne
from coord2region.fetching import AtlasFetcher
from coord2region.coord2region import AtlasMapper
import numpy as np

try:
    data_path = Path(mne.datasets.sample.data_path(download=False))
except Exception:
    print("Sample dataset not available; skipping MEG example.")
    raise SystemExit

required = [
    data_path / "MEG" / "sample" / "sample_audvis-ave.fif",
    data_path / "MEG" / "sample" / "sample_audvis-meg-eeg-oct-6-meg-eeg-inv.fif",
    data_path / "subjects",
]
for path in required:
    if not path.exists():
        print("Missing:", path)
        print("Sample dataset not available; skipping MEG example.")
        raise SystemExit

subjects_dir = data_path / "subjects"

# %%
# Read an evoked response and a precomputed inverse operator
evoked = mne.read_evokeds(
    data_path / "MEG" / "sample" / "sample_audvis-ave.fif",
    condition="Left Auditory",
    baseline=(None, 0),
)
inv = mne.minimum_norm.read_inverse_operator(
    data_path
    / "MEG"
    / "sample"
    / "sample_audvis-meg-eeg-oct-6-meg-eeg-inv.fif",
)

# %%
# Apply the inverse to obtain source estimates and map the peak
stc = mne.minimum_norm.apply_inverse(evoked, inv)
# get the peak vertex (as vertno) and determine hemisphere
peak_vertno, _ = stc.get_peak()
peak_vertno = int(peak_vertno)
lh_set = set(map(int, stc.vertices[0]))
peak_hemi = 0 if peak_vertno in lh_set else 1
coord = mne.vertex_to_mni([peak_vertno], peak_hemi, "sample", subjects_dir)[0]

fetcher = AtlasFetcher()
# Use an MNE/FreeSurfer atlas. We pass the subject and subjects_dir so that
# surface vertices are correctly mapped to MNI coordinates for this dataset.
atlas = fetcher.fetch_atlas(
    "aparc", prefer="mne", subject="sample", subjects_dir=str(subjects_dir)
)
mapper = AtlasMapper(
    name="aparc",
    vol=atlas["vol"],
    hdr=atlas["hdr"],
    labels=atlas.get("labels"),
    indexes=atlas.get("indexes"),
    subject="sample",
    subjects_dir=str(subjects_dir),
    regions=atlas.get("regions"),
    system="mni",
)
label = mapper.mni_to_region_name(coord)
print(f"Peak at {coord} (vertex={peak_vertno}, hemi={peak_hemi}) lies in {label}")

# Surface atlas sanity checks
regions = atlas.get("regions", {})
region_vertices = set(map(int, regions.get(label, [])))
if region_vertices:
    print("Vertex membership:", peak_vertno in region_vertices)

# mapper vs MNE vertex->MNI agreement
mapper_vtx_coord = mapper.vertex_to_mni([peak_vertno], peak_hemi)
mne_vtx_coord = mne.vertex_to_mni([peak_vertno], peak_hemi, "sample", subjects_dir)
print("Vertex->MNI coords match:", np.allclose(mapper_vtx_coord, mne_vtx_coord))

# Distance to region centroid from peak MNI coordinate
centroid = mapper.region_centroid(label)
if centroid.size:
    dist_centroid = float(np.linalg.norm(np.asarray(coord) - centroid))
    print(f"Distance to {label} centroid: {dist_centroid:.2f} mm")

# Round-trip: nearest vertex from the MNI peak coordinate
nearest_vtx = mapper.mni_to_vertex(coord, hemi=peak_hemi)
if isinstance(nearest_vtx, np.ndarray):
    nearest_vtx = int(nearest_vtx.ravel()[0]) if nearest_vtx.size else -1
print(
    f"Nearest vertex from coord: {nearest_vtx}; matches peak: {nearest_vtx == peak_vertno}"
)

