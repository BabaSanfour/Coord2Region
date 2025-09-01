"""
MEG source localization
=======================

Localize auditory MEG activity from MNE's ``sample`` dataset and map the peak
activation to an anatomical label using Coord2Region. The dataset is accessed
only if already available locally.
"""

# %%
# Load the sample dataset
from pathlib import Path

import mne

from coord2region.fetching import AtlasFetcher
from coord2region.coord2region import AtlasMapper

try:
    data_path = Path(mne.datasets.sample.data_path(download=False))
except Exception:
    print("Sample dataset not available; skipping MEG example.")
    raise SystemExit

required = [
    data_path / "MEG" / "sample" / "sample_audvis-ave.fif",
    data_path / "MEG" / "sample" / "sample_audvis-meg-eeg-oct-6-inv.fif",
    data_path / "subjects",
]
for path in required:
    if not path.exists():
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
    / "sample_audvis-meg-eeg-oct-6-inv.fif",
)

# %%
# Apply the inverse to obtain source estimates and map the peak
stc = mne.minimum_norm.apply_inverse(evoked, inv)
vertex, _ = stc.get_peak()
coord = mne.vertex_to_mni(vertex, 0, "sample", subjects_dir)[0]

fetcher = AtlasFetcher()
atlas = fetcher.fetch_atlas("harvard-oxford")
mapper = AtlasMapper(
    "harvard-oxford", atlas["vol"], atlas["hdr"], atlas["labels"]
)
label = mapper.mni_to_region_name(coord)
print(f"Peak at {coord} lies in {label}")
