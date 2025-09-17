"""
iEEG electrode localization
===========================

Load electrode coordinates from the MNE epilepsy ECoG dataset and assign each
contact to an atlas region. The dataset is used only if it is already
available locally.

Data download
-------------

This example requires the MNE epilepsy ECoG dataset. The script avoids
downloading automatically. To fetch the dataset beforehand, run in a Python
session:

.. code-block:: python

    import mne
    # Downloads the dataset to the default MNE data folder
    mne.datasets.epilepsy_ecog.data_path()

You can control the download location using the ``MNE_DATA`` environment
variable or by passing ``path=...`` to ``data_path()``. The example will then
discover the dataset and look for electrodes under the appropriate subject
folder.
"""

# %%
# Load electrode coordinates
from pathlib import Path

import pandas as pd
import mne

from coord2region.fetching import AtlasFetcher
from coord2region.coord2region import AtlasMapper

try:
    data_path = Path(mne.datasets.epilepsy_ecog.data_path(download=False))
except Exception:
    data_path = None
    print("ECoG dataset not available; skipping electrode mapping.")

if data_path is not None:
    electrode_file = data_path / "sub-pt1" / "ses-presurgery" /"ieeg" / "sub-pt1_ses-presurgery_space-fsaverage_electrodes.tsv"
    if electrode_file.is_file():
        contacts = pd.read_table(electrode_file)
        coords = contacts[["x", "y", "z"]].values

        # %%
        # Map contacts to atlas regions
        # For each electrode contact (x, y, z in MNI), find the
        # corresponding anatomical label in the chosen atlas.
        fetcher = AtlasFetcher()
        atlas = fetcher.fetch_atlas("aal")
        mapper = AtlasMapper("aal", atlas["vol"], atlas["hdr"], atlas["labels"])
        labels = [mapper.mni_to_region_name(c) for c in coords]
        print(list(zip(contacts["name"], labels))[:5])
    else:
        print("Electrode file not found; skipping electrode mapping.")
