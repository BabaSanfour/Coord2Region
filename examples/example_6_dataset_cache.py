"""Demonstrate dataset caching with :func:`prepare_datasets`.

Running this script twice shows that the second invocation loads the
previously merged dataset from disk instead of downloading it again.
"""

import os
from coord2region.coord2study import prepare_datasets

# Use a custom data directory; the deduplicated dataset will be stored in
# ``<data_dir>/cached_data`` alongside any downloaded atlases.
home = os.path.expanduser("~")
data_dir = os.path.join(home, "coord2region_example")

print(f"Using data directory: {data_dir}")

# First call will download/merge if the cache file is missing
merged = prepare_datasets(data_dir, sources=["nidm_pain"])  # limit to small dataset for speed
print(f"Merged dataset contains {len(merged.ids)} studies")

# Second call immediately reuses the cache
merged_again = prepare_datasets(data_dir, sources=["nidm_pain"])
print(f"Loaded cached dataset with {len(merged_again.ids)} studies")
