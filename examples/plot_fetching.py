"""
Fetching atlases
================

Showcase the :class:`coord2region.fetching.AtlasFetcher` by listing a few
available atlases and downloading one of them.
"""

# %%
# Create the fetcher and list atlases
from coord2region.fetching import AtlasFetcher

fetcher = AtlasFetcher()
print("Available atlases:", fetcher.list_available_atlases()[:5])

# %%
# Download the AAL atlas and inspect its labels
atlas_data = fetcher.fetch_atlas("aal")
labels = atlas_data.get("labels", [])
print(f"Fetched AAL atlas with {len(labels)} labels")
