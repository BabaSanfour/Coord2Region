"""
Atlas mapping
=============

Use :class:`coord2region.coord2region.AtlasMapper` to convert an MNI coordinate
into a brain region label. The atlas is fetched on demand with
:class:`coord2region.fetching.AtlasFetcher`.
"""

# %%
# Fetch the atlas
from coord2region.fetching import AtlasFetcher
from coord2region.coord2region import AtlasMapper

fetcher = AtlasFetcher()
atlas_data = fetcher.fetch_atlas("harvard-oxford")

# %%
# Create the mapper and look up a coordinate
mapper = AtlasMapper(
    name="harvard-oxford",
    vol=atlas_data["vol"],
    hdr=atlas_data["hdr"],
    labels=atlas_data["labels"],
)

coord = [0, -52, 26]
label = mapper.mni_to_region_name(coord)
print(f"{coord} -> {label}")
