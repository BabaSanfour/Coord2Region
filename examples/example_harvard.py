"""
===================================
Example Usage
===================================

"""

#%%
from coord2region.fetching import AtlasFetcher
from coord2region.coord2region import AtlasMapper, BatchAtlasMapper, MultiAtlasMapper

#%%
# Fetch an atlas
# ---------------
#
# Fetch nilearn harvard-oxford

af = AtlasFetcher()

atlas_name = 'harvard-oxford'
vol = af.fetch_atlas(atlas_name)

#
# %% 
# Create a volumetric mapper
# ---------------------------
# 
# Create a volumetric mapper for the harvard-oxford atlas
#

vol_mapper = AtlasMapper(
    name=atlas_name,
    vol=vol['vol'],
    hdr=vol['hdr'],
    labels=vol['labels'],
)

#%%
print(vol_mapper.region_name_from_index(4204))

#%%
print(vol_mapper.region_index_from_name('Lat_Fis-post-rh'))

#%%
print(vol_mapper.list_all_regions())

#%%
print(vol_mapper.infer_hemisphere("Lat_Fis-post-rh"))
