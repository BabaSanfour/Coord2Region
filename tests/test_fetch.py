from coord2region import AtlasFetcher

"""
{
#"aal": self._fetch_atlas_aal,
"brodmann": self._fetch_atlas_brodmann,
"harvard-oxford": self._fetch_atlas_harvard_oxford,
"juelich": self._fetch_atlas_juelich,
"schaefer": self._fetch_atlas_schaefer,
"yeo": self._fetch_atlas_yeo,
# MNE-based atlases:
"aparc2009": self._fetch_atlas_aparc2009,
}

{
'vol': vol_data,
'hdr': hdr_matrix,
'labels': labels,
'description': desc,
'file': fname
}
"""
def test_fetch_all_atlases():
    atlases = ["harvard-oxford","brodmann", "juelich", "schaefer", "yeo", "aparc2009"]
    good_atlases = []
    bad_atlases = []
    for atlas in atlases:
        try:
            _fetch_atlas_helper(atlas)
            good_atlases.append(atlas)
        except Exception as e:
            print(f"Error fetching atlas {atlas}: {e}")
            bad_atlases.append(atlas)
    print(f"Good atlases: {good_atlases}")
    print(f"Bad atlases: {bad_atlases}")

def _fetch_atlas_helper(atlas_name):
    af = AtlasFetcher(data_dir="atlas_data")
    atlas = af.fetch_atlas(atlas_name)

    # assert for volume data
    assert 'vol' in atlas.keys() and atlas['vol'] is not None, f"Volume data is None for atlas {atlas_name}"
    assert 'hdr' in atlas.keys() and atlas['hdr'] is not None, f"Header data is None for atlas {atlas_name}"
    assert 'labels' in atlas.keys() and atlas['labels'] is not None, f"Labels data is None for atlas {atlas_name}"
    assert 'description' in atlas.keys() and atlas['description'] is not None, f"Description data is None for atlas {atlas_name}"
    # Right now is returning the description coming from nilearn, Hamza wants the name of the atlas and the version
    print('Warning! Right now is returning the description coming from nilearn, Hamza wants the name of the atlas and the version')
    assert 'file' in atlas.keys(), f"File name is not in atlas {atlas_name}"
    return atlas

def test_fetch_atlas_nolabels():
    af = AtlasFetcher(data_dir="atlas_data")
    atlas = af.fetch_atlas("yeo")
    print(atlas.keys())

if __name__ == "__main__":
    test_fetch_all_atlases()

