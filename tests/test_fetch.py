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
def test_fetch_nilearn_atlases():
    atlases = ["harvard-oxford"]#, "aparc2009"] #"brodmann", "aal", "yeo","juelich", "schaefer"
    good_atlases = []
    bad_atlases = []
    for atlas in atlases:
        try:
            output=_fetch_atlas_helper(atlas)
            print(output)
            good_atlases.append(atlas)
            print('vol',output['vol'].shape)
            print('hdr',output['hdr'].shape)
            print('labels',len(output['labels']))

        except Exception as e:
            print(f"Error fetching atlas {atlas}: {e}")
            bad_atlases.append(atlas)
    print(f"Good atlases: {good_atlases}")
    print(f"Bad atlases: {bad_atlases}")
    # assert len(bad_atlases) == 0, f"Failed to fetch atlases: {bad_atlases}"

def not_ready_test_fetch_mne_atlases():
    af = AtlasFetcher(data_dir="mne_data")
    atlas = af.fetch_atlas('aparc.a2009s')
    print(atlas)

def _fetch_atlas_helper(atlas_name):
    af = AtlasFetcher(data_dir="atlas_data")
    atlas = af.fetch_atlas(atlas_name)
    return atlas

if __name__ == "__main__":
    #not_ready_test_fetch_mne_atlases()
    test_fetch_nilearn_atlases()
    

