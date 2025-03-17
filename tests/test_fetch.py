from coord2region import AtlasFetcher


def test_fetch_atlas():
    af = AtlasFetcher(data_dir="atlas_data")
    atlas = af.fetch_atlas("yeo")
    print(atlas.keys())

if __name__ == "__main__":
    test_fetch_atlas()

