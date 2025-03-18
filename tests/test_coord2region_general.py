import pytest
import numpy as np

from coord2region.fetching import AtlasFetcher
from coord2region.coord2region import (
    VolumetricAtlasMapper,
    BatchAtlasMapper,
    MultiAtlasMapper
)


PROPERTIES ={
"harvard-oxford":{
    'infer_hemisphere':[('Frontal Pole',None)],
    'region2index':[('Insular Cortex',2)],
    'allregions':49,
    },
"juelich":{
    'infer_hemisphere':[('GM Primary motor cortex BA4p', None)],
    'region2index':[('GM Amygdala_laterobasal group',2)],
    'allregions':63,
    },
"schaefer":{
    'infer_hemisphere':[('7Networks_LH_Vis_1','L'),('7Networks_RH_Default_PFCv_4','R')],
    'region2index':[('7Networks_LH_Vis_3',2)],
    'allregions':400,
    },
"yeo":{
    'infer_hemisphere':[('17Networks_9',None)],
    'region2index':[('17Networks_2',2)],
    'allregions':18,
    }
}

## We should add ground truth for the following tests
TEST_MNIS =[[-54., 36., -4.]]
TEST_VOXELS=[[30, 40, 50]]


def get_data(atlas_name):
    """
    This fixture downloads/loads the atlas once
    and returns the dict that includes 'vol', 'hdr', 'labels', etc.
    """
    af = AtlasFetcher(data_dir="coord2region_data")
    data = af.fetch_atlas(atlas_name)
    return data

def get_volumetric(atlas_name):
    """
    This fixture creates a VolumetricAtlasMapper for the atlas.
    """
    return VolumetricAtlasMapper(
        name=atlas_name,
        vol=get_data(atlas_name)["vol"],
        hdr=get_data(atlas_name)["hdr"],
        labels=get_data(atlas_name).get("labels", None)
    )


def test_all_atlases():
    atlases = ["harvard-oxford","juelich", "schaefer", "yeo"]
    good_atlases = []
    bad_atlases = []
    for atlas in atlases:
        try:
            output=get_data(atlas)
            #print(output)
            good_atlases.append(atlas)
            print(atlas)
            print('vol',output['vol'].shape)
            print('hdr',output['hdr'].shape)
            print('labels',len(output['labels']))
            print(output['labels'])

            volumetric = VolumetricAtlasMapper(
                name=atlas,
                vol=output["vol"],
                hdr=output["hdr"],
                labels=output.get("labels", None)
            )
            print(volumetric)

            print('infer_hemisphere')
            for region,answer in PROPERTIES[atlas]['infer_hemisphere']:
                hemi = volumetric.infer_hemisphere(region)
                assert hemi==answer,f"Error in infer hemisphere for atlas {atlas}. Expected {answer}, got {hemi}"

            print('label 2 index, region name 2 mni')

            for region,index in PROPERTIES[atlas]['region2index']:
                idx = volumetric.region_index_from_name(region)
                assert idx==index,f"Error in region2index for atlas {atlas}. Expected {index}, got {idx}"


                coords = volumetric.region_name_to_mni(region)
                assert coords.ndim == 2 and coords.shape[1] == 3, f"Expected Nx3 array, got {coords.shape}"


            print('index 2 label, region index 2 mni')

            for region,index in PROPERTIES[atlas]['region2index']:
                reg = volumetric.region_name_from_index(index)
                assert reg==region,f"Error in index2region for atlas {atlas}. Expected {region}, got {reg}"
                
                # e.g., if index=1 is a real region
                coords = volumetric.region_index_to_mni(index)
                assert coords.ndim == 2 and coords.shape[1] == 3, f"Expected Nx3 array, got {coords.shape}"

            print('all regions')
            regions = volumetric.list_all_regions()
            assert len(regions) == PROPERTIES[atlas]['allregions'],f"Error in all regions for atlas {atlas}. Expected {PROPERTIES[atlas]['allregions']}, got {len(regions)}"

            for test_mni in TEST_MNIS:
                print('mni to voxel')
                # TODO: we need ground truth for this
                voxel_idx = volumetric.mni_to_voxel(test_mni)
                assert len(voxel_idx) == 3, f"Expected 3D voxel index, got {voxel_idx}"
                for v in voxel_idx:
                    assert isinstance(v, int), "Voxel indices must be integers."

                print('mni to region index')
                # TODO: we need ground truth for this
                region_idx = volumetric.mni_to_region_index(test_mni)
                assert isinstance(region_idx, (int, str)), f"Expected int or 'Unknown', got {region_idx}"

                print('mni to region name')
                region_name = volumetric.mni_to_region_name(test_mni)
                assert isinstance(region_name, str), f"Expected string, got {type(region_name)}"

            for test_voxel in TEST_VOXELS:
                print('voxel to mni')
                coords = volumetric.voxel_to_mni(test_voxel)
                assert coords.shape == (3,), f"Expected shape (3,), got {coords.shape}"




        except Exception as e:
            print(f"Error fetching atlas {atlas}: {e}")
            bad_atlases.append(atlas)
    print(f"Good atlases: {good_atlases}")
    print(f"Bad atlases: {bad_atlases}")
    assert len(bad_atlases) == 0, f"Failed to fetch atlases: {bad_atlases}"


# def test_get_volumetric_all_atlases():
#     atlases = ["harvard-oxford","juelich", "schaefer", "yeo"]
#     good_atlases = []
#     bad_atlases = []
#     for atlas in atlases:
#         try:
#             output=get_volumetric(atlas)
#             print(output)
#             good_atlases.append(atlas)
#         except Exception as e:
#             print(f"Error fetching atlas {atlas}: {e}")
#             bad_atlases.append(atlas)



"""


def vectorized_mapper(atlas_name):
    #Create a BatchAtlasMapper for the Harvard-Oxford atlas.
    return BatchAtlasMapper(get_volumetric(atlas_name))


def test_batch_mni_to_region_name(vectorized_mapper, harvard_mapper):
    labels = harvard_mapper.list_all_regions()[:5]
    coords_for_tests = []
    for label in labels:
        arr = harvard_mapper.region_name_to_mni(label)
        if arr.shape[0] > 0:
            coords_for_tests.append(arr[0])  # pick first voxel as example

    if len(coords_for_tests) == 0:
        pytest.skip("No valid coords found for testing batch MNI->region")

    result = vectorized_mapper.batch_mni_to_region_name(coords_for_tests)
    assert len(result) == len(coords_for_tests)
    for r in result:
        assert isinstance(r, str)


def test_batch_region_name_from_index(vectorized_mapper):
    region_names = vectorized_mapper.batch_region_name_from_index([2, 3, 4])
    assert len(region_names) == 3


def test_batch_region_index_from_name(vectorized_mapper):
    some_names = ["Frontal Pole", "Precentral Gyrus", "Unknown Region"]
    region_indices = vectorized_mapper.batch_region_index_from_name(some_names)
    assert len(region_indices) == len(some_names)


def test_multiatlas_api():
    # Test the high-level MultiAtlasMapper class
    c2r = MultiAtlasMapper(
        data_dir="coord2region_data", 
        atlases={"harvard-oxford": {}}
    )
    coords = [[-54., 36., -4.], [10., 20., 30.]]
    result_dict = c2r.batch_mni_to_region_names(coords)
    assert "harvard-oxford" in result_dict
    assert len(result_dict["harvard-oxford"]) == 2

    # region -> coords
    # example call:
    # c2r.batch_region_name_to_mni(["Frontal Pole", "Precentral Gyrus"])
    # etc.

"""

if __name__ == "__main__":
    test_all_atlases()
    print('done')