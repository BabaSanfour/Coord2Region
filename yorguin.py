def get_vertinfo(subjects_dir,parc='aparc.a2009s'):
    subject='fstemplate'
    src = mne.read_source_spaces(os.path.join(subjects_dir, subject, 'bem', subject + '-ico-4-src.fif'),verbose=False)
    verts = [v["vertno"] for v in src]
    #colormap = colormap

    subject = 'fsaverage' #source space subject ('fsaverage') did not match stc.subject (fstemplate)
    vmapping=get_vert_mappings(subject,subjects_dir,src=src,parc=parc)
    labmap=invert_vmapping(vmapping)
    mni = mne.vertex_to_mni(verts[:2], [0,1], subject, subjects_dir=subjects_dir)
    mni = np.concatenate(mni,axis=0)#.shape
    return {'mni':mni,'vmap':vmapping,'labmap':labmap}

def get_vert_mappings(subject,subjects_dir,parc='aparc',src=None,verbose=False):
    if src is None:
        src = mne.read_source_spaces(os.path.join(subjects_dir, subject, 'bem', subject + '-ico-4-src.fif'),verbose=False)

    labels_cortex = mne.read_labels_from_annot(
        subject, parc=parc, subjects_dir=subjects_dir,verbose=False)

    vertno = [s['vertno'] for s in src]
    nvert = [len(vn) for vn in vertno]
    if verbose:
        print(labels_cortex)

        print('the src space contains {} spaces and {} points'.format(
                len(src), sum(nvert)))
        print('the cortex contains {} spaces and {} points'.format(
                len(src[:2]), sum(nvert[:2])))
        print('the volumes contains {} spaces and {} points'.format(
                len(src[2:]), sum(nvert[2:])))

    labels_aseg = mne.get_volume_labels_from_src(src, subject, subjects_dir)

    label_vertidx_cortex = list()
    label_name_cortex = list()

    for label in labels_cortex:
        if label.hemi == 'lh':
            this_vertno = np.intersect1d(vertno[0], label.vertices)
            vertidx = np.searchsorted(vertno[0], this_vertno)
        elif label.hemi == 'rh':
            this_vertno = np.intersect1d(vertno[1], label.vertices)
            vertidx = nvert[0] + np.searchsorted(vertno[1], this_vertno)

        label_vertidx_cortex.append(vertidx)
        label_name_cortex.append(label.name)


    nv_ROIs_cortex = [len(lab) for lab in label_vertidx_cortex]
    n_ROIs_cortex = len(label_vertidx_cortex)

    # label_vertidx_deep contains the vertices of deep structures,
    # v=label_vertidx_deep[0] is the Left-Amygdala and so on
    label_vertidx_deep = list()
    label_name_deep = list()
    all_deep = list()

    n_hemi = 2
    for s, label in enumerate(labels_aseg):
        n_deep = s + n_hemi
        print(n_deep)
        print(label)
        this_vertno = np.intersect1d(vertno[n_deep], label.vertices)
        vertidx = sum(nvert[:n_deep]) + np.searchsorted(
                vertno[n_deep], this_vertno)

        label_vertidx_deep.append(vertidx)
        label_name_deep.append(label.name)

    n_ROIs_deep = len(label_vertidx_deep)

    # TEST
    if n_ROIs_deep>0:
        all_deep = np.concatenate(label_vertidx_deep)

        assert len(all_deep) == sum(nvert[2:]), 'Something wrong!!!'
        assert np.sum(all_deep - np.arange(sum(nvert[:2]), sum(nvert))) == 0, 'Something wrong!!!'  # noqa

    n_ROIs = n_ROIs_cortex + n_ROIs_deep

    cortex_dict = {'cortex-'+label_name_cortex[i]:label_vertidx_cortex[i] for i in range(len(label_name_cortex))}
    if n_ROIs_deep>0:
        deep_dict = {'deep-'+label_name_deep[i]:label_vertidx_deep[i] for i in range(len(label_name_deep))}
    else:
        deep_dict = {}
    all_dict = {**cortex_dict,**deep_dict}
    return all_dict

def invert_vmapping(vmapping):
    labmap ={}
    for lab,verts in vmapping.items():
        for v in verts:
            labmap[v]=lab
    return labmap

# As you may notice, there is a special treatment when deep sources are involved.
# 3:59
# I have a very similar function to get_vertinfo which is called get_ROI_info, I believe it was an older version, perhaps less general, I dont remember, probably better to compare with an llm:
def get_ROI_info(src, subject, subjects_dir, parc='aparc', aseg=False):

    labels_cortex = mne.read_labels_from_annot(
        subject, parc=parc, subjects_dir=subjects_dir)
    # print(labels_cortex)
    vertno = [s['vertno'] for s in src]
    nvert = [len(vn) for vn in vertno]
    n_vertices = sum(nvert)
    ROI_mapping = np.zeros(n_vertices, dtype=int)

    print('the src space contains {} spaces and {} points'.format(
            len(src), sum(nvert)))
    print('the cortex contains {} spaces and {} points'.format(
            len(src[:2]), sum(nvert[:2])))
    print('the volumes contains {} spaces and {} points'.format(
            len(src[2:]), sum(nvert[2:])))

    if aseg:
        labels_aseg = mne.get_volume_labels_from_src(
            src, subject, subjects_dir)

    label_vertidx_cortex = list()
    label_name_cortex = list()

    for label in labels_cortex:
        if label.hemi == 'lh':
            this_vertno = np.intersect1d(vertno[0], label.vertices)
            vertidx = np.searchsorted(vertno[0], this_vertno)
        elif label.hemi == 'rh':
            this_vertno = np.intersect1d(vertno[1], label.vertices)
            vertidx = nvert[0] + np.searchsorted(vertno[1], this_vertno)

        label_vertidx_cortex.append(vertidx)
        label_name_cortex.append(label.name)

    nv_ROIs_cortex = [len(lab) for lab in label_vertidx_cortex]
    n_ROIs_cortex = len(label_vertidx_cortex)

    cortex_dict = {
        label_name_cortex[i]:label_vertidx_cortex[i] for i in range(len(label_name_cortex))}  # noqa
    # label_vertidx_deep contains the vertices of deep structures,
    # v=label_vertidx_deep[0] is the Left-Amygdala and so on
    if aseg:
        label_vertidx_deep = list()
        label_name_deep = list()
        all_deep = list()

        n_hemi = 2
        for s, label in enumerate(labels_aseg):
            n_deep = s + n_hemi
            # print(n_deep)
            # print(label)
            this_vertno = np.intersect1d(vertno[n_deep], label.vertices)
            vertidx = sum(nvert[:n_deep]) + np.searchsorted(
                    vertno[n_deep], this_vertno)

            label_vertidx_deep.append(vertidx)
            label_name_deep.append(label.name)

        n_ROIs_deep = len(label_vertidx_deep)

        # TEST
        all_deep = np.concatenate(label_vertidx_deep)

        assert len(all_deep) == sum(nvert[2:]), 'Something wrong!!!'
        assert np.sum(all_deep - np.arange(sum(nvert[:2]), sum(nvert))) == 0, 'Something wrong!!!'  # noqa

        n_ROIs = n_ROIs_cortex + n_ROIs_deep

        deep_dict = {
            label_name_deep[i]:label_vertidx_deep[i] for i in range(len(label_name_deep))}  # noqa
        all_dict = {**cortex_dict, **deep_dict}
    else:
        n_ROIs = n_ROIs_cortex
        all_dict = {**cortex_dict}

    for nr, roi in enumerate(all_dict):
        # print(roi)
        idx = all_dict[roi]
        ROI_mapping[idx] = nr

    return all_dict, ROI_mapping

def get_morph(subject,subjects_dir,bidspath,subtemplate='fsaverage',spacing='ico-4',aseg=True,only_check=False):
    if aseg:
        asegstr = '-aseg'
    else:
        asegstr=''

    fpath = os.path.join(bidspath,f"sub-{subject}",f"sub-{subject}_modality-meg_type-epo-{spacing}{asegstr}-fwd.fif")

    if only_check:
        return os.path.exists(fpath)

    sbj = subject
    #fwd = mne.read_forward_solution(fpath)

    fsaverage_fpath = os.path.join(subjects_dir, f'{subtemplate}/bem/{subtemplate}-ico-4-src.fif')
    fsaverage_src = mne.read_source_spaces(fsaverage_fpath)

    vertices_to = [s['vertno'] for s in fsaverage_src]


    bem_dir = os.path.join(subjects_dir, sbj, 'bem')
    print('bem path {}'.format(bem_dir))
    fwd_fpath = fpath
    assert os.path.exists(fwd_fpath)
    print('source path {}'.format(fwd_fpath))

    fwd = mne.read_forward_solution(fwd_fpath)
    src = fwd['src']
    surf_src = mne.source_space.SourceSpaces(fwd['src'][:2])

    n_cortex = (src[0]['nuse'] + src[1]['nuse'])

    morph_surf = mne.compute_source_morph(
            src=surf_src, subject_from=sbj, subject_to=subtemplate,
            spacing=vertices_to, subjects_dir=subjects_dir)
    # M = morph_surf.morph_mat has dim N_fs x N_sbj  => M*data
    # gives the data in fs_average space
    print(morph_surf.kind)
    print(morph_surf.morph_mat.shape)
    print((src[0]['nuse'] + src[1]['nuse']))

    return morph_surf,n_cortex

# PALS_B12_Brodmann
# aparc.a2009s
# aparc
# aparc_sub
# Which were the freesurfer .annot files under "fsaverage\label"
# vertex to mni:
{"mni": {
        "0": [
            -36.806278228759766,
            -18.292722702026367,
            64.4615249633789
        ],
}# label to vertices:
{    "vmap": {
        "cortex-bankssts_1-lh": [
            129,
            290,
            1232,
            1233,
            1234,
            1236,
            1674,
            1675,
            1676,
            1677
        ]
}# vertices to label
{    "labmap": {
        "129": "cortex-bankssts_1-lh",
        "290": "cortex-bankssts_1-lh",
        "1232": "cortex-bankssts_1-lh",
        "1233": "cortex-bankssts_1-lh",
        "1234": "cortex-bankssts_1-lh",
}# vertices to tal coordinates
{    "tal": {
        "0": [
            -35.88577840194702,
            -24.24612915802002,
            59.44336203365325
        ],
}
# Finally I have this script to see if regions of differents parcellations intersect

import json
import numpy as np
aparc=json.load(open("vertinfo_aparc.json"))
pals = json.load(open("vertinfo_PALS_B12_Brodmann.json"))
aparc_sub=json.load(open("vertinfo_aparc_sub.json"))

regions = {}

for atlas_dict,atlas in zip([aparc, pals, aparc_sub], ['aparc', 'pals', 'aparc_sub']):
    regions[atlas] = {}
    for info in atlas_dict.keys():
        regions[atlas][info] = atlas_dict[info]


a = 'cortex-postcentral_1-rh' + '@' + 'aparc_sub'
b = 'cortex-Brodmann.43-rh' + '@' + 'pals'

region_a=a.split('@')[0]
region_b=b.split('@')[0]
atlas_a=a.split('@')[1]
atlas_b=b.split('@')[1]

vmap_a = regions[atlas_a]['vmap'][region_a]
vmap_b = regions[atlas_b]['vmap'][region_b]

intersect = set.intersection(set(vmap_a),set(vmap_b))


# get centroid of regions

coords_a = np.array([regions[atlas_a]['mni'][str(x)] for x in vmap_a])
coords_b = np.array([regions[atlas_b]['mni'][str(x)] for x in vmap_b])

centroid_a = np.mean(coords_a, axis=0)
centroid_b = np.mean(coords_b, axis=0)

dist_comp=np.abs(centroid_a-centroid_b)
dist = np.linalg.norm(centroid_a - centroid_b)
