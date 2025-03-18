import os
import json
import numpy as np
import mne

# =============================================================================
# Helper Functions for Vertex / ROI Mapping
# =============================================================================

def invert_vmapping(vmapping):
    """
    Invert a label-to-vertices mapping in order to generate a vertex-to-label mapping.
    
    Parameters:
      vmapping (dict): Mapping from label names to lists of vertex indices
      
    Returns:
      labmap (dict): Mapping from vertex index to label name
    """
    labmap = {}
    for lab, verts in vmapping.items():
        for v in verts:
            labmap[v] = lab
    return labmap

# =============================================================================
# Functions to obtain vertex mappings information
# =============================================================================

def g(subject, subjects_dir, parc='aparc', src=None, verbose=False):
    """
    Generate a mapping of labels (both cortex and deep structures) to vertex indices.
    
    Parameters:
      subject (str): Subject name whose source-space is used.
      subjects_dir (str): Directory containing subject folders.
      parc (str): Parcellation to use (default 'aparc').
      src (list): Optional pre-read source space (if not provided, it will be read).
      verbose (bool): If True, prints debug information.
    
    Returns:
      all_dict (dict): Mapping from label names (including cortex and deep) to vertex indices.
    """
    # Read source space if not provided
    if src is None:
        fpath = os.path.join(subjects_dir, subject, 'bem', subject + '-ico-4-src.fif')
        src = mne.read_source_spaces(fpath, verbose=False)

    # Read cortical labels from the annotation file
    labels_cortex = mne.read_labels_from_annot(subject, parc=parc, subjects_dir=subjects_dir, verbose=False)
    
    # Total vertices and per-space vertices
    vertno = [s['vertno'] for s in src]
    nvert = [len(vn) for vn in vertno]
    
    if verbose:
        print(labels_cortex)
        print('The src space contains {} spaces and {} points'.format(len(src), sum(nvert)))
        print('Cortical parts contain {} spaces and {} points'.format(len(src[:2]), sum(nvert[:2])))
        print('Volume parts contain {} spaces and {} points'.format(len(src[2:]), sum(nvert[2:])))
    
    # Process deep structures (aseg)
    labels_aseg = mne.get_volume_labels_from_src(src, subject, subjects_dir)
    
    # Create mappings for cortical labels
    label_vertidx_cortex = []
    label_name_cortex = []
    for label in labels_cortex:
        if label.hemi == 'lh':
            this_vertno = np.intersect1d(vertno[0], label.vertices)
            vertidx = np.searchsorted(vertno[0], this_vertno)
        elif label.hemi == 'rh':
            this_vertno = np.intersect1d(vertno[1], label.vertices)
            vertidx = nvert[0] + np.searchsorted(vertno[1], this_vertno)
        label_vertidx_cortex.append(vertidx)
        label_name_cortex.append(label.name)
    
    # Process deep structures 
    label_vertidx_deep = []
    label_name_deep = []
    n_hemi = 2
    for s, label in enumerate(labels_aseg):
        n_deep = s + n_hemi
        if verbose:
            print("Processing deep structure index:", n_deep)
            print(label)
        this_vertno = np.intersect1d(vertno[n_deep], label.vertices)
        vertidx = sum(nvert[:n_deep]) + np.searchsorted(vertno[n_deep], this_vertno)
        label_vertidx_deep.append(vertidx)
        label_name_deep.append(label.name)
    
    # Test deep structures mapping
    if len(label_vertidx_deep) > 0:
        all_deep = np.concatenate(label_vertidx_deep)
        assert len(all_deep) == sum(nvert[2:]), 'Something wrong with deep mapping.'
        assert np.sum(all_deep - np.arange(sum(nvert[:2]), sum(nvert))) == 0, 'Mismatch in vertex indices for deep structures.'
    
    cortex_dict = {'cortex-' + label_name_cortex[i]: label_vertidx_cortex[i] 
                   for i in range(len(label_name_cortex))}
    deep_dict = {'deep-' + label_name_deep[i]: label_vertidx_deep[i] 
                 for i in range(len(label_name_deep))} if label_vertidx_deep else {}
    
    all_dict = {**cortex_dict, **deep_dict}
    return all_dict

def get_vertinfo(subjects_dir, parc='aparc.a2009s'):
    """
    Read source spaces and compute vertex-to-MNI coordinates along with label mappings.
    
    Parameters:
      subjects_dir (str): Directory with subject folders.
      parc (str): Parcellation to use (default: aparc.a2009s)
    
    Returns:
      info (dict): A dictionary containing:
         'mni' : MNI coordinates for vertices,
         'vmap': Mapping from labels to vertices,
         'labmap': Inverted mapping (vertex-to-label).
    """
    subject = 'fstemplate'
    src_path = os.path.join(subjects_dir, subject, 'bem', subject + '-ico-4-src.fif')
    src = mne.read_source_spaces(src_path, verbose=False)
    
    # Retrieve vertex numbers for all source spaces
    verts = [v["vertno"] for v in src]
    
    # Switch to fsaverage for mne.vertex_to_mni conversion 
    subject = 'fsaverage'
    vmapping = get_vert_mappings(subject, subjects_dir, src=src, parc=parc)
    labmap = invert_vmapping(vmapping)
    
    # Convert vertices to MNI coordinates for the cortical parts (spaces 0 and 1)
    mni = mne.vertex_to_mni(verts[:2], [0, 1], subject, subjects_dir=subjects_dir)
    mni = np.concatenate(mni, axis=0)
    
    return {'mni': mni, 'vmap': vmapping, 'labmap': labmap}

# =============================================================================
# ROI Information Function
# =============================================================================

def get_ROI_info(src, subject, subjects_dir, parc='aparc', aseg=False):
    """
    Obtain Region of Interest (ROI) info including label mapping and ROI mapping.
    
    Parameters:
      src (list): Source-space structure.
      subject (str): Subject name.
      subjects_dir (str): Subjects directory.
      parc (str): Parcellation (default 'aparc').
      aseg (bool): Include deep (aseg) structures if True.
    
    Returns:
      all_dict (dict): Mapping from ROI names to list of vertex indices.
      ROI_mapping (np.ndarray): Array where each vertex is assigned a region number.
    """
    # Read cortical labels from the annotation file
    labels_cortex = mne.read_labels_from_annot(subject, parc=parc, subjects_dir=subjects_dir)
    vertno = [s['vertno'] for s in src]
    nvert = [len(vn) for vn in vertno]
    n_vertices = sum(nvert)
    ROI_mapping = np.zeros(n_vertices, dtype=int)
    
    print('The src space contains {} spaces and {} points'.format(len(src), sum(nvert)))
    print('Cortex: {} spaces and {} points'.format(len(src[:2]), sum(nvert[:2])))
    print('Volumes: {} spaces and {} points'.format(len(src[2:]), sum(nvert[2:])))
    
    label_vertidx_cortex = []
    label_name_cortex = []
    for label in labels_cortex:
        if label.hemi == 'lh':
            this_vertno = np.intersect1d(vertno[0], label.vertices)
            vertidx = np.searchsorted(vertno[0], this_vertno)
        elif label.hemi == 'rh':
            this_vertno = np.intersect1d(vertno[1], label.vertices)
            vertidx = nvert[0] + np.searchsorted(vertno[1], this_vertno)
        label_vertidx_cortex.append(vertidx)
        label_name_cortex.append(label.name)
    
    cortex_dict = {label_name_cortex[i]: label_vertidx_cortex[i] for i in range(len(label_name_cortex))}
    
    # If aseg flag is set, process deep structures
    if aseg:
        labels_aseg = mne.get_volume_labels_from_src(src, subject, subjects_dir)
        label_vertidx_deep = []
        label_name_deep = []
        
        n_hemi = 2
        for s, label in enumerate(labels_aseg):
            n_deep = s + n_hemi
            this_vertno = np.intersect1d(vertno[n_deep], label.vertices)
            vertidx = sum(nvert[:n_deep]) + np.searchsorted(vertno[n_deep], this_vertno)
            label_vertidx_deep.append(vertidx)
            label_name_deep.append(label.name)
        
        # Verify deep structure vertices align with the volume vertices
        all_deep = np.concatenate(label_vertidx_deep)
        assert len(all_deep) == sum(nvert[2:]), 'Deep ROI mapping error.'
        assert np.sum(all_deep - np.arange(sum(nvert[:2]), sum(nvert))) == 0, 'Deep ROI vertex indices error.'

        deep_dict = {label_name_deep[i]: label_vertidx_deep[i] for i in range(len(label_name_deep))}
        all_dict = {**cortex_dict, **deep_dict}
    else:
        all_dict = cortex_dict

    # Create an ROI mapping array
    for nr, roi in enumerate(all_dict):
        idx = all_dict[roi]
        ROI_mapping[idx] = nr

    return all_dict, ROI_mapping

# =============================================================================
# Morph Information Function
# =============================================================================

def get_morph(subject, subjects_dir, bidspath, subtemplate='fsaverage', spacing='ico-4', aseg=True, only_check=False):
    """
    Compute a source morph between subject and template (e.g., fsaverage).
    
    Parameters:
      subject (str): Subject identifier.
      subjects_dir (str): Directory containing subject folders.
      bidspath (str): Base path for BIDS structure.
      subtemplate (str): Template subject for morphing (default 'fsaverage').
      spacing (str): Spacing for the source space (default 'ico-4').
      aseg (bool): If True, include aseg information in filename.
      only_check (bool): If True, only check for file existence.
    
    Returns:
      If only_check is False:
          morph_surf: Morphing object computed by MNE.
          n_cortex: Number of cortical vertices used in source morph.
      Else:
          Boolean indicating whether the forward solution file exists.
    """
    asegstr = '-aseg' if aseg else ''
    fpath = os.path.join(bidspath, f"sub-{subject}", f"sub-{subject}_modality-meg_type-epo-{spacing}{asegstr}-fwd.fif")
    
    if only_check:
        return os.path.exists(fpath)
    
    sbj = subject

    # Read source space for template subject
    fsaverage_fpath = os.path.join(subjects_dir, f'{subtemplate}', 'bem', f'{subtemplate}-ico-4-src.fif')
    fsaverage_src = mne.read_source_spaces(fsaverage_fpath)
    vertices_to = [s['vertno'] for s in fsaverage_src]
    
    # Check and load the subject forward solution
    bem_dir = os.path.join(subjects_dir, sbj, 'bem')
    print('bem path {}'.format(bem_dir))
    fwd_fpath = fpath
    assert os.path.exists(fwd_fpath), "Forward solution file does not exist."
    print('source path {}'.format(fwd_fpath))
    
    fwd = mne.read_forward_solution(fwd_fpath)
    src = fwd['src']
    surf_src = mne.source_space.SourceSpaces(fwd['src'][:2])
    n_cortex = (src[0]['nuse'] + src[1]['nuse'])
    
    # Compute the morph from the subject to template
    morph_surf = mne.compute_source_morph(src=surf_src, subject_from=sbj, subject_to=subtemplate,
                                            spacing=vertices_to, subjects_dir=subjects_dir)
    print("Morph kind:", morph_surf.kind)
    print("Morph matrix shape:", morph_surf.morph_mat.shape)
    print("Number of cortical vertices:", n_cortex)
    
    return morph_surf, n_cortex

# =============================================================================
# Example Script to Compare Regions from Different Atlases
# =============================================================================

if __name__ == "__main__":
    # Load atlas vertex info from JSON files
    aparc = json.load(open("vertinfo_aparc.json"))
    pals = json.load(open("vertinfo_PALS_B12_Brodmann.json"))
    aparc_sub = json.load(open("vertinfo_aparc_sub.json"))
    
    # Organize regions per atlas
    regions = {}
    for atlas_dict, atlas in zip([aparc, pals, aparc_sub], ['aparc', 'pals', 'aparc_sub']):
        regions[atlas] = {info: atlas_dict[info] for info in atlas_dict.keys()}
    
    # Define two regions from different atlases 
    a = 'cortex-postcentral_1-rh' + '@' + 'aparc_sub'
    b = 'cortex-Brodmann.43-rh' + '@' + 'pals'
    
    # Split region and atlas names
    region_a = a.split('@')[0]
    atlas_a = a.split('@')[1]
    region_b = b.split('@')[0]
    atlas_b = b.split('@')[1]
    
    # Get vertex mappings for each region
    vmap_a = regions[atlas_a]['vmap'][region_a]
    vmap_b = regions[atlas_b]['vmap'][region_b]
    
    # Find intersection of vertex indices between the two regions
    intersect = set(vmap_a).intersection(set(vmap_b))
    
    # Compute centroids for each region based on their MNI coordinates
    coords_a = np.array([regions[atlas_a]['mni'][str(x)] for x in vmap_a])
    coords_b = np.array([regions[atlas_b]['mni'][str(x)] for x in vmap_b])
    
    centroid_a = np.mean(coords_a, axis=0)
    centroid_b = np.mean(coords_b, axis=0)
    
    dist_comp = np.abs(centroid_a - centroid_b)
    dist = np.linalg.norm(centroid_a - centroid_b)
    
    print("Intersection vertices:", intersect)
    print("Centroid A:", centroid_a)
    print("Centroid B:", centroid_b)
    print("Distance components:", dist_comp)
    print("Euclidean distance between centroids:", dist)
