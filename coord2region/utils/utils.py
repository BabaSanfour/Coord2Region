import os
import numpy as np

def fetch_labels(labels):
    """
    Process the labels input.
    - If a list is provided, return it.
    - If a filename is provided, raise NotImplementedError.
    """
    if isinstance(labels, str):
        raise NotImplementedError("Reading labels from file is not yet implemented.")
    elif isinstance(labels, list):
        return labels
    else:
        raise ValueError(f"Invalid labels type: {type(labels)}")


def pack_vol_output(file):
    """
    Load an atlas file (NIfTI, NPZ, or Nifti1Image) and package the output.

    Returns a dictionary with keys: 'vol' and 'hdr'.
    """
    if isinstance(file, str):
        path = os.path.abspath(file)
        _, ext = os.path.splitext(file)
        ext = ext.lower()

        if ext in ['.nii', '.gz', '.nii.gz']:
            import nibabel as nib
            img = nib.load(file)
            vol_data = img.get_fdata(dtype=np.float32)
            hdr_matrix = img.affine
            return {
                'vol': vol_data,
                'hdr': hdr_matrix,
            }
 
        elif ext == '.npz':
            arch = np.load(path, allow_pickle=True)
            vol_data = arch['vol']
            hdr_matrix = arch['hdr']
            return {
                'vol': vol_data,
                'hdr': hdr_matrix,
            }
        else:
            raise ValueError(f"Unrecognized file format '{ext}' for path: {path}")
    else:
        from nibabel.nifti1 import Nifti1Image
        if isinstance(file, Nifti1Image):
            vol_data = file.get_fdata(dtype=np.float32)
            hdr_matrix = file.affine
            return {
                'vol': vol_data,
                'hdr': hdr_matrix,
            }
        else:
            raise ValueError("Unsupported type for pack_vol_output")


def pack_surf_output(atlas_name, fetcher, subject: str = 'fsaverage', subjects_dir: str = None, **kwargs):
    """
    Load a surface-based atlas using MNE (from FreeSurfer annotation files).

    Returns a dictionary with keys: 'vol', 'hdr', 'labels', and 'indexes'.
    """
    # Determine subjects_dir: use provided or from MNE config
    import mne
    if subjects_dir is None:
        subjects_dir = mne.get_config('SUBJECTS_DIR', None)
        if subjects_dir is None:
            import os
            subjects_dir = os.path.join(mne.datasets.sample.data_path(), "subjects")
    if fetcher is None:
        try:
            labels = mne.read_labels_from_annot(subject, atlas_name, subjects_dir=subjects_dir, **kwargs)
        except Exception as e:
            mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir, verbose=True)
            labels = mne.read_labels_from_annot(subject, atlas_name, subjects_dir=subjects_dir, **kwargs)
    else:
        try:
            labels = fetcher(subject=subject, subjects_dir=subjects_dir, **kwargs)
        except Exception as e:
            fetcher(subjects_dir=subjects_dir, **kwargs)
            labels = mne.read_labels_from_annot(subject, atlas_name, subjects_dir=subjects_dir, **kwargs)
    
    src = mne.setup_source_space(subject, spacing='oct6', subjects_dir=subjects_dir, add_dist=False)
    lh_vert = src[0]['vertno']  # Left hemisphere vertices
    rh_vert = src[1]['vertno']  # Right hemisphere vertices

    # Map label names to indices in the vertex arrays.
    cortex_dict_lh = {
        label.name: np.nonzero(np.in1d(lh_vert, label.vertices))[0]
        for label in labels if label.hemi == 'lh'
    }
    cortex_dict_rh = {
        label.name: np.nonzero(np.in1d(rh_vert, label.vertices))[0]
        for label in labels if label.hemi == 'rh'
    }

    labmap_lh = {}
    for lab, indices in cortex_dict_lh.items():
        for idx in indices:
            labmap_lh[idx] = lab
    labmap_rh = {}
    for lab, indices in cortex_dict_rh.items():
        for idx in indices:
            labmap_rh[idx] = lab

    index_lh = np.sort(np.array(list(labmap_lh.keys())))
    labels_lh = np.array([labmap_lh[i] for i in index_lh])
    vmap_lh = lh_vert[index_lh]

    index_rh = np.sort(np.array(list(labmap_rh.keys())))
    labels_rh = np.array([labmap_rh[i] for i in index_rh])
    vmap_rh = rh_vert[index_rh]

    labels_combined = np.concatenate([labels_lh, labels_rh])
    indexes_combined = np.concatenate([vmap_lh, vmap_rh])

    return {
        'vol': [lh_vert, rh_vert],
        'hdr': None,
        'labels': labels_combined,
        'indexes': indexes_combined,
    }
