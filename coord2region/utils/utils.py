"""Utility functions for handling atlas files and labels.

This module provides functions to parse label files, load atlas volumes,
and manage surface-based atlases using FreeSurfer annotations.
"""

import os
from pathlib import Path

import numpy as np


def fetch_labels(labels):
    """Parse a labels input.

    A list is returned as-is. If ``labels`` is a string, it is treated as the
    path to an XML file and parsed for entries within a
    ``<data><label><name>...</name></label></data>`` structure.

    Parameters
    ----------
    labels : list of str or str
        A list of label names or a path to an XML file containing labels.

    Returns
    -------
    list of str
        Parsed label names.

    Raises
    ------
    ValueError
        If the XML file is invalid, cannot be parsed, contains no labels, or
        if ``labels`` is neither a list nor a string.

    Examples
    --------
    >>> fetch_labels(['A', 'B'])
    ['A', 'B']
    >>> fetch_labels('atlas.xml')  # doctest: +SKIP
    ['Region1', 'Region2']
    """
    if isinstance(labels, str):
        import xml.etree.ElementTree as ET

        try:
            tree = ET.parse(labels)
            root = tree.getroot()
            data = root.find("data")
            if data is None:
                raise ValueError("Invalid XML file: missing 'data' element.")
            label_list = []
            for label in data.findall("label"):
                name_elem = label.find("name")
                if name_elem is not None:
                    label_list.append(name_elem.text)
            if not label_list:
                raise ValueError("No labels found in the XML file.")
            return label_list
        except Exception as e:
            raise ValueError(f"Error processing XML file {labels}: {e}")
    elif isinstance(labels, list):
        return labels
    else:
        raise ValueError(f"Invalid labels type: {type(labels)}")


def pack_vol_output(file):
    """Load an atlas file and return volume data and header.

    Parameters
    ----------
    file : str or Nifti1Image
        Path to a NIfTI/NPZ file or a loaded
        :class:`~nibabel.nifti1.Nifti1Image`.

    Returns
    -------
    dict
        Dictionary with ``'vol'`` and ``'hdr'`` entries.

    Raises
    ------
    ValueError
        If the file format or object type is not supported.

    Examples
    --------
    >>> pack_vol_output('atlas.nii.gz')  # doctest: +SKIP
    {'vol': array(...), 'hdr': array(...)}
    """
    if isinstance(file, str):
        path = os.path.abspath(file)
        _, ext = os.path.splitext(file)
        ext = ext.lower()

        if ext in [".nii", ".gz", ".nii.gz"]:
            import nibabel as nib

            img = nib.load(file)
            vol_data = img.get_fdata(dtype=np.float32)
            hdr_matrix = img.affine
            return {
                "vol": vol_data,
                "hdr": hdr_matrix,
            }

        elif ext == ".npz":
            arch = np.load(path, allow_pickle=True)
            vol_data = arch["vol"]
            hdr_matrix = arch["hdr"]
            return {
                "vol": vol_data,
                "hdr": hdr_matrix,
            }
        else:
            raise ValueError(f"Unrecognized file format '{ext}' for path: {path}")
    else:
        from nibabel.nifti1 import Nifti1Image

        if isinstance(file, Nifti1Image):
            vol_data = file.get_fdata(dtype=np.float32)
            hdr_matrix = file.affine
            return {
                "vol": vol_data,
                "hdr": hdr_matrix,
            }
        else:
            raise ValueError("Unsupported type for pack_vol_output")


def pack_surf_output(
    atlas_name,
    fetcher,
    subject: str = "fsaverage",
    subjects_dir: str | os.PathLike | None = None,
    **kwargs,
):
    """Load a surface-based atlas using FreeSurfer annotations.

    Parameters
    ----------
    atlas_name : str
        Name of the atlas (e.g., ``'aparc'``).
    fetcher : callable or None
        Function used to download the atlas if necessary.
    subject : str, optional
        Subject identifier, by default ``'fsaverage'``.
    subjects_dir : path-like or None, optional
        FreeSurfer subjects directory, by default ``None``.
    **kwargs
        Additional keyword arguments passed to the fetcher.

    Returns
    -------
    dict
        Dictionary with ``'vol'``, ``'hdr'``, ``'labels'``, ``'indexes'`` and
        ``'regions'`` keys where ``'regions'`` maps region names to vertex
        indices.

    Raises
    ------
    ValueError
        If the atlas cannot be located or fetched.

    Examples
    --------
    >>> pack_surf_output('aparc', None)  # doctest: +SKIP
    {'vol': [...], 'hdr': None, 'labels': array([...]), 'indexes': array([...]),
     'regions': {'Region': array([...])}}
    """
    # Determine subjects_dir: use provided or from MNE config
    import mne

    subjects_dir_path = Path(subjects_dir) if subjects_dir is not None else None
    subjects_dir_arg = str(subjects_dir_path) if subjects_dir_path is not None else None

    def _download_fsaverage_annotations() -> None:
        """Ensure fsaverage annotations are available locally."""
        if str(subject).lower() != "fsaverage":
            return

        resolved_subjects_dir = mne.utils.get_subjects_dir(
            subjects_dir_arg, raise_error=False
        )
        resolved_subjects_path = (
            Path(resolved_subjects_dir) if resolved_subjects_dir is not None else None
        )
        if resolved_subjects_path is not None:
            annot_dir = resolved_subjects_path / subject / "label"
            if annot_dir.exists():
                return

        fetch_kwargs = {}
        if subjects_dir_arg is not None:
            fetch_kwargs["subjects_dir"] = subjects_dir_arg
        try:
            mne.datasets.fetch_fsaverage(verbose=False, **fetch_kwargs)
        except TypeError:  # pragma: no cover - older MNE versions
            if subjects_dir_arg is not None:
                mne.datasets.fetch_fsaverage(subjects_dir_arg)
            else:
                mne.datasets.fetch_fsaverage()

    def _read_labels_from_annot():
        return mne.read_labels_from_annot(
            subject,
            atlas_name,
            subjects_dir=subjects_dir_arg,
            **kwargs,
        )

    if fetcher is None:
        _download_fsaverage_annotations()
        try:
            labels = _read_labels_from_annot()
        except (OSError, FileNotFoundError):
            _download_fsaverage_annotations()
            labels = _read_labels_from_annot()
    else:
        try:
            labels = fetcher(subject=subject, subjects_dir=subjects_dir_arg, **kwargs)
        except Exception:
            _download_fsaverage_annotations()
            try:
                fetcher(subject=None, subjects_dir=subjects_dir_arg, **kwargs)
            except Exception:
                pass
            labels = _read_labels_from_annot()

    src = mne.setup_source_space(
        subject,
        spacing="oct6",
        subjects_dir=subjects_dir_arg,
        add_dist=False,
    )
    lh_vert = src[0]["vertno"]  # Left hemisphere vertices
    rh_vert = src[1]["vertno"]  # Right hemisphere vertices

    # Map label names to indices in the vertex arrays.
    from collections import defaultdict

    region_vertices_lh = defaultdict(list)
    region_vertices_rh = defaultdict(list)
    labmap_lh = {}
    labmap_rh = {}

    for label in labels:
        if label.hemi == "lh":
            match = np.nonzero(np.isin(lh_vert, label.vertices))[0]
            verts = lh_vert[match]
            region_vertices_lh[label.name].extend(verts.tolist())
            for idx in match:
                labmap_lh[idx] = label.name
        elif label.hemi == "rh":
            match = np.nonzero(np.isin(rh_vert, label.vertices))[0]
            verts = rh_vert[match]
            region_vertices_rh[label.name].extend(verts.tolist())
            for idx in match:
                labmap_rh[idx] = label.name

    region_vertices_lh = {
        k: np.array(v, dtype=int) for k, v in region_vertices_lh.items()
    }
    region_vertices_rh = {
        k: np.array(v, dtype=int) for k, v in region_vertices_rh.items()
    }
    region_vertices = {**region_vertices_lh, **region_vertices_rh}

    indexes_lh = np.sort(np.array(list(labmap_lh.keys())))
    labels_lh = np.array([labmap_lh[i] for i in indexes_lh])
    vmap_lh = lh_vert[indexes_lh]

    indexes_rh = np.sort(np.array(list(labmap_rh.keys())))
    labels_rh = np.array([labmap_rh[i] for i in indexes_rh])
    vmap_rh = rh_vert[indexes_rh]

    labels_combined = np.concatenate([labels_lh, labels_rh])
    indexes_combined = np.concatenate([vmap_lh, vmap_rh])

    return {
        "vol": [lh_vert, rh_vert],
        "hdr": None,
        "labels": labels_combined,
        "indexes": indexes_combined,
        "regions": region_vertices,
    }
