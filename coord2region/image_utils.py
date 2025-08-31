"""Utility functions for generating simple brain images.

This module currently exposes :func:`generate_mni152_image`, which creates a
static visualization of a spherical region overlaid on the MNI152 template
using Nilearn's plotting utilities. The resulting image is returned as PNG
bytes so it can be saved or embedded by callers without touching the
filesystem.
"""

from __future__ import annotations

from io import BytesIO
from typing import Sequence

import numpy as np
import nibabel as nib
from nilearn.datasets import load_mni152_template
from nilearn.plotting import plot_stat_map


def generate_mni152_image(
    coord: Sequence[float],
    radius: int = 6,
    cmap: str = "autumn",
) -> bytes:
    """Return a PNG image of a sphere drawn on the MNI152 template.

    Parameters
    ----------
    coord : sequence of float
        MNI coordinate (x, y, z) in millimetres.
    radius : int, optional
        Radius of the sphere in millimetres. Defaults to ``6``.
    cmap : str, optional
        Matplotlib colormap used for the overlay. Defaults to ``"autumn"``.

    Returns
    -------
    bytes
        PNG-encoded image bytes representing the sphere on the MNI152
        template.
    """

    template = load_mni152_template()
    data = np.zeros(template.shape, dtype=float)
    affine = template.affine

    # Convert the coordinate from mm space to voxel indices.
    voxel = nib.affines.apply_affine(np.linalg.inv(affine), coord)

    # Create a spherical mask around the coordinate.
    x, y, z = np.ogrid[: data.shape[0], : data.shape[1], : data.shape[2]]
    voxel_sizes = nib.affines.voxel_sizes(affine)
    radius_vox = radius / float(np.mean(voxel_sizes))
    mask = (
        (x - voxel[0]) ** 2 + (y - voxel[1]) ** 2 + (z - voxel[2]) ** 2
    ) <= radius_vox**2
    data[mask] = 1

    img = nib.Nifti1Image(data, affine)

    display = plot_stat_map(img, bg_img=template, cmap=cmap, display_mode="ortho")
    buffer = BytesIO()
    display.savefig(buffer, format="png", bbox_inches="tight")
    display.close()
    buffer.seek(0)
    return buffer.getvalue()
