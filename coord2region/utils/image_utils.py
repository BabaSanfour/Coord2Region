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
from PIL import Image, ImageDraw, ImageFont


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


def add_watermark(
    image_bytes: bytes,
    text: str = "AI approximation for illustrative purposes",
) -> bytes:
    """Overlay a semi-transparent watermark onto image bytes.

    Parameters
    ----------
    image_bytes : bytes
        Original image encoded as bytes.
    text : str, optional
        Watermark text to overlay. Defaults to
        ``"AI approximation for illustrative purposes"``.

    Returns
    -------
    bytes
        PNG-encoded image bytes with the watermark applied.
    """

    base = Image.open(BytesIO(image_bytes)).convert("RGBA")
    width, height = base.size

    # Create transparent overlay for the text
    overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Choose a font size that covers much of the image width
    font_size = max(12, int(width * 0.05))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    # If using a scalable font, adjust size so text fits within image
    if hasattr(font, "getbbox"):
        while True:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            if text_width <= width * 0.9 or font_size <= 10:
                break
            font_size -= 2
            try:
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()
                break
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    position = ((width - text_width) / 2, height - text_height - height * 0.05)

    draw.text(position, text, font=font, fill=(255, 255, 255, 128))
    watermarked = Image.alpha_composite(base, overlay)

    out = BytesIO()
    watermarked.convert("RGB").save(out, format="PNG")
    out.seek(0)
    return out.getvalue()
