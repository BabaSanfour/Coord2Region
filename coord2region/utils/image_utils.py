"""Utility functions for generating simple brain images.

This module currently exposes :func:`generate_mni152_image`, which creates a
static visualization of a spherical region overlaid on the MNI152 template
using Nilearn's plotting utilities. The resulting image is returned as PNG
bytes so it can be saved or embedded by callers without touching the
filesystem.
"""

from __future__ import annotations

from io import BytesIO
from typing import Sequence, Tuple

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


def build_side_by_side_panel(
    left_image: bytes,
    right_image: bytes,
    *,
    left_title: str = "AI-generated approximation",
    right_title: str = "Nilearn reference",
    background_color: Tuple[int, int, int] = (20, 20, 24),
    padding: int = 36,
) -> bytes:
    """Return a labelled side-by-side comparison panel.

    Parameters
    ----------
    left_image, right_image : bytes
        PNG-encoded images for the left and right panels respectively.
    left_title, right_title : str, optional
        Captions rendered above the corresponding image.
    background_color : tuple[int, int, int], optional
        RGB colour applied to the canvas background. Defaults to a dark grey.
    padding : int, optional
        Padding (in pixels) surrounding the images and text.
    """
    left = Image.open(BytesIO(left_image)).convert("RGB")
    right = Image.open(BytesIO(right_image)).convert("RGB")

    font_size = max(14, int(min(left.width, right.width) * 0.035))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    dummy = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(dummy)
    left_bbox = draw.textbbox((0, 0), left_title, font=font)
    right_bbox = draw.textbbox((0, 0), right_title, font=font)
    title_height = max(left_bbox[3] - left_bbox[1], right_bbox[3] - right_bbox[1])
    spacer = padding // 2

    total_width = left.width + right.width + padding * 3
    total_height = max(left.height, right.height) + title_height + padding * 3

    canvas = Image.new("RGB", (total_width, total_height), background_color)
    draw = ImageDraw.Draw(canvas)

    left_text_width = left_bbox[2] - left_bbox[0]
    right_text_width = right_bbox[2] - right_bbox[0]
    text_y = padding
    left_text_x = padding + max(0, (left.width - left_text_width) // 2)
    right_text_x = (
        padding * 2 + left.width + max(0, (right.width - right_text_width) // 2)
    )

    draw.text((left_text_x, text_y), left_title, fill=(235, 235, 245), font=font)
    draw.text((right_text_x, text_y), right_title, fill=(235, 235, 245), font=font)

    image_y = text_y + title_height + spacer
    canvas.paste(left, (padding, image_y))
    canvas.paste(right, (padding * 2 + left.width, image_y))

    out = BytesIO()
    canvas.save(out, format="PNG")
    out.seek(0)
    return out.getvalue()
