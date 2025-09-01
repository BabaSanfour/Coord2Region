"""Utilities for Coord2Region: file I/O, image generation, and data-directory helpers.

Main utility categories:
- label fetching and output packing
- file saving to CSV, PDF, or batch folders
- data directory resolution
- MNI152 image generation and watermarking
"""

from .utils import fetch_labels, pack_vol_output, pack_surf_output
from .paths import resolve_data_dir
from .file_handler import (
    AtlasFileHandler,
    save_as_csv,
    save_as_pdf,
    save_batch_folder,
)
from .image_utils import generate_mni152_image, add_watermark

__all__ = [
    "fetch_labels",
    "pack_vol_output",
    "pack_surf_output",
    "resolve_data_dir",
    "AtlasFileHandler",
    "save_as_csv",
    "save_as_pdf",
    "save_batch_folder",
    "generate_mni152_image",
    "add_watermark",
]
