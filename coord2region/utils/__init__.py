"""Coord2Region utilities for file I/O, images, and working-directory tools."""

from .utils import fetch_labels, pack_vol_output, pack_surf_output
from .paths import resolve_working_directory, ensure_mne_data_directory
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
    "resolve_working_directory",
    "ensure_mne_data_directory",
    "AtlasFileHandler",
    "save_as_csv",
    "save_as_pdf",
    "save_batch_folder",
    "generate_mni152_image",
    "add_watermark",
]
