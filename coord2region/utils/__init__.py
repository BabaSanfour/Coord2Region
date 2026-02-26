"""Coord2Region utilities for file I/O, images, and working-directory tools."""

from .file_handler import (
    AtlasFileHandler,
    save_as_csv,
    save_as_pdf,
    save_batch_folder,
)
from .image_utils import add_watermark, generate_mni152_image
from .paths import ensure_mne_data_directory, resolve_working_directory
from .utils import fetch_labels, pack_surf_output, pack_vol_output

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
