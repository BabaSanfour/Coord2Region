"""Coord2Region: A package for mapping brain coordinates to regions and studies.

This package provides tools to map MNI coordinates to brain regions using 
various atlases, fetch and manage atlases, and retrieve neuroimaging studies 
associated with specific coordinates.
"""
from .coord2region import (
    AtlasMapper,
    BatchAtlasMapper,
    MultiAtlasMapper,
)
from .fetching import AtlasFetcher
from .file_handler import AtlasFileHandler
from .coord2study import get_studies_for_coordinate

__all__ = [
    "AtlasMapper",
    "BatchAtlasMapper",
    "MultiAtlasMapper",
    "AtlasFetcher",
    "AtlasFileHandler",
    "get_studies_for_coordinate",
]
