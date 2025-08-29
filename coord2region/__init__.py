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

# Import coord2study functions
from .coord2study import (
    # Dataset handling
    fetch_datasets,
    load_deduplicated_dataset,
    deduplicate_datasets,
    
    # Coordinate to study mapping
    get_studies_for_coordinate,
    get_studies_for_coordinate_dedup,
    
    # Prompt generation
    generate_llm_prompt
)

# AI model interface
try:
    from .ai_model_interface import AIModelInterface
except ImportError:
    # Make AIModelInterface optional to avoid breaking if dependencies aren't installed
    pass
__all__ = [
    "AtlasMapper",
    "BatchAtlasMapper",
    "MultiAtlasMapper",
    "AtlasFetcher",
    "AtlasFileHandler",
    "get_studies_for_coordinate",
]
