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

# coord2study utilities
from .coord2study import (
    fetch_datasets,
    load_deduplicated_dataset,
    deduplicate_datasets,
    prepare_datasets,
    search_studies,
    get_studies_for_coordinate,
    generate_llm_prompt,
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
    "fetch_datasets",
    "prepare_datasets",
    "search_studies",
    "get_studies_for_coordinate",
]
