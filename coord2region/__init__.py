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
)
from .prompt_utils import (
    IMAGE_PROMPT_TEMPLATES,
    LLM_PROMPT_TEMPLATES,
    generate_llm_prompt,
    generate_region_image_prompt,
)
from .llm_service import generate_summary

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
    "generate_llm_prompt",
    "generate_region_image_prompt",
    "generate_summary",
    "LLM_PROMPT_TEMPLATES",
    "IMAGE_PROMPT_TEMPLATES",
]
