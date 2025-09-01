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
from .utils.file_handler import AtlasFileHandler
from .paths import get_data_directory

# coord2study utilities
from .coord2study import (
    fetch_datasets,
    load_deduplicated_dataset,
    deduplicate_datasets,
    prepare_datasets,
    search_studies,
    get_studies_for_coordinate,
)
from .llm import (
    IMAGE_PROMPT_TEMPLATES,
    LLM_PROMPT_TEMPLATES,
    generate_llm_prompt,
    generate_region_image,
    generate_summary,
    generate_batch_summaries,
    generate_summary_async,
    stream_summary,
    generate_mni152_image,
)
from .pipeline import run_pipeline
from .ai_model_interface import AIModelInterface  # noqa: F401

__all__ = [
    "AtlasMapper",
    "BatchAtlasMapper",
    "MultiAtlasMapper",
    "AtlasFetcher",
    "AtlasFileHandler",
    "get_data_directory",
    "fetch_datasets",
    "load_deduplicated_dataset",
    "deduplicate_datasets",
    "prepare_datasets",
    "search_studies",
    "get_studies_for_coordinate",
    "generate_llm_prompt",
    "generate_region_image_prompt",
    "generate_region_image",
    "generate_summary",
    "generate_batch_summaries",
    "generate_summary_async",
    "stream_summary",
    "generate_mni152_image",
    "run_pipeline",
    "LLM_PROMPT_TEMPLATES",
    "IMAGE_PROMPT_TEMPLATES",
    "AIModelInterface",
]
