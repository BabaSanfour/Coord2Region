"""Coord2Region: map brain coordinates to regions, studies, and AI insights."""

from __future__ import annotations

from .ai_model_interface import AIModelInterface
from .ai_reports import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_SYSTEM_MESSAGE,
    ReasonedReport,
    ReasonedReportContext,
    build_reasoned_report_messages,
    build_region_image_request,
    infer_hemisphere,
    parse_reasoned_report_output,
    run_reasoned_report,
)
from .coord2region import AtlasMapper, BatchAtlasMapper, MultiAtlasMapper
from .coord2study import (
    deduplicate_datasets,
    fetch_datasets,
    get_studies_for_coordinate,
    load_deduplicated_dataset,
    prepare_datasets,
    search_studies,
)
from .fetching import AtlasFetcher
from .llm import (
    IMAGE_PROMPT_TEMPLATES,
    LLM_PROMPT_TEMPLATES,
    generate_batch_summaries,
    generate_llm_prompt,
    generate_mni152_image,
    generate_region_image,
    generate_summary,
    generate_summary_async,
    stream_summary,
)
from .paths import get_working_directory
from .pipeline import run_pipeline
from .utils.file_handler import AtlasFileHandler

__all__ = [
    "AIModelInterface",
    "ReasonedReportContext",
    "ReasonedReport",
    "DEFAULT_SYSTEM_MESSAGE",
    "DEFAULT_NEGATIVE_PROMPT",
    "infer_hemisphere",
    "build_reasoned_report_messages",
    "parse_reasoned_report_output",
    "run_reasoned_report",
    "build_region_image_request",
    "AtlasMapper",
    "BatchAtlasMapper",
    "MultiAtlasMapper",
    "AtlasFetcher",
    "AtlasFileHandler",
    "get_working_directory",
    "fetch_datasets",
    "load_deduplicated_dataset",
    "deduplicate_datasets",
    "prepare_datasets",
    "search_studies",
    "get_studies_for_coordinate",
    "run_pipeline",
    "IMAGE_PROMPT_TEMPLATES",
    "LLM_PROMPT_TEMPLATES",
    "generate_llm_prompt",
    "generate_region_image",
    "generate_summary",
    "generate_batch_summaries",
    "generate_summary_async",
    "stream_summary",
    "generate_mni152_image",
]
