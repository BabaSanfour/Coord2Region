"""Coord2Region: map brain coordinates to regions, studies, and AI insights."""

from __future__ import annotations

from .ai_model_interface import AIModelInterface
from .ai_reports import (
    ReasonedReportContext,
    ReasonedReport,
    DEFAULT_SYSTEM_MESSAGE,
    DEFAULT_NEGATIVE_PROMPT,
    infer_hemisphere,
    build_reasoned_report_messages,
    parse_reasoned_report_output,
    run_reasoned_report,
    build_region_image_request,
)

from .coord2region import AtlasMapper, BatchAtlasMapper, MultiAtlasMapper
from .fetching import AtlasFetcher
from .utils.file_handler import AtlasFileHandler
from .paths import get_working_directory

from .coord2study import (
    fetch_datasets,
    load_deduplicated_dataset,
    deduplicate_datasets,
    prepare_datasets,
    search_studies,
    get_studies_for_coordinate,
)

from .pipeline import run_pipeline
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
