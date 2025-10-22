"""Coord2Region: map brain coordinates to regions, studies, and AI insights."""

from __future__ import annotations

import warnings

from .ai_model_interface import AIModelInterface  # noqa: F401
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
]

try:
    import sklearn.cluster  # type: ignore
except Exception:
    _HAS_SKLEARN = False
else:
    _HAS_SKLEARN = True

if _HAS_SKLEARN:
    try:
        from .coord2region import AtlasMapper, BatchAtlasMapper, MultiAtlasMapper
        from .fetching import AtlasFetcher
        from .utils.file_handler import AtlasFileHandler
        from .paths import get_working_directory

        __all__.extend([
            "AtlasMapper",
            "BatchAtlasMapper",
            "MultiAtlasMapper",
            "AtlasFetcher",
            "AtlasFileHandler",
            "get_working_directory",
        ])
    except Exception as exc:  # pragma: no cover
        warnings.warn(
            f"Optional atlas utilities unavailable: {exc!r}. Install full dependencies to enable coord2region atlas helpers.",
            RuntimeWarning,
        )

    try:
        from .coord2study import (
            fetch_datasets,
            load_deduplicated_dataset,
            deduplicate_datasets,
            prepare_datasets,
            search_studies,
            get_studies_for_coordinate,
        )

        __all__.extend([
            "fetch_datasets",
            "load_deduplicated_dataset",
            "deduplicate_datasets",
            "prepare_datasets",
            "search_studies",
            "get_studies_for_coordinate",
        ])
    except Exception as exc:  # pragma: no cover
        warnings.warn(
            f"Optional coord2study utilities unavailable: {exc!r}.",
            RuntimeWarning,
        )

    try:
        from .pipeline import run_pipeline

        __all__.append("run_pipeline")
    except Exception as exc:  # pragma: no cover
        warnings.warn(
            f"Pipeline helpers unavailable: {exc!r}. Install full dependencies to use the pipeline.",
            RuntimeWarning,
        )

if _HAS_SKLEARN:
    try:
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

        __all__.extend([
            "IMAGE_PROMPT_TEMPLATES",
            "LLM_PROMPT_TEMPLATES",
            "generate_llm_prompt",
            "generate_region_image",
            "generate_summary",
            "generate_batch_summaries",
            "generate_summary_async",
            "stream_summary",
            "generate_mni152_image",
        ])
    except Exception as exc:  # pragma: no cover
        warnings.warn(
            f"Optional LLM utilities unavailable: {exc!r}.",
            RuntimeWarning,
        )
