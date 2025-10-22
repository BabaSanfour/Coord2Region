"""High-level analysis pipeline for Coord2Region.

This module exposes a single convenience function :func:`run_pipeline` which
coordinates the existing building blocks in the package to provide an
end-to-end workflow. Users can submit coordinates or region names and request
different types of outputs such as atlas labels, textual summaries, generated
images and the raw study metadata.

The implementation builds directly on the lower-level modules in the package.
Atlas lookups are performed via :mod:`coord2region.coord2region`, studies are
retrieved using :mod:`coord2region.coord2study`, and text or image generation is
handled through :mod:`coord2region.llm`.

The function also supports exporting the produced results to a variety of
formats.
"""

from __future__ import annotations

import asyncio
import json
import os
import pickle
import logging
from pathlib import Path
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

from .utils.file_handler import save_as_csv, save_as_pdf, save_batch_folder

from .coord2study import get_studies_for_coordinate, prepare_datasets
from .coord2region import MultiAtlasMapper
from .llm import (
    generate_mni152_image,
    generate_region_image,
    generate_summary,
    generate_summary_async,
)
from .ai_model_interface import AIModelInterface
from .utils import resolve_working_directory
from .fetching import AtlasFetcher  # noqa: F401 - used by tests via patching


@dataclass
class PipelineResult:
    """Structured container returned by :func:`run_pipeline`.

    Parameters
    ----------
    coordinate : Optional[List[float]]
        Coordinate associated with this result (if available).
    mni_coordinates : Optional[List[float]]
        Representative MNI coordinate resolved from region name inputs when
        requested via ``outputs``.
    region_labels : Dict[str, str]
        Atlas region labels keyed by atlas name.
    summaries : Dict[str, str]
        Mapping of language-model identifiers to their generated summaries.
    summary : Optional[str]
        Primary summary (first entry in :attr:`summaries`) kept for
        backward compatibility.
    studies : List[Dict[str, Any]]
        Raw study metadata dictionaries.
    image : Optional[str]
        Primary image path (first generated), kept for backward compatibility.
    images : Dict[str, str]
        Mapping of image backend names to generated image paths.
    warnings : List[str]
        Non-fatal issues encountered while processing the input item.
    """

    coordinate: Optional[List[float]] = None
    mni_coordinates: Optional[List[float]] = None
    region_labels: Dict[str, str] = field(default_factory=dict)
    summaries: Dict[str, str] = field(default_factory=dict)
    summary: Optional[str] = None
    studies: List[Dict[str, Any]] = field(default_factory=list)
    image: Optional[str] = None
    images: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def _normalize_model_list(value: Any) -> List[str]:
    """Coerce a config value into a list of unique model identifiers."""
    if value is None:
        return []
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, Sequence):
        candidates = list(value)
    else:
        candidates = [value]

    normalized: List[str] = []
    for item in candidates:
        if item is None:
            continue
        name = str(item).strip()
        if name and name not in normalized:
            normalized.append(name)
    return normalized


def _get_summary_models(config: Dict[str, Any], default_model: str) -> List[str]:
    """Return the ordered list of summary models honoring config defaults."""
    raw = config.get("summary_models")
    models = _normalize_model_list(raw)
    if not models and default_model:
        models = [default_model]
    return models


def _export_results(results: List[PipelineResult], fmt: str, path: str) -> None:
    """Export pipeline results to the requested format."""
    dict_results = [asdict(r) for r in results]

    if fmt in {"json", "pickle"}:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    if fmt == "json":
        with open(path, "w", encoding="utf8") as f:
            json.dump(dict_results, f, indent=2)
        return

    if fmt == "pickle":
        with open(path, "wb") as f:
            pickle.dump(dict_results, f)
        return

    if fmt == "csv":
        save_as_csv(results, path)
        return

    if fmt == "pdf":
        save_as_pdf(results, path)
        return

    if fmt == "directory":
        save_batch_folder(results, path)
        return

    raise ValueError(f"Unknown export format: {fmt}")


def run_pipeline(
    inputs: Sequence[Any],
    input_type: str,
    outputs: Sequence[str],
    output_format: Optional[str] = None,
    output_name: Optional[str] = None,
    image_backend: str = "ai",
    *,
    config: Optional[Dict[str, Any]] = None,
    async_mode: bool = False,
    progress_callback: Optional[Callable[[int, int, PipelineResult], None]] = None,
) -> List[PipelineResult]:
    """Run the Coord2Region analysis pipeline.

    Parameters
    ----------
    inputs : sequence
        Iterable containing the inputs. The interpretation depends on
        ``input_type``.
    input_type : {"coords", "region_names"}
        Specifies how to treat ``inputs``.
    outputs : sequence of
        {"region_labels", "summaries", "images", "raw_studies", "mni_coordinates"}
        Requested pieces of information for each input item.
        The ``"mni_coordinates"`` option is only supported when
        ``input_type == "region_names"``.
    output_format : {"json", "pickle", "csv", "pdf", "directory"}, optional
        When provided, results are exported to the specified format.
    output_name : str, optional
        File or directory name to use when exporting results. The name is
        created inside the working directory's ``results`` subfolder.
        Required when ``output_format`` is specified.
    image_backend : {"ai", "nilearn", "both"}, optional
        Backend used to generate images when ``"images"`` is requested.
    prompt_template : str, optional
        Template to use for AI image generation prompts. One of: "
        'anatomical', 'functional', 'schematic', 'artistic', or 'custom'.
    async_mode : bool, optional
        When ``True``, processing occurs concurrently using asyncio and summaries
        are generated with :func:`generate_summary_async`.
    progress_callback : callable, optional
        Function invoked after each input is processed. Receives the number of
        completed items, the total count and the :class:`PipelineResult` for the
        processed item. When ``None``, progress is logged via ``logging``.

    Returns
    -------
    list of :class:`PipelineResult`
        One result object per item in ``inputs``.
    """
    input_type = input_type.lower()
    if input_type not in {"coords", "region_names"}:
        raise ValueError("input_type must be 'coords' or 'region_names'")

    outputs = [o.lower() for o in outputs]
    base_outputs = {"region_labels", "summaries", "images", "raw_studies"}
    valid_outputs = set(base_outputs)
    if input_type == "region_names":
        valid_outputs.add("mni_coordinates")

    invalid_outputs = sorted(set(outputs) - valid_outputs)
    if invalid_outputs:
        raise ValueError(
            "outputs must be a subset of "
            f"{sorted(valid_outputs)} for input_type='{input_type}'"
        )

    if output_format and output_name is None:
        raise ValueError("output_name must be provided when output_format is set")

    image_backend = image_backend.lower()
    if image_backend not in {"ai", "nilearn", "both"}:
        raise ValueError("image_backend must be 'ai', 'nilearn' or 'both'")

    if async_mode:
        return asyncio.run(
            _run_pipeline_async(
                inputs,
                input_type,
                outputs,
                output_format,
                output_name,
                image_backend=image_backend,
                config=config,
                progress_callback=progress_callback,
            )
        )

    kwargs = config or {}
    study_search_radius = float(kwargs.get("study_search_radius", 0))
    region_radius_value = kwargs.get("region_search_radius")
    region_search_radius = (
        float(region_radius_value) if region_radius_value is not None else None
    )
    # unified sources control both dataset preparation and study search
    sources = kwargs.get("sources")
    summary_models = _get_summary_models(kwargs, default_model="gpt-4o-mini")
    prompt_type = kwargs.get("prompt_type") or "summary"
    custom_prompt = kwargs.get("custom_prompt")
    summary_max_tokens = kwargs.get("summary_max_tokens", 1000)

    working_dir = resolve_working_directory(kwargs.get("working_directory"))
    working_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = working_dir / "cached_data"
    image_dir = working_dir / "generated_images"
    results_dir = working_dir / "results"
    for p in (cache_dir, image_dir, results_dir):
        p.mkdir(parents=True, exist_ok=True)

    export_path: Optional[Path] = None
    if output_format:
        output_label = cast(str, output_name)
        name_path = Path(output_label)
        if not output_label or name_path.name != output_label:
            message = (
                "output_name must be a single file or directory name without path"
                " separators"
            )
            raise ValueError(message)
        export_path = results_dir / output_label

    email = kwargs.get("email_for_abstracts")
    use_cached_dataset = kwargs.get("use_cached_dataset", True)
    atlas_names = kwargs.get("atlas_names", ["harvard-oxford", "juelich", "aal"])
    provider_configs = kwargs.get("providers")
    gemini_api_key = kwargs.get("gemini_api_key")
    openrouter_api_key = kwargs.get("openrouter_api_key")
    openai_api_key = kwargs.get("openai_api_key")
    openai_project = kwargs.get("openai_project")
    anthropic_api_key = kwargs.get("anthropic_api_key")
    huggingface_api_key = kwargs.get("huggingface_api_key")
    image_model = kwargs.get("image_model", "stabilityai/stable-diffusion-2")
    image_prompt_type = kwargs.get("image_prompt_type") or "anatomical"
    image_custom_prompt = kwargs.get("image_custom_prompt")

    dataset = (
        prepare_datasets(str(working_dir), sources=sources)
        if use_cached_dataset
        else None
    )
    ai = None
    if provider_configs:
        ai = AIModelInterface()
        for name, cfg in provider_configs.items():
            ai.register_provider(name, **cfg)
    elif any(
        [
            gemini_api_key,
            openrouter_api_key,
            openai_api_key,
            anthropic_api_key,
            huggingface_api_key,
        ]
    ):
        ai = AIModelInterface(
            gemini_api_key=gemini_api_key,
            openrouter_api_key=openrouter_api_key,
            openai_api_key=openai_api_key,
            openai_project=openai_project,
            anthropic_api_key=anthropic_api_key,
            huggingface_api_key=huggingface_api_key,
        )

    atlas_configs = kwargs.get("atlas_configs") or {}
    atlas_dict = {name: dict(atlas_configs.get(name, {})) for name in atlas_names or []}
    if not atlas_dict:
        raise ValueError(
            "At least one atlas name must be provided to run the pipeline."
        )
    try:
        multi_atlas: MultiAtlasMapper = MultiAtlasMapper(str(working_dir), atlas_dict)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise RuntimeError("Failed to initialize atlas mappers") from exc

    def _from_region_name(name: str) -> Optional[List[float]]:
        coords_dict = multi_atlas.batch_region_name_to_mni([name])
        for atlas_coords in coords_dict.values():
            if atlas_coords:
                coord = atlas_coords[0]
                if coord is not None:
                    try:
                        return coord.tolist()  # type: ignore[attr-defined]
                    except Exception:
                        return list(coord)  # type: ignore[arg-type]
        return None

    results: List[PipelineResult] = []

    for item in inputs:
        region_name_input: Optional[str] = None
        if input_type == "coords":
            coord = list(item) if item is not None else None
        elif input_type == "region_names":
            region_name_input = str(item)
            coord = _from_region_name(region_name_input)
        else:
            # only "coords" or "region_names" supported
            coord = None

        res = PipelineResult(coordinate=coord)
        if coord is not None and "mni_coordinates" in outputs:
            res.mni_coordinates = list(coord)

        # No special case for input_type "studies" (unsupported)

        if coord is None:
            if region_name_input is not None:
                message = (
                    "Region '{name}' could not be resolved to coordinates "
                    "with the configured atlases."
                ).format(name=region_name_input)
                logging.warning(message)
                res.warnings.append(message)
            results.append(res)
            if progress_callback:
                progress_callback(len(results), len(inputs), res)
            else:
                logging.info("Processed %d/%d inputs", len(results), len(inputs))
            continue

        if "region_labels" in outputs:
            try:
                batch = multi_atlas.batch_mni_to_region_names(
                    [coord], max_distance=region_search_radius
                )
                # Extract first match per atlas
                res.region_labels = {
                    atlas: (names[0] if names else "Unknown")
                    for atlas, names in batch.items()
                }
            except Exception:
                res.region_labels = {}

        if ("raw_studies" in outputs or "summaries" in outputs) and dataset is not None:
            try:
                if isinstance(dataset, dict):
                    res.studies = get_studies_for_coordinate(
                        dataset,
                        coord,
                        radius=study_search_radius,
                        email=email,
                        sources=sources,
                    )
                else:
                    # When using a single deduplicated Dataset (not a mapping),
                    # do not pass 'sources' to avoid filtering out the combined set.
                    res.studies = get_studies_for_coordinate(
                        dataset, coord, radius=study_search_radius, email=email
                    )
            except Exception:
                res.studies = []

        if "summaries" in outputs and ai and summary_models:
            for model_idx, model_name in enumerate(summary_models):
                summary_text = generate_summary(
                    ai,
                    res.studies,
                    coord,
                    prompt_type=prompt_type,
                    model=model_name,
                    atlas_labels=res.region_labels or None,
                    custom_prompt=custom_prompt if prompt_type == "custom" else None,
                    max_tokens=summary_max_tokens,
                )
                res.summaries[model_name] = summary_text
                if model_idx == 0:
                    res.summary = summary_text

        if "images" in outputs:
            img_dir = image_dir
            os.makedirs(img_dir, exist_ok=True)

            if image_backend in {"ai", "both"} and ai:
                region_info = {
                    "summary": res.summary or "",
                    "atlas_labels": res.region_labels,
                }
                try:
                    img_bytes = generate_region_image(
                        ai,
                        coord,
                        region_info,
                        image_type=image_prompt_type,
                        model=image_model,
                        watermark=True,
                        prompt_template=(
                            image_custom_prompt
                            if image_prompt_type == "custom"
                            else None
                        ),
                    )
                    img_path = img_dir / f"image_{len(list(img_dir.iterdir())) + 1}.png"
                    with open(img_path, "wb") as f:
                        f.write(img_bytes)
                    res.image = res.image or str(img_path)
                    res.images["ai"] = str(img_path)
                except Exception:
                    pass

            if image_backend in {"nilearn", "both"}:
                try:
                    img_bytes = generate_mni152_image(coord)
                    img_path = img_dir / f"image_{len(list(img_dir.iterdir())) + 1}.png"
                    with open(img_path, "wb") as f:
                        f.write(img_bytes)
                    if res.image is None:
                        res.image = str(img_path)
                    res.images["nilearn"] = str(img_path)
                except Exception:
                    pass

        results.append(res)
        if progress_callback:
            progress_callback(len(results), len(inputs), res)
        else:
            logging.info("Processed %d/%d inputs", len(results), len(inputs))

    if output_format and export_path is not None:
        _export_results(results, output_format.lower(), str(export_path))

    return results


async def _run_pipeline_async(
    inputs: Sequence[Any],
    input_type: str,
    outputs: Sequence[str],
    output_format: Optional[str],
    output_name: Optional[str],
    image_backend: str,
    *,
    config: Optional[Dict[str, Any]],
    progress_callback: Optional[Callable[[int, int, PipelineResult], None]],
) -> List[PipelineResult]:
    """Asynchronous implementation backing :func:`run_pipeline`."""
    kwargs = config or {}
    study_search_radius = float(kwargs.get("study_search_radius", 0))
    region_radius_value = kwargs.get("region_search_radius")
    region_search_radius = (
        float(region_radius_value) if region_radius_value is not None else None
    )
    # unified sources control both dataset preparation and study search
    sources = kwargs.get("sources")
    summary_models = _get_summary_models(kwargs, default_model="gpt-4o-mini")
    prompt_type = kwargs.get("prompt_type") or "summary"
    custom_prompt = kwargs.get("custom_prompt")
    summary_max_tokens = kwargs.get("summary_max_tokens", 1000)

    working_dir = resolve_working_directory(kwargs.get("working_directory"))
    working_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = working_dir / "cached_data"
    image_dir = working_dir / "generated_images"
    results_dir = working_dir / "results"
    for p in (cache_dir, image_dir, results_dir):
        p.mkdir(parents=True, exist_ok=True)

    export_path: Optional[Path] = None
    if output_format:
        output_label = cast(str, output_name)
        name_path = Path(output_label)
        if not output_label or name_path.name != output_label:
            message = (
                "output_name must be a single file or directory name without path"
                " separators"
            )
            raise ValueError(message)
        export_path = results_dir / output_label

    email = kwargs.get("email_for_abstracts")
    use_cached_dataset = kwargs.get("use_cached_dataset", True)
    atlas_names = kwargs.get("atlas_names", ["harvard-oxford", "juelich", "aal"])
    provider_configs = kwargs.get("providers")
    gemini_api_key = kwargs.get("gemini_api_key")
    openrouter_api_key = kwargs.get("openrouter_api_key")
    openai_api_key = kwargs.get("openai_api_key")
    openai_project = kwargs.get("openai_project")
    anthropic_api_key = kwargs.get("anthropic_api_key")
    huggingface_api_key = kwargs.get("huggingface_api_key")
    image_model = kwargs.get("image_model", "stabilityai/stable-diffusion-2")
    image_prompt_type = kwargs.get("image_prompt_type") or "anatomical"
    image_custom_prompt = kwargs.get("image_custom_prompt")

    dataset = (
        await asyncio.to_thread(prepare_datasets, str(working_dir), sources)
        if use_cached_dataset
        else None
    )
    ai = None
    if provider_configs:
        ai = AIModelInterface()
        for name, cfg in provider_configs.items():
            ai.register_provider(name, **cfg)
    elif any(
        [
            gemini_api_key,
            openrouter_api_key,
            openai_api_key,
            anthropic_api_key,
            huggingface_api_key,
        ]
    ):
        ai = AIModelInterface(
            gemini_api_key=gemini_api_key,
            openrouter_api_key=openrouter_api_key,
            openai_api_key=openai_api_key,
            openai_project=openai_project,
            anthropic_api_key=anthropic_api_key,
            huggingface_api_key=huggingface_api_key,
        )

    atlas_configs = kwargs.get("atlas_configs") or {}
    atlas_dict = {name: dict(atlas_configs.get(name, {})) for name in atlas_names or []}
    if not atlas_dict:
        raise ValueError(
            "At least one atlas name must be provided to run the pipeline."
        )
    try:
        multi_atlas: MultiAtlasMapper = MultiAtlasMapper(str(working_dir), atlas_dict)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise RuntimeError("Failed to initialize atlas mappers") from exc

    def _from_region_name(name: str) -> Optional[List[float]]:
        coords_dict = multi_atlas.batch_region_name_to_mni([name])
        for atlas_coords in coords_dict.values():
            if atlas_coords:
                coord = atlas_coords[0]
                if coord is not None:
                    try:
                        return coord.tolist()  # type: ignore[attr-defined]
                    except Exception:
                        return list(coord)  # type: ignore[arg-type]
        return None

    total = len(inputs)
    results: List[Optional[PipelineResult]] = [None] * total

    async def _process(idx: int, item: Any) -> Tuple[int, PipelineResult]:
        region_name_input: Optional[str] = None
        if input_type == "coords":
            coord = list(item) if item is not None else None
        elif input_type == "region_names":
            region_name_input = str(item)
            coord = await asyncio.to_thread(_from_region_name, region_name_input)
        else:
            # only "coords" or "region_names" supported
            coord = None

        res = PipelineResult(coordinate=coord)
        if coord is not None and "mni_coordinates" in outputs:
            res.mni_coordinates = list(coord)

        if coord is None:
            if region_name_input is not None:
                message = (
                    "Region '{name}' could not be resolved to coordinates "
                    "with the configured atlases."
                ).format(name=region_name_input)
                logging.warning(message)
                res.warnings.append(message)
            return idx, res

        if "region_labels" in outputs:
            try:
                batch = await asyncio.to_thread(
                    multi_atlas.batch_mni_to_region_names,
                    [coord],
                    max_distance=region_search_radius,
                )
                res.region_labels = {
                    atlas: (names[0] if names else "Unknown")
                    for atlas, names in batch.items()
                }
            except Exception:
                res.region_labels = {}

        if ("raw_studies" in outputs or "summaries" in outputs) and dataset is not None:
            try:
                if isinstance(dataset, dict):
                    res.studies = await asyncio.to_thread(
                        lambda: get_studies_for_coordinate(
                            dataset,
                            coord,
                            radius=study_search_radius,
                            email=email,
                            sources=sources,
                        )
                    )
                else:
                    res.studies = await asyncio.to_thread(
                        lambda: get_studies_for_coordinate(
                            dataset, coord, radius=study_search_radius, email=email
                        )
                    )
            except Exception:
                res.studies = []

        if "summaries" in outputs and ai and summary_models:
            for model_idx, model_name in enumerate(summary_models):
                summary_text = await generate_summary_async(
                    ai,
                    res.studies,
                    coord,
                    prompt_type=prompt_type,
                    model=model_name,
                    atlas_labels=res.region_labels or None,
                    custom_prompt=custom_prompt if prompt_type == "custom" else None,
                    max_tokens=summary_max_tokens,
                )
                res.summaries[model_name] = summary_text
                if model_idx == 0:
                    res.summary = summary_text

        if "images" in outputs:
            img_dir = image_dir
            os.makedirs(img_dir, exist_ok=True)

            if image_backend in {"ai", "both"} and ai:
                region_info = {
                    "summary": res.summary or "",
                    "atlas_labels": res.region_labels,
                }

                def _save_ai_image() -> str:
                    img_bytes = generate_region_image(
                        ai,
                        coord,
                        region_info,
                        image_type=image_prompt_type,
                        model=image_model,
                        watermark=True,
                        prompt_template=(
                            image_custom_prompt
                            if image_prompt_type == "custom"
                            else None
                        ),
                    )
                    img_path = img_dir / f"image_{len(list(img_dir.iterdir())) + 1}.png"
                    with open(img_path, "wb") as f:
                        f.write(img_bytes)
                    return str(img_path)

                try:
                    path = await asyncio.to_thread(_save_ai_image)
                    res.image = res.image or path
                    res.images["ai"] = path
                except Exception:
                    pass

            if image_backend in {"nilearn", "both"}:

                def _save_nilearn_image() -> str:
                    img_bytes = generate_mni152_image(coord)
                    img_path = img_dir / f"image_{len(list(img_dir.iterdir())) + 1}.png"
                    with open(img_path, "wb") as f:
                        f.write(img_bytes)
                    return str(img_path)

                try:
                    path = await asyncio.to_thread(_save_nilearn_image)
                    if res.image is None:
                        res.image = path
                    res.images["nilearn"] = path
                except Exception:
                    pass

        return idx, res

    tasks = [asyncio.create_task(_process(i, item)) for i, item in enumerate(inputs)]

    completed = 0
    for fut in asyncio.as_completed(tasks):
        idx, res = await fut
        results[idx] = res
        completed += 1
        if progress_callback:
            progress_callback(completed, total, res)
        else:
            logging.info("Processed %d/%d inputs", completed, total)

    final_results = [r for r in results if r is not None]

    if output_format and export_path is not None:
        await asyncio.to_thread(
            _export_results,
            final_results,
            output_format.lower(),
            str(export_path),
        )

    return final_results
