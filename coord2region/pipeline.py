"""High-level analysis pipeline for Coord2Region.

This module exposes a single convenience function :func:`run_pipeline` which
coordinates the existing building blocks in the package to provide an
end-to-end workflow. Users can submit coordinates, region names or pre-fetched
studies and request different types of outputs such as atlas labels, textual
summaries, generated images and the raw study metadata.

The implementation builds directly on the lower-level modules in the package.
Atlas lookups are performed via :mod:`coord2region.coord2region`, studies are
retrieved using :mod:`coord2region.coord2study`, and text or image generation is
handled through :mod:`coord2region.llm`.  Earlier versions delegated to a
``BrainInsights`` wrapper, but the relevant logic now lives in this module for a
leaner public API.

The function also supports exporting the produced results to a variety of
formats. Only standard library modules are used by default; PDF export relies
on the optional ``fpdf`` package which is lightweight and pure Python.
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
from .coord2region import AtlasMapper, MultiAtlasMapper
from .fetching import AtlasFetcher
from .llm import (
    generate_mni152_image,
    generate_region_image,
    generate_summary,
    generate_summary_async,
)
from .ai_model_interface import AIModelInterface
from .utils import resolve_data_dir


@dataclass
class PipelineResult:
    """Structured container returned by :func:`run_pipeline`.

    Parameters
    ----------
    coordinate : Optional[List[float]]
        Coordinate associated with this result (if available).
    region_labels : Dict[str, str]
        Atlas region labels keyed by atlas name.
    summary : Optional[str]
        Text summary produced by the language model.
    studies : List[Dict[str, Any]]
        Raw study metadata dictionaries.
    image : Optional[str]
        Path or URL to the first generated image (kept for backward
        compatibility).
    images : Dict[str, str]
        Mapping of image backend names to generated image paths.
    """

    coordinate: Optional[List[float]] = None
    region_labels: Dict[str, str] = field(default_factory=dict)
    summary: Optional[str] = None
    studies: List[Dict[str, Any]] = field(default_factory=list)
    image: Optional[str] = None
    images: Dict[str, str] = field(default_factory=dict)


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
    output_path: Optional[str] = None,
    image_backend: str = "ai",
    *,
    brain_insights_kwargs: Optional[Dict[str, Any]] = None,
    async_mode: bool = False,
    progress_callback: Optional[
        Callable[[int, int, PipelineResult], None]
    ] = None,
) -> List[PipelineResult]:
    """Run the Coord2Region analysis pipeline.

    Parameters
    ----------
    inputs : sequence
        Iterable containing the inputs. The interpretation depends on
        ``input_type``.
    input_type : {"coords", "region_names", "studies"}
        Specifies how to treat ``inputs``.
    outputs : sequence of {"region_labels", "summaries", "images", "raw_studies"}
        Requested pieces of information for each input item.
    output_format : {"json", "pickle", "csv", "pdf", "directory"}, optional
        When provided, results are exported to the specified format.
    output_path : str, optional
        Target file or directory for ``output_format``. Relative paths are
        resolved against the base data directory. Required when an
        ``output_format`` is specified.
    image_backend : {"ai", "nilearn", "both"}, optional
        Backend used to generate images when ``"images"`` is requested.
        Defaults to ``"ai"``.
    brain_insights_kwargs : dict, optional
        Additional configuration for datasets, atlases and model providers. The
        name is kept for backward compatibility with earlier API versions. To
        enable or disable AI providers, supply a ``providers`` dictionary mapping
        provider names to keyword arguments understood by
        :meth:`AIModelInterface.register_provider`.
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
    if input_type not in {"coords", "region_names", "studies"}:
        raise ValueError("input_type must be 'coords', 'region_names' or 'studies'")

    outputs = [o.lower() for o in outputs]
    valid_outputs = {"region_labels", "summaries", "images", "raw_studies"}
    if any(o not in valid_outputs for o in outputs):
        raise ValueError(f"outputs must be a subset of {valid_outputs}")

    if output_format and output_path is None:
        raise ValueError("output_path must be provided when output_format is set")

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
                output_path,
                image_backend=image_backend,
                brain_insights_kwargs=brain_insights_kwargs,
                progress_callback=progress_callback,
            )
        )

    kwargs = brain_insights_kwargs or {}
    base_dir = resolve_data_dir(kwargs.get("data_dir"))
    base_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = base_dir / "cached_data"
    image_dir = base_dir / "generated_images"
    results_dir = base_dir / "results"
    for p in (cache_dir, image_dir, results_dir):
        p.mkdir(parents=True, exist_ok=True)

    if output_path is not None:
        op = Path(output_path).expanduser()
        if not op.is_absolute():
            op = results_dir / op
        output_path = str(op)

    email = kwargs.get("email_for_abstracts")
    use_cached_dataset = kwargs.get("use_cached_dataset", True)
    use_atlases = kwargs.get("use_atlases", True)
    atlas_names = kwargs.get("atlas_names", ["harvard-oxford", "juelich", "aal"])
    provider_configs = kwargs.get("providers")
    gemini_api_key = kwargs.get("gemini_api_key")
    openrouter_api_key = kwargs.get("openrouter_api_key")
    openai_api_key = kwargs.get("openai_api_key")
    anthropic_api_key = kwargs.get("anthropic_api_key")
    huggingface_api_key = kwargs.get("huggingface_api_key")
    image_model = kwargs.get("image_model", "stabilityai/stable-diffusion-2")

    dataset = prepare_datasets(str(base_dir)) if use_cached_dataset else None
    ai = None
    if provider_configs:
        ai = AIModelInterface()
        for name, cfg in provider_configs.items():
            ai.register_provider(name, **cfg)
    elif any(
        [gemini_api_key, openrouter_api_key, openai_api_key, anthropic_api_key, huggingface_api_key]
    ):
        ai = AIModelInterface(
            gemini_api_key=gemini_api_key,
            openrouter_api_key=openrouter_api_key,
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_key,
            huggingface_api_key=huggingface_api_key,
        )

    multi_atlas: Optional[MultiAtlasMapper] = None
    if use_atlases:
        try:
            fetcher = AtlasFetcher(data_dir=str(base_dir))
            mappers = []
            for name in atlas_names:
                try:
                    atlas = fetcher.fetch_atlas(name)
                    mappers.append(
                        AtlasMapper(
                            name=name,
                            vol=atlas["vol"],
                            hdr=atlas["hdr"],
                            labels=atlas["labels"],
                        )
                    )
                except Exception:
                    continue
            if mappers:
                multi_atlas = MultiAtlasMapper(mappers)
        except Exception:
            multi_atlas = None

    def _from_region_name(name: str) -> Optional[List[float]]:
        if not multi_atlas:
            return None
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
        if input_type == "coords":
            coord = list(item) if item is not None else None
        elif input_type == "region_names":
            coord = _from_region_name(str(item))
        else:  # "studies"
            coord = None

        res = PipelineResult(coordinate=coord)

        if input_type == "studies":
            if "raw_studies" in outputs:
                res.studies = [item] if isinstance(item, dict) else list(item)
            if "summaries" in outputs and ai:
                res.summary = generate_summary(ai, res.studies, coord or [0, 0, 0])
            results.append(res)
            if progress_callback:
                progress_callback(len(results), len(inputs), res)
            else:
                logging.info("Processed %d/%d inputs", len(results), len(inputs))
            continue

        if coord is None:
            results.append(res)
            if progress_callback:
                progress_callback(len(results), len(inputs), res)
            else:
                logging.info("Processed %d/%d inputs", len(results), len(inputs))
            continue

        if "region_labels" in outputs and multi_atlas:
            try:
                res.region_labels = multi_atlas.mni_to_region_names(coord)
            except Exception:
                res.region_labels = {}

        if ("raw_studies" in outputs or "summaries" in outputs) and dataset is not None:
            try:
                res.studies = get_studies_for_coordinate(coord, dataset, email=email)
            except Exception:
                res.studies = []

        if "summaries" in outputs and ai:
            res.summary = generate_summary(
                ai, res.studies, coord, atlas_labels=res.region_labels or None
            )

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
                        ai, coord, region_info, model=image_model, watermark=True
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

    if output_format:
        _export_results(results, output_format.lower(), cast(str, output_path))

    return results


async def _run_pipeline_async(
    inputs: Sequence[Any],
    input_type: str,
    outputs: Sequence[str],
    output_format: Optional[str],
    output_path: Optional[str],
    image_backend: str,
    *,
    brain_insights_kwargs: Optional[Dict[str, Any]],
    progress_callback: Optional[Callable[[int, int, PipelineResult], None]],
) -> List[PipelineResult]:
    """Asynchronous implementation backing :func:`run_pipeline`."""

    kwargs = brain_insights_kwargs or {}
    base_dir = resolve_data_dir(kwargs.get("data_dir"))
    base_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = base_dir / "cached_data"
    image_dir = base_dir / "generated_images"
    results_dir = base_dir / "results"
    for p in (cache_dir, image_dir, results_dir):
        p.mkdir(parents=True, exist_ok=True)

    if output_path is not None:
        op = Path(output_path).expanduser()
        if not op.is_absolute():
            op = results_dir / op
        output_path = str(op)

    email = kwargs.get("email_for_abstracts")
    use_cached_dataset = kwargs.get("use_cached_dataset", True)
    use_atlases = kwargs.get("use_atlases", True)
    atlas_names = kwargs.get("atlas_names", ["harvard-oxford", "juelich", "aal"])
    provider_configs = kwargs.get("providers")
    gemini_api_key = kwargs.get("gemini_api_key")
    openrouter_api_key = kwargs.get("openrouter_api_key")
    openai_api_key = kwargs.get("openai_api_key")
    anthropic_api_key = kwargs.get("anthropic_api_key")
    huggingface_api_key = kwargs.get("huggingface_api_key")
    image_model = kwargs.get("image_model", "stabilityai/stable-diffusion-2")

    dataset = (
        await asyncio.to_thread(prepare_datasets, str(base_dir))
        if use_cached_dataset
        else None
    )
    ai = None
    if provider_configs:
        ai = AIModelInterface()
        for name, cfg in provider_configs.items():
            ai.register_provider(name, **cfg)
    elif any(
        [gemini_api_key, openrouter_api_key, openai_api_key, anthropic_api_key, huggingface_api_key]
    ):
        ai = AIModelInterface(
            gemini_api_key=gemini_api_key,
            openrouter_api_key=openrouter_api_key,
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_key,
            huggingface_api_key=huggingface_api_key,
        )

    multi_atlas: Optional[MultiAtlasMapper] = None
    if use_atlases:
        try:
            fetcher = AtlasFetcher(data_dir=str(base_dir))
            mappers: List[AtlasMapper] = []
            for name in atlas_names:
                try:
                    atlas = fetcher.fetch_atlas(name)
                    mappers.append(
                        AtlasMapper(
                            name=name,
                            vol=atlas["vol"],
                            hdr=atlas["hdr"],
                            labels=atlas["labels"],
                        )
                    )
                except Exception:
                    continue
            if mappers:
                multi_atlas = MultiAtlasMapper(mappers)
        except Exception:
            multi_atlas = None

    def _from_region_name(name: str) -> Optional[List[float]]:
        if not multi_atlas:
            return None
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
        if input_type == "coords":
            coord = list(item) if item is not None else None
        elif input_type == "region_names":
            coord = await asyncio.to_thread(_from_region_name, str(item))
        else:  # "studies"
            coord = None

        res = PipelineResult(coordinate=coord)

        if input_type == "studies":
            if "raw_studies" in outputs:
                res.studies = [item] if isinstance(item, dict) else list(item)
            if "summaries" in outputs and ai:
                res.summary = await generate_summary_async(
                    ai, res.studies, coord or [0, 0, 0]
                )
            return idx, res

        if coord is None:
            return idx, res

        if "region_labels" in outputs and multi_atlas:
            try:
                res.region_labels = await asyncio.to_thread(
                    multi_atlas.mni_to_region_names, coord
                )
            except Exception:
                res.region_labels = {}

        if ("raw_studies" in outputs or "summaries" in outputs) and dataset is not None:
            try:
                res.studies = await asyncio.to_thread(
                    get_studies_for_coordinate, coord, dataset, email
                )
            except Exception:
                res.studies = []

        if "summaries" in outputs and ai:
            res.summary = await generate_summary_async(
                ai, res.studies, coord, atlas_labels=res.region_labels or None
            )

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
                        ai, coord, region_info, model=image_model, watermark=True
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

    if output_format:
        await asyncio.to_thread(
            _export_results,
            final_results,
            output_format.lower(),
            cast(str, output_path),
        )

    return final_results
