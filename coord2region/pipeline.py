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

import json
import os
import pickle
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, cast

from .file_handler import save_as_csv, save_as_pdf, save_batch_folder

from .coord2study import get_studies_for_coordinate, prepare_datasets
from .coord2region import AtlasMapper, MultiAtlasMapper
from .fetching import AtlasFetcher
from .llm import generate_region_image, generate_summary
from .ai_model_interface import AIModelInterface


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
        Path or URL to a generated image representing the region.
    """

    coordinate: Optional[List[float]] = None
    region_labels: Dict[str, str] = field(default_factory=dict)
    summary: Optional[str] = None
    studies: List[Dict[str, Any]] = field(default_factory=list)
    image: Optional[str] = None


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
    *,
    brain_insights_kwargs: Optional[Dict[str, Any]] = None,
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
        Target file or directory for ``output_format``. Required when an
        ``output_format`` is specified.
    brain_insights_kwargs : dict, optional
        Additional configuration for datasets, atlases and model providers. The
        name is kept for backward compatibility with earlier API versions. To
        enable or disable AI providers, supply a ``providers`` dictionary mapping
        provider names to keyword arguments understood by
        :meth:`AIModelInterface.register_provider`.

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

    kwargs = brain_insights_kwargs or {}
    data_dir = kwargs.get("data_dir", "nimare_data")
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

    dataset = prepare_datasets(data_dir) if use_cached_dataset else None
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
            fetcher = AtlasFetcher()
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
            continue

        if coord is None:
            results.append(res)
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

        if "images" in outputs and ai:
            region_info = {
                "summary": res.summary or "",
                "atlas_labels": res.region_labels,
            }
            try:
                img_bytes = generate_region_image(
                    ai, coord, region_info, model=image_model
                )
                img_dir = os.path.join(data_dir, "generated_images")
                os.makedirs(img_dir, exist_ok=True)
                img_path = os.path.join(
                    img_dir, f"image_{len(os.listdir(img_dir)) + 1}.png"
                )
                with open(img_path, "wb") as f:
                    f.write(img_bytes)
                res.image = img_path
            except Exception:
                pass

        results.append(res)

    if output_format:
        _export_results(results, output_format.lower(), cast(str, output_path))

    return results
