"""Command-line interface for Coord2Region.

The CLI provides dedicated sub-commands tailored to the different
capabilities exposed by the pipeline. Coordinate-focused commands can map
MNI coordinates to atlas labels, retrieve related studies, generate
summaries, render images and combine these insights. Region-based commands
resolve atlas region names to coordinates before performing the same
operations.
"""

import argparse
import json
import os
import shlex
import sys
from dataclasses import asdict
from typing import Iterable, List, Sequence, Dict, Optional
import numbers

from importlib import metadata as importlib_metadata

import pandas as pd
import yaml

from .pipeline import run_pipeline
from .config import Coord2RegionConfig, ValidationError
from .fetching import AtlasFetcher


def _package_version() -> str:
    """Return the installed Coord2Region version string."""
    try:
        return importlib_metadata.version("coord2region")
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover - dev installs
        return "0+unknown"


def _print_available_atlases() -> None:
    """Print the known atlas identifiers."""
    fetcher = AtlasFetcher()
    for name in fetcher.list_available_atlases():
        print(name)


def _parse_coord(text: str) -> List[float]:
    """Parse a coordinate string of the form 'x,y,z' or 'x y z'."""
    parts = text.replace(",", " ").split()
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Coordinates must have three values")
    try:
        return [float(p) for p in parts]
    except ValueError as exc:  # pragma: no cover - user input
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_coords_tokens(tokens: List[str]) -> List[List[float]]:
    """Parse a list of CLI tokens into a list of coordinate triples.

    Supports both styles:
    - Separate numbers: ``30 -22 50 10 0 0``
    - Grouped strings: ``"30,-22,50" "10 0 0"``
    """
    if not tokens:
        return []

    # Try numeric grouping first: len(tokens) % 3 == 0 and all castable to float
    if len(tokens) % 3 == 0:
        try:
            vals = [float(t) for t in tokens]
            return [vals[i : i + 3] for i in range(0, len(vals), 3)]
        except ValueError:
            pass  # Fall back to per-token parsing

    # Fall back to parsing each token as "x,y,z" or "x y z"
    return [_parse_coord(tok) for tok in tokens]


def _load_coords_file(path: str) -> List[List[float]]:
    """Load coordinates from a CSV or Excel file.

    The file is expected to contain at least three columns representing ``x``,
    ``y`` and ``z`` values. Any additional columns are ignored.
    """
    if path.lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    if df.shape[1] < 3:
        raise argparse.ArgumentTypeError(
            "Input file must have at least three columns for x, y, z"
        )
    return df.iloc[:, :3].astype(float).values.tolist()


def _batch(seq: Sequence, size: int) -> Iterable[Sequence]:
    """Yield ``seq`` in chunks of ``size`` (or the full sequence if ``size`` <= 0)."""
    if size <= 0 or size >= len(seq):
        yield seq
    else:
        for i in range(0, len(seq), size):
            yield seq[i : i + size]


def _atlas_source_from_value(value: str) -> Optional[Dict[str, str]]:
    text = str(value).strip()
    if not text:
        return None
    lower = text.lower()
    if lower.startswith(("http://", "https://")):
        return {"atlas_url": text}
    if text.startswith(("~", "./", "../")):
        return {"atlas_file": text}
    expanded = os.path.expanduser(text)
    if os.path.isabs(expanded):
        return {"atlas_file": text}
    if os.sep in text or (os.altsep and os.altsep in text):
        return {"atlas_file": text}
    if len(text) > 2 and text[1] == ":" and text[0].isalpha():
        return {"atlas_file": text}
    return None


def _collect_kwargs(args: argparse.Namespace) -> dict:
    """Collect keyword arguments for :func:`run_pipeline` from parsed args."""
    kwargs = {}
    if getattr(args, "gemini_api_key", None):
        kwargs["gemini_api_key"] = args.gemini_api_key
    if getattr(args, "openrouter_api_key", None):
        kwargs["openrouter_api_key"] = args.openrouter_api_key
    if getattr(args, "openai_api_key", None):
        kwargs["openai_api_key"] = args.openai_api_key
    if getattr(args, "anthropic_api_key", None):
        kwargs["anthropic_api_key"] = args.anthropic_api_key
    if getattr(args, "huggingface_api_key", None):
        kwargs["huggingface_api_key"] = args.huggingface_api_key
    if getattr(args, "image_model", None):
        kwargs["image_model"] = args.image_model
    if getattr(args, "image_prompt_type", None):
        kwargs["image_prompt_type"] = args.image_prompt_type
    if getattr(args, "image_custom_prompt", None):
        kwargs["image_custom_prompt"] = args.image_custom_prompt
    if getattr(args, "working_directory", None):
        kwargs["working_directory"] = args.working_directory
    if getattr(args, "email_for_abstracts", None):
        kwargs["email_for_abstracts"] = args.email_for_abstracts
    # Dataset/study sources (canonical name: sources)
    sources_tokens: List[str] = []
    values = getattr(args, "sources", None) or []
    for item in values:
        # Allow comma-separated items per token
        parts = [p.strip() for p in str(item).split(",")]
        sources_tokens.extend([p for p in parts if p])
    if sources_tokens:
        # de-duplicate while preserving order
        seen = set()
        ordered = []
        for s in sources_tokens:
            low = s  # Keep original case; normalization happens later
            if low not in seen:
                seen.add(low)
                ordered.append(s)
        kwargs["sources"] = ordered
    # Atlas selection
    atlas_names = getattr(args, "atlas_names", None)
    if atlas_names:
        names: List[str] = []
        atlas_configs: Dict[str, Dict[str, str]] = {}
        for item in atlas_names:
            parts = [p.strip() for p in str(item).split(",")]
            for part in parts:
                if not part:
                    continue
                names.append(part)
                if part not in kwargs.get("atlas_configs", {}):
                    source = _atlas_source_from_value(part)
                    if source:
                        atlas_configs.setdefault(part, {}).update(source)
        if names:
            kwargs["atlas_names"] = list(dict.fromkeys(names))
        if atlas_configs:
            kwargs["atlas_configs"] = atlas_configs
    atlas_urls = getattr(args, "atlas_urls", None)
    if atlas_urls:
        configs = kwargs.setdefault("atlas_configs", {})
        names = kwargs.setdefault("atlas_names", [])
        for entry in atlas_urls:
            if "=" not in entry:
                raise argparse.ArgumentTypeError("--atlas-url expects NAME=URL entries")
            name, url = entry.split("=", 1)
            name = name.strip()
            url = url.strip()
            if not name or not url:
                raise argparse.ArgumentTypeError("--atlas-url expects NAME=URL entries")
            configs.setdefault(name, {})["atlas_url"] = url
            if name not in names:
                names.append(name)
    atlas_files = getattr(args, "atlas_files", None)
    if atlas_files:
        configs = kwargs.setdefault("atlas_configs", {})
        names = kwargs.setdefault("atlas_names", [])
        for entry in atlas_files:
            if "=" not in entry:
                raise argparse.ArgumentTypeError(
                    "--atlas-file expects NAME=PATH entries"
                )
            name, path = entry.split("=", 1)
            name = name.strip()
            path = path.strip()
            if not name or not path:
                raise argparse.ArgumentTypeError(
                    "--atlas-file expects NAME=PATH entries"
                )
            configs.setdefault(name, {})["atlas_file"] = path
            if name not in names:
                names.append(name)
    if "atlas_names" in kwargs:
        kwargs["atlas_names"] = list(dict.fromkeys(kwargs["atlas_names"]))
    return kwargs


def _print_results(results):
    """Pretty-print pipeline results as JSON."""
    print(json.dumps([asdict(r) for r in results], indent=2))


def _add_coordinate_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("coords", nargs="*", help="Coordinates as x y z or x,y,z")
    p.add_argument("--coords-file", help="CSV/XLSX file with coordinates")


def _add_region_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("regions", nargs="+", help="Region names")


def _add_execution_options(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--output-format",
        choices=["json", "pickle", "csv", "pdf", "directory"],
        help="Export results to the chosen format",
    )
    p.add_argument(
        "--output-name",
        dest="output_name",
        help=(
            "File or directory name without path separators for exported "
            "results stored under the working directory"
        ),
    )
    p.add_argument("--batch-size", type=int, default=0, help="Batch size")
    p.add_argument(
        "--working-directory",
        dest="working_directory",
        help="Base working directory for caches and outputs",
    )


def _add_atlas_options(
    p: argparse.ArgumentParser,
    *,
    allow_multiple: bool = True,
    required: bool = False,
) -> None:
    if allow_multiple:
        atlas_help = (
            "Atlas name(s) to use (repeat --atlas or use comma-separated list). "
            "Defaults: harvard-oxford,juelich,aal"
        )
    else:
        atlas_help = "Atlas name to use for region lookups"
    p.add_argument(
        "--atlas",
        dest="atlas_names",
        action="append",
        required=required,
        help=atlas_help,
    )
    p.add_argument(
        "--atlas-url",
        dest="atlas_urls",
        action="append",
        help="Associate an atlas alias with a download URL (NAME=URL)",
    )
    p.add_argument(
        "--atlas-file",
        dest="atlas_files",
        action="append",
        help="Associate an atlas alias with a local file path (NAME=PATH)",
    )


def _add_study_options(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--sources",
        action="append",
        help=(
            "Datasets to use (repeat --sources or provide comma-separated list). "
            "Examples: neurosynth,neuroquery,nidm_pain"
        ),
    )
    p.add_argument(
        "--email-for-abstracts",
        help="Contact email used when querying study abstracts",
    )


def _add_llm_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--gemini-api-key", help="API key for Google Gemini provider")
    p.add_argument("--openrouter-api-key", help="API key for OpenRouter provider")
    p.add_argument("--openai-api-key", help="API key for OpenAI provider")
    p.add_argument("--anthropic-api-key", help="API key for Anthropic provider")
    p.add_argument("--huggingface-api-key", help="API key for Hugging Face provider")


def _add_image_options(
    p: argparse.ArgumentParser,
    *,
    default_backend: str = "ai",
    include_huggingface: bool = True,
    include_api_options: bool = True,
) -> None:
    if include_api_options:
        if include_huggingface:
            p.add_argument(
                "--huggingface-api-key", help="API key for Hugging Face provider"
            )
        p.add_argument("--openai-api-key", help="API key for OpenAI provider")
        p.add_argument("--anthropic-api-key", help="API key for Anthropic provider")
    p.add_argument("--image-model", default="stabilityai/stable-diffusion-2")
    p.add_argument(
        "--image-backend",
        choices=["ai", "nilearn", "both"],
        default=default_backend,
        help="Image generation backend",
    )
    p.add_argument(
        "--image-prompt-type",
        choices=[
            "anatomical",
            "functional",
            "schematic",
            "artistic",
            "custom",
            "default",
        ],
        help="Prompt template to use for AI image generation",
    )
    p.add_argument(
        "--image-custom-prompt",
        help="Custom image prompt template (use with --image-prompt-type custom)",
    )


def _format_cli_tokens(tokens: Sequence[str]) -> str:
    """Join CLI tokens into a shell-friendly command string."""
    return " ".join(shlex.quote(t) for t in tokens)


def _common_config_flags(
    cfg: dict,
    *,
    include_api: bool,
    include_sources: bool,
    include_atlas: bool,
    include_image_model: bool,
) -> List[str]:
    """Translate shared configuration values to CLI flags."""
    flags: List[str] = []
    working_dir = cfg.get("working_directory")
    if working_dir:
        flags.extend(["--working-directory", str(working_dir)])

    if include_api:
        mapping = {
            "gemini_api_key": "--gemini-api-key",
            "openrouter_api_key": "--openrouter-api-key",
            "openai_api_key": "--openai-api-key",
            "anthropic_api_key": "--anthropic-api-key",
            "huggingface_api_key": "--huggingface-api-key",
        }
        for key, flag in mapping.items():
            value = cfg.get(key)
            if value:
                flags.extend([flag, str(value)])
    else:
        # Image generation may still rely on Hugging Face
        if include_image_model:
            value = cfg.get("huggingface_api_key")
            if value:
                flags.extend(["--huggingface-api-key", str(value)])

    if include_sources:
        sources = cfg.get("sources") or []
        if isinstance(sources, list) and sources:
            flags.extend(["--sources", ",".join(str(s) for s in sources)])
        email = cfg.get("email_for_abstracts")
        if email:
            flags.extend(["--email-for-abstracts", str(email)])

    if include_atlas:
        atlas_names = cfg.get("atlas_names") or []
        for name in atlas_names:
            flags.extend(["--atlas", str(name)])

        atlas_configs = cfg.get("atlas_configs") or {}
        for name, options in atlas_configs.items():
            if not isinstance(options, dict):
                continue
            atlas_url = options.get("atlas_url")
            if atlas_url and atlas_url != name:
                flags.extend(["--atlas-url", f"{name}={atlas_url}"])
            atlas_file = options.get("atlas_file")
            if atlas_file and atlas_file != name:
                flags.extend(["--atlas-file", f"{name}={atlas_file}"])

    if include_image_model:
        image_model = cfg.get("image_model")
        if image_model:
            flags.extend(["--image-model", str(image_model)])

    return flags


def _inputs_to_tokens(input_type: str, inputs: Sequence) -> List[str]:
    def _format_value(value) -> str:
        if isinstance(value, numbers.Integral):
            return str(int(value))
        if isinstance(value, numbers.Real):
            as_float = float(value)
            if as_float.is_integer():
                return str(int(as_float))
            return str(as_float)
        return str(value)

    if input_type == "coords":
        tokens: List[str] = []
        for item in inputs:
            if isinstance(item, (list, tuple)):
                tokens.extend(_format_value(v) for v in item)
            else:
                tokens.append(_format_value(item))
        return tokens

    if input_type == "region_names":
        return [str(item) for item in inputs]

    raise ValueError(f"Dry-run not supported for input_type '{input_type}'")


def _commands_from_config(cfg: dict) -> List[str]:
    input_type = str(cfg.get("input_type", "coords")).lower()
    inputs = cfg.get("inputs", [])
    outputs = cfg.get("outputs", []) or []
    if not isinstance(outputs, list):
        raise ValueError("Config 'outputs' must be a list when using dry-run")

    config_section = cfg.get("config") or {}

    commands: List[str] = []
    base_tokens = ["coord2region"]

    coord_command_map = {
        frozenset(["region_labels"]): (
            "coords-to-atlas",
            dict(include_api=False, include_sources=False, include_image=False),
        ),
        frozenset(["region_labels", "raw_studies"]): (
            "coords-to-study",
            dict(include_api=False, include_sources=True, include_image=False),
        ),
        frozenset(["region_labels", "raw_studies", "summaries"]): (
            "coords-to-summary",
            dict(include_api=True, include_sources=True, include_image=False),
        ),
        frozenset(["region_labels", "raw_studies", "images"]): (
            "coords-to-image",
            dict(include_api=True, include_sources=True, include_image=True),
        ),
        frozenset(
            [
                "region_labels",
                "raw_studies",
                "summaries",
                "images",
            ]
        ): (
            "coords-to-insights",
            dict(include_api=True, include_sources=True, include_image=True),
        ),
    }

    region_command_map = {
        frozenset(["mni_coordinates"]): (
            "region-to-coords",
            dict(include_api=False, include_sources=False, include_image=False),
        ),
        frozenset(["mni_coordinates", "raw_studies"]): (
            "region-to-study",
            dict(include_api=False, include_sources=True, include_image=False),
        ),
        frozenset(
            [
                "mni_coordinates",
                "raw_studies",
                "summaries",
            ]
        ): (
            "region-to-summary",
            dict(include_api=True, include_sources=True, include_image=False),
        ),
        frozenset(
            [
                "mni_coordinates",
                "raw_studies",
                "images",
            ]
        ): (
            "region-to-image",
            dict(include_api=False, include_sources=True, include_image=True),
        ),
        frozenset(
            [
                "mni_coordinates",
                "raw_studies",
                "summaries",
                "images",
            ]
        ): (
            "region-to-insights",
            dict(include_api=True, include_sources=True, include_image=True),
        ),
    }

    if input_type == "coords":
        output_key = frozenset(str(o).lower() for o in outputs)
        if not output_key:
            raise ValueError("No supported outputs found for dry-run")
        if output_key not in coord_command_map:
            raise ValueError(
                "Dry-run does not support coords commands for outputs "
                f"{sorted(output_key)}"
            )
        command, capabilities = coord_command_map[output_key]
        coord_tokens = _inputs_to_tokens("coords", inputs)
        tokens = base_tokens + [command]
        tokens.extend(coord_tokens)

        shared_flags = _common_config_flags(
            config_section,
            include_api=capabilities["include_api"],
            include_sources=capabilities["include_sources"],
            include_atlas=True,
            include_image_model=capabilities["include_image"],
        )
        tokens.extend(shared_flags)

        if cfg.get("output_format"):
            tokens.extend(["--output-format", str(cfg["output_format"])])
        if cfg.get("output_name"):
            tokens.extend(["--output-name", str(cfg["output_name"])])

        image_backend = cfg.get("image_backend")
        if capabilities["include_image"] and image_backend:
            tokens.extend(["--image-backend", str(image_backend)])
        if capabilities["include_image"]:
            image_prompt_type = config_section.get("image_prompt_type")
            if image_prompt_type:
                tokens.extend(["--image-prompt-type", str(image_prompt_type)])
            image_custom_prompt = config_section.get("image_custom_prompt")
            if image_custom_prompt:
                tokens.extend(["--image-custom-prompt", str(image_custom_prompt)])

        commands.append(_format_cli_tokens(tokens))
        return commands

    if input_type == "region_names":
        output_key = frozenset(str(o).lower() for o in outputs)
        if not output_key:
            raise ValueError("No supported outputs found for dry-run")
        if output_key not in region_command_map:
            raise ValueError(
                "Dry-run does not support region commands for outputs "
                f"{sorted(output_key)}"
            )
        command, capabilities = region_command_map[output_key]
        atlas_names = config_section.get("atlas_names") or []
        if isinstance(atlas_names, list) and len(atlas_names) > 1:
            raise ValueError(
                "Dry-run region commands require exactly one atlas name in the config"
            )
        region_tokens = _inputs_to_tokens("region_names", inputs)
        tokens = base_tokens + [command]
        tokens.extend(region_tokens)

        shared_flags = _common_config_flags(
            config_section,
            include_api=capabilities["include_api"],
            include_sources=capabilities["include_sources"],
            include_atlas=True,
            include_image_model=capabilities["include_image"],
        )
        tokens.extend(shared_flags)

        if cfg.get("output_format"):
            tokens.extend(["--output-format", str(cfg["output_format"])])
        if cfg.get("output_name"):
            tokens.extend(["--output-name", str(cfg["output_name"])])

        image_backend = cfg.get("image_backend")
        if capabilities["include_image"] and image_backend:
            tokens.extend(["--image-backend", str(image_backend)])
        if capabilities["include_image"]:
            image_prompt_type = config_section.get("image_prompt_type")
            if image_prompt_type:
                tokens.extend(["--image-prompt-type", str(image_prompt_type)])
            image_custom_prompt = config_section.get("image_custom_prompt")
            if image_custom_prompt:
                tokens.extend(["--image-custom-prompt", str(image_custom_prompt)])

        commands.append(_format_cli_tokens(tokens))
        return commands

    raise ValueError(f"Dry-run not supported for input_type '{input_type}'")


def run_from_config(path: str, *, dry_run: bool = False) -> None:
    """Execute the pipeline using a YAML configuration file."""
    with open(path, "r", encoding="utf8") as f:
        raw_cfg = yaml.safe_load(f) or {}

    try:
        cfg = Coord2RegionConfig.model_validate(raw_cfg)
    except ValidationError as exc:
        for err in exc.errors():
            loc = "->".join(str(p) for p in err.get("loc", ()))
            msg = err.get("msg", "Invalid configuration value")
            print(f"Config error at {loc or '<root>'}: {msg}", file=sys.stderr)
        raise SystemExit(1) from exc

    inputs = cfg.collect_inputs(load_coords_file=_load_coords_file)
    runtime = cfg.to_pipeline_runtime(inputs)

    if dry_run:
        commands = _commands_from_config(runtime)
        for cmd in commands:
            print(cmd)
        return

    res = run_pipeline(**runtime)
    _print_results(res)


def create_parser() -> argparse.ArgumentParser:
    """Create the top-level argument parser for the CLI."""
    parser = argparse.ArgumentParser(prog="coord2region")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_package_version()}",
        help="Show the installed coord2region version and exit",
    )
    parser.add_argument(
        "--list-atlases",
        action="store_true",
        help="List bundled atlas identifiers and exit",
    )
    subparsers = parser.add_subparsers(dest="command")

    p_run = subparsers.add_parser(
        "run", help="Execute a pipeline described in a YAML config file"
    )
    p_run.add_argument("--config", required=True, help="YAML configuration file")
    p_run.add_argument(
        "--dry-run",
        action="store_true",
        help="Print equivalent CLI commands without executing",
    )

    # Coordinate commands
    p_atlas = subparsers.add_parser(
        "coords-to-atlas", help="Map coordinates to atlas regions"
    )
    _add_coordinate_arguments(p_atlas)
    _add_execution_options(p_atlas)
    _add_atlas_options(p_atlas)

    p_study = subparsers.add_parser(
        "coords-to-study", help="Retrieve studies for coordinates"
    )
    _add_coordinate_arguments(p_study)
    _add_execution_options(p_study)
    _add_atlas_options(p_study)
    _add_study_options(p_study)

    p_sum = subparsers.add_parser(
        "coords-to-summary",
        help="Generate summaries alongside atlas labels and studies",
    )
    _add_coordinate_arguments(p_sum)
    _add_execution_options(p_sum)
    _add_atlas_options(p_sum)
    _add_study_options(p_sum)
    _add_llm_options(p_sum)

    p_img = subparsers.add_parser(
        "coords-to-image", help="Generate images informed by studies"
    )
    _add_coordinate_arguments(p_img)
    _add_execution_options(p_img)
    _add_atlas_options(p_img)
    _add_study_options(p_img)
    _add_image_options(p_img, default_backend="nilearn")

    p_insights = subparsers.add_parser(
        "coords-to-insights", help="Produce labels, studies, summaries and images"
    )
    _add_coordinate_arguments(p_insights)
    _add_execution_options(p_insights)
    _add_atlas_options(p_insights)
    _add_study_options(p_insights)
    _add_llm_options(p_insights)
    _add_image_options(
        p_insights,
        default_backend="nilearn",
        include_huggingface=False,
        include_api_options=False,
    )

    # Region commands
    p_rtc = subparsers.add_parser(
        "region-to-coords", help="Convert region names to coordinates"
    )
    _add_region_arguments(p_rtc)
    _add_execution_options(p_rtc)
    _add_atlas_options(p_rtc, allow_multiple=False, required=True)

    p_rts = subparsers.add_parser(
        "region-to-study", help="Retrieve studies for regions"
    )
    _add_region_arguments(p_rts)
    _add_execution_options(p_rts)
    _add_atlas_options(p_rts, allow_multiple=False, required=True)
    _add_study_options(p_rts)

    p_rtsum = subparsers.add_parser(
        "region-to-summary",
        help="Generate summaries for regions based on related studies",
    )
    _add_region_arguments(p_rtsum)
    _add_execution_options(p_rtsum)
    _add_atlas_options(p_rtsum, allow_multiple=False, required=True)
    _add_study_options(p_rtsum)
    _add_llm_options(p_rtsum)

    p_rtimg = subparsers.add_parser(
        "region-to-image", help="Generate images for regions"
    )
    _add_region_arguments(p_rtimg)
    _add_execution_options(p_rtimg)
    _add_atlas_options(p_rtimg, allow_multiple=False, required=True)
    _add_study_options(p_rtimg)
    _add_image_options(p_rtimg, default_backend="nilearn")

    p_rtinsights = subparsers.add_parser(
        "region-to-insights", help="Combine coordinates, studies, summaries and images"
    )
    _add_region_arguments(p_rtinsights)
    _add_execution_options(p_rtinsights)
    _add_atlas_options(p_rtinsights, allow_multiple=False, required=True)
    _add_study_options(p_rtinsights)
    _add_llm_options(p_rtinsights)
    _add_image_options(
        p_rtinsights,
        default_backend="nilearn",
        include_huggingface=False,
        include_api_options=False,
    )

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the ``coord2region`` console script."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if getattr(args, "list_atlases", False):
        _print_available_atlases()
        return

    if not args.command:
        parser.print_help()
        return

    if args.command == "run":
        run_from_config(args.config, dry_run=getattr(args, "dry_run", False))
        return

    kwargs = _collect_kwargs(args)

    def _resolve_coords() -> List[List[float]]:
        coords: List[List[float]] = []
        coords_file = getattr(args, "coords_file", None)
        if coords_file:
            coords.extend(_load_coords_file(coords_file))
        coords.extend(_parse_coords_tokens(getattr(args, "coords", [])))
        if not coords:
            parser.error("No coordinates provided")
        return coords

    def _resolve_regions() -> List[str]:
        names = list(getattr(args, "regions", []) or [])
        if not names:
            parser.error("No region names provided")
        return names

    def _ensure_single_atlas(command_name: str) -> None:
        atlas_names = kwargs.get("atlas_names") or []
        if len(atlas_names) != 1:
            parser.error(
                f"{command_name} requires exactly one atlas specified via --atlas"
            )

    if args.command == "coords-to-atlas":
        coords = _resolve_coords()
        for batch in _batch(coords, args.batch_size):
            res = run_pipeline(
                batch,
                "coords",
                ["region_labels"],
                args.output_format,
                args.output_name,
                config=kwargs,
            )
            _print_results(res)
    elif args.command == "coords-to-study":
        coords = _resolve_coords()
        for batch in _batch(coords, args.batch_size):
            res = run_pipeline(
                batch,
                "coords",
                ["region_labels", "raw_studies"],
                args.output_format,
                args.output_name,
                config=kwargs,
            )
            _print_results(res)
    elif args.command == "coords-to-summary":
        coords = _resolve_coords()
        for batch in _batch(coords, args.batch_size):
            res = run_pipeline(
                batch,
                "coords",
                ["region_labels", "raw_studies", "summaries"],
                args.output_format,
                args.output_name,
                config=kwargs,
            )
            _print_results(res)
    elif args.command == "coords-to-image":
        coords = _resolve_coords()
        backend = getattr(args, "image_backend", "nilearn")
        for batch in _batch(coords, args.batch_size):
            res = run_pipeline(
                batch,
                "coords",
                ["region_labels", "raw_studies", "images"],
                args.output_format,
                args.output_name,
                image_backend=backend,
                config=kwargs,
            )
            _print_results(res)
    elif args.command == "coords-to-insights":
        coords = _resolve_coords()
        backend = getattr(args, "image_backend", "nilearn")
        for batch in _batch(coords, args.batch_size):
            res = run_pipeline(
                batch,
                "coords",
                ["region_labels", "raw_studies", "summaries", "images"],
                args.output_format,
                args.output_name,
                image_backend=backend,
                config=kwargs,
            )
            _print_results(res)
    elif args.command == "region-to-coords":
        _ensure_single_atlas("region-to-coords")
        names = _resolve_regions()
        for batch in _batch(names, args.batch_size):
            res = run_pipeline(
                batch,
                "region_names",
                ["mni_coordinates"],
                args.output_format,
                args.output_name,
                config=kwargs,
            )
            _print_results(res)
    elif args.command == "region-to-study":
        _ensure_single_atlas("region-to-study")
        names = _resolve_regions()
        for batch in _batch(names, args.batch_size):
            res = run_pipeline(
                batch,
                "region_names",
                ["mni_coordinates", "raw_studies"],
                args.output_format,
                args.output_name,
                config=kwargs,
            )
            _print_results(res)
    elif args.command == "region-to-summary":
        _ensure_single_atlas("region-to-summary")
        names = _resolve_regions()
        for batch in _batch(names, args.batch_size):
            res = run_pipeline(
                batch,
                "region_names",
                ["mni_coordinates", "raw_studies", "summaries"],
                args.output_format,
                args.output_name,
                config=kwargs,
            )
            _print_results(res)
    elif args.command == "region-to-image":
        _ensure_single_atlas("region-to-image")
        names = _resolve_regions()
        backend = getattr(args, "image_backend", "nilearn")
        for batch in _batch(names, args.batch_size):
            res = run_pipeline(
                batch,
                "region_names",
                ["mni_coordinates", "raw_studies", "images"],
                args.output_format,
                args.output_name,
                image_backend=backend,
                config=kwargs,
            )
            _print_results(res)
    elif args.command == "region-to-insights":
        _ensure_single_atlas("region-to-insights")
        names = _resolve_regions()
        backend = getattr(args, "image_backend", "nilearn")
        for batch in _batch(names, args.batch_size):
            res = run_pipeline(
                batch,
                "region_names",
                ["mni_coordinates", "raw_studies", "summaries", "images"],
                args.output_format,
                args.output_name,
                image_backend=backend,
                config=kwargs,
            )
            _print_results(res)


if __name__ == "__main__":  # pragma: no cover
    main()
