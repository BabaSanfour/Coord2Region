"""Command-line interface for Coord2Region.

Enhancements:
- Accept coordinate triples as separate numbers (e.g. ``30 -22 50``).
- Add ``--atlas`` option (repeatable / comma-separated) to choose atlas names.
- Add ``--image-backend`` option for image generation.
- Add common options like ``--data-dir`` and ``--email-for-abstracts``.
"""

import argparse
import json
import os
import shlex
import sys
from dataclasses import asdict
from typing import Iterable, List, Sequence, Dict, Optional
import numbers

import pandas as pd
import yaml

from .pipeline import run_pipeline
from .config import Coord2RegionConfig, ValidationError


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
    if getattr(args, "working_directory", None):
        kwargs["working_directory"] = args.working_directory
    if getattr(args, "email_for_abstracts", None):
        kwargs["email_for_abstracts"] = args.email_for_abstracts
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


def _format_cli_tokens(tokens: Sequence[str]) -> str:
    """Join CLI tokens into a shell-friendly command string."""
    return " ".join(shlex.quote(t) for t in tokens)


def _common_config_flags(cfg: dict) -> List[str]:
    """Translate shared configuration values to CLI flags."""
    flags: List[str] = []
    mapping = {
        "gemini_api_key": "--gemini-api-key",
        "openrouter_api_key": "--openrouter-api-key",
        "openai_api_key": "--openai-api-key",
        "anthropic_api_key": "--anthropic-api-key",
        "huggingface_api_key": "--huggingface-api-key",
        "working_directory": "--working-directory",
        "email_for_abstracts": "--email-for-abstracts",
    }
    for key, flag in mapping.items():
        value = cfg.get(key)
        if value:
            flags.extend([flag, str(value)])

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

    batch_size = cfg.get("batch_size")
    if batch_size:
        flags.extend(["--batch-size", str(batch_size)])

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
    shared_flags = _common_config_flags(config_section)

    commands: List[str] = []
    base_tokens = ["coord2region"]

    if input_type == "region_names":
        tokens = base_tokens + ["region-to-coords"]
        tokens.extend(_inputs_to_tokens(input_type, inputs))
        tokens.extend(shared_flags)
        commands.append(_format_cli_tokens(tokens))
        return commands

    if input_type != "coords":
        raise ValueError(f"Dry-run not supported for input_type '{input_type}'")

    coord_tokens = _inputs_to_tokens("coords", inputs)
    output_map = {
        "region_labels": "coords-to-atlas",
        "summaries": "coords-to-summary",
        "images": "coords-to-image",
    }

    selected_outputs = []
    for output in outputs:
        key = str(output).lower()
        if key in output_map:
            selected_outputs.append((key, output_map[key]))
        elif key == "raw_studies":
            raise ValueError("Dry-run does not currently support 'raw_studies' output")
        else:
            raise ValueError(f"Unknown output '{output}' in config")

    if not selected_outputs:
        raise ValueError("No supported outputs found for dry-run")

    include_export_flags = len(selected_outputs) == 1
    export_flags: List[str] = []
    if include_export_flags:
        if cfg.get("output_format"):
            export_flags.extend(["--output-format", str(cfg["output_format"])])
        if cfg.get("output_name"):
            export_flags.extend(["--output-name", str(cfg["output_name"])])

    image_backend = cfg.get("image_backend")

    for key, command in selected_outputs:
        tokens = base_tokens + [command]
        tokens.extend(coord_tokens)
        tokens.extend(shared_flags)
        if include_export_flags:
            tokens.extend(export_flags)
        if key == "images":
            if image_backend:
                tokens.extend(["--image-backend", str(image_backend)])
            image_model = config_section.get("image_model")
            if image_model:
                tokens.extend(["--image-model", str(image_model)])
        commands.append(_format_cli_tokens(tokens))

    return commands


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

    def add_common(p: argparse.ArgumentParser) -> None:
        # Provider/API configuration
        p.add_argument("--gemini-api-key", help="API key for Google Gemini provider")
        p.add_argument("--openrouter-api-key", help="API key for OpenRouter provider")
        p.add_argument("--openai-api-key", help="API key for OpenAI provider")
        p.add_argument("--anthropic-api-key", help="API key for Anthropic provider")
        p.add_argument(
            "--huggingface-api-key", help="API key for Hugging Face provider"
        )

        # IO & batching
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

        # Datasets & atlas options
        p.add_argument(
            "--email-for-abstracts",
            help="Contact email used when querying study abstracts",
        )
        p.add_argument(
            "--atlas",
            dest="atlas_names",
            action="append",
            help=(
                "Atlas name(s) to use (repeat --atlas or use comma-separated list). "
                "Defaults: harvard-oxford,juelich,aal"
            ),
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

    p_sum = subparsers.add_parser(
        "coords-to-summary", help="Generate summaries for coordinates"
    )
    p_sum.add_argument("coords", nargs="*", help="Coordinates as x y z or x,y,z")
    p_sum.add_argument("--coords-file", help="CSV/XLSX file with coordinates")
    add_common(p_sum)

    p_atlas = subparsers.add_parser(
        "coords-to-atlas", help="Map coordinates to atlas regions"
    )
    p_atlas.add_argument("coords", nargs="*", help="Coordinates as x y z or x,y,z")
    p_atlas.add_argument("--coords-file", help="CSV/XLSX file with coordinates")
    add_common(p_atlas)

    p_img = subparsers.add_parser(
        "coords-to-image", help="Generate images for coordinates"
    )
    p_img.add_argument("coords", nargs="*", help="Coordinates as x y z or x,y,z")
    p_img.add_argument("--coords-file", help="CSV/XLSX file with coordinates")
    p_img.add_argument("--image-model", default="stabilityai/stable-diffusion-2")
    p_img.add_argument(
        "--image-backend",
        choices=["ai", "nilearn", "both"],
        default="ai",
        help="Image generation backend",
    )
    add_common(p_img)

    p_rtc = subparsers.add_parser(
        "region-to-coords", help="Convert region names to coordinates"
    )
    p_rtc.add_argument("regions", nargs="+", help="Region names")
    add_common(p_rtc)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the ``coord2region`` console script."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return

    if args.command == "run":
        run_from_config(args.config, dry_run=getattr(args, "dry_run", False))
        return

    kwargs = _collect_kwargs(args)

    if args.command == "coords-to-summary":
        coords: List[List[float]] = []
        if args.coords_file:
            coords.extend(_load_coords_file(args.coords_file))
        coords.extend(_parse_coords_tokens(args.coords))
        if not coords:
            parser.error("No coordinates provided")
        for batch in _batch(coords, args.batch_size):
            res = run_pipeline(
                batch,
                "coords",
                ["summaries"],
                args.output_format,
                args.output_name,
                config=kwargs,
            )
            _print_results(res)
    elif args.command == "coords-to-atlas":
        coords = []
        if args.coords_file:
            coords.extend(_load_coords_file(args.coords_file))
        coords.extend(_parse_coords_tokens(args.coords))
        if not coords:
            parser.error("No coordinates provided")
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
    elif args.command == "coords-to-image":
        coords = []
        if args.coords_file:
            coords.extend(_load_coords_file(args.coords_file))
        coords.extend(_parse_coords_tokens(args.coords))
        if not coords:
            parser.error("No coordinates provided")
        for batch in _batch(coords, args.batch_size):
            res = run_pipeline(
                batch,
                "coords",
                ["images"],
                args.output_format,
                args.output_name,
                image_backend=getattr(args, "image_backend", "ai"),
                config=kwargs,
            )
            _print_results(res)
    elif args.command == "region-to-coords":
        names = args.regions
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


if __name__ == "__main__":  # pragma: no cover
    main()
