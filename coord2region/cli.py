"""Command-line interface for Coord2Region."""

import argparse
import json
from dataclasses import asdict
from typing import Iterable, List, Sequence

import pandas as pd
import yaml

from .pipeline import run_pipeline


def _parse_coord(text: str) -> List[float]:
    """Parse a coordinate string of the form 'x,y,z' or 'x y z'."""
    parts = text.replace(",", " ").split()
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Coordinates must have three values")
    try:
        return [float(p) for p in parts]
    except ValueError as exc:  # pragma: no cover - user input
        raise argparse.ArgumentTypeError(str(exc)) from exc


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


def _collect_kwargs(args: argparse.Namespace) -> dict:
    """Collect keyword arguments for :func:`run_pipeline` from parsed args."""
    kwargs = {}
    if getattr(args, "gemini_api_key", None):
        kwargs["gemini_api_key"] = args.gemini_api_key
    if getattr(args, "openrouter_api_key", None):
        kwargs["openrouter_api_key"] = args.openrouter_api_key
    if getattr(args, "image_model", None):
        kwargs["image_model"] = args.image_model
    return kwargs


def _print_results(results):
    """Pretty-print pipeline results as JSON."""
    print(json.dumps([asdict(r) for r in results], indent=2))


def run_from_config(path: str) -> None:
    """Execute the pipeline using a YAML configuration file."""
    with open(path, "r", encoding="utf8") as f:
        cfg = yaml.safe_load(f) or {}
    res = run_pipeline(
        cfg.get("inputs", []),
        cfg.get("input_type", "coords"),
        cfg.get("outputs", []),
        cfg.get("output_format"),
        cfg.get("output_path"),
        config=cfg.get("config"),
    )
    _print_results(res)


def create_parser() -> argparse.ArgumentParser:
    """Create the top-level argument parser for the CLI."""
    parser = argparse.ArgumentParser(prog="coord2region")
    parser.add_argument("--config", help="YAML configuration file")
    subparsers = parser.add_subparsers(dest="command")

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--gemini-api-key")
        p.add_argument("--openrouter-api-key")
        p.add_argument(
            "--output-format",
            choices=["json", "pickle", "csv", "pdf", "directory"],
        )
        p.add_argument("--output-path")
        p.add_argument("--batch-size", type=int, default=0)

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

    if args.config:
        run_from_config(args.config)
        return

    if not args.command:
        parser.print_help()
        return

    kwargs = _collect_kwargs(args)

    if args.command == "coords-to-summary":
        coords: List[List[float]] = []
        if args.coords_file:
            coords.extend(_load_coords_file(args.coords_file))
        coords.extend([_parse_coord(c) for c in args.coords])
        if not coords:
            parser.error("No coordinates provided")
        for batch in _batch(coords, args.batch_size):
            res = run_pipeline(
                batch,
                "coords",
                ["summaries"],
                args.output_format,
                args.output_path,
                config=kwargs,
            )
            _print_results(res)
    elif args.command == "coords-to-atlas":
        coords = []
        if args.coords_file:
            coords.extend(_load_coords_file(args.coords_file))
        coords.extend([_parse_coord(c) for c in args.coords])
        if not coords:
            parser.error("No coordinates provided")
        for batch in _batch(coords, args.batch_size):
            res = run_pipeline(
                batch,
                "coords",
                ["region_labels"],
                args.output_format,
                args.output_path,
                config=kwargs,
            )
            _print_results(res)
    elif args.command == "coords-to-image":
        coords = []
        if args.coords_file:
            coords.extend(_load_coords_file(args.coords_file))
        coords.extend([_parse_coord(c) for c in args.coords])
        if not coords:
            parser.error("No coordinates provided")
        for batch in _batch(coords, args.batch_size):
            res = run_pipeline(
                batch,
                "coords",
                ["images"],
                args.output_format,
                args.output_path,
                config=kwargs,
            )
            _print_results(res)
    elif args.command == "region-to-coords":
        names = args.regions
        for batch in _batch(names, args.batch_size):
            res = run_pipeline(
                batch,
                "region_names",
                [],
                args.output_format,
                args.output_path,
                config=kwargs,
            )
            _print_results(res)


if __name__ == "__main__":  # pragma: no cover
    main()
