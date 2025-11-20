#!/usr/bin/env python
"""Interactive helper for generating Coord2Region configuration files.

This script reads `config/coord2region-config.template.yaml`, prompts the user
for each declared field, and writes the answers to `config/coord2region-config.yaml`
by default. Existing values are preserved unless explicitly overwritten.

Example usage:
    python scripts/configure_coord2region.py
    python scripts/configure_coord2region.py --non-interactive --force
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Read a YAML mapping from the provided path.

    Parameters
    ----------
    path : Path
        Path to the template YAML file.

    Returns
    -------
    dict
        Parsed YAML mapping extracted from the file.

    Raises
    ------
    FileNotFoundError
        When the template file does not exist.
    ValueError
        If the YAML root is not a mapping.
    """
    if not path.exists():
        raise FileNotFoundError(f"Template file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Template file {path} must contain a YAML mapping.")
    return data


def _coerce_bool(value: Any) -> bool:
    """Interpret a YAML field as a boolean value.

    Parameters
    ----------
    value : Any
        Value extracted from the template metadata.

    Returns
    -------
    bool
        ``True`` if the value is truthy, ``False`` otherwise.
    """
    return bool(value)


def _prompt(
    name: str,
    settings: Dict[str, Any],
    *,
    current: Optional[str],
) -> Optional[str]:
    """Prompt the user for a single configuration value.

    Parameters
    ----------
    name : str
        Key name of the configuration entry.
    settings : dict
        Metadata extracted from the template describing prompts/defaults.
    current : str or None
        Value read from an existing configuration file, if any.

    Returns
    -------
    str or None
        User-provided value, default, or ``None`` if omitted and not required.
    """
    prompt_text = settings.get("prompt") or name
    default = current or settings.get("default")
    help_text = settings.get("help")
    required = _coerce_bool(settings.get("required"))
    secret = _coerce_bool(settings.get("secret"))

    if help_text:
        print(help_text)
    if default:
        prompt_text = f"{prompt_text} [{default}]"
    prompt_text += ": "

    while True:
        try:
            if secret:
                value = getpass.getpass(prompt_text)
            else:
                value = input(prompt_text)
        except (EOFError, KeyboardInterrupt):
            print("\nAborted by user.")
            sys.exit(1)
        value = value.strip()
        if not value:
            if current:
                return current
            if default not in (None, ""):
                return str(default)
            if required:
                print("This value is required. Press Ctrl+C to abort.")
                continue
            return None
        return value


def build_configuration(
    template: Dict[str, Any],
    *,
    existing_values: Dict[str, Any],
    interactive: bool,
) -> Dict[str, Any]:
    """Resolve a configuration dictionary from the template and existing values.

    Parameters
    ----------
    template : dict
        Parsed template describing the expected environment variables.
    existing_values : dict
        Values already present in an existing configuration file.
    interactive : bool
        Whether prompts should be presented to the user.

    Returns
    -------
    dict
        Final configuration ready to be dumped to YAML.

    Raises
    ------
    ValueError
        When the template's ``environment`` section is malformed or required
        fields are missing in non-interactive mode.
    """
    env_template = template.get("environment") or {}
    if not isinstance(env_template, dict):
        raise ValueError("Template 'environment' section must be a mapping.")

    environment: Dict[str, str] = {}
    for key, meta in env_template.items():
        meta = meta or {}
        if not isinstance(meta, dict):
            raise ValueError(f"Template entry for {key!r} must be a mapping.")
        current_value = existing_values.get(key)

        if interactive:
            value = _prompt(key, meta, current=current_value)
        else:
            value = current_value or meta.get("default")
            if value is not None:
                value = str(value).strip()
            if not value and meta.get("required"):
                raise ValueError(
                    f"Field {key!r} is required but missing in non-interactive mode."
                )

        if value is None:
            continue
        value = str(value).strip()
        if value:
            environment[key] = value

    notes = template.get("notes")
    config: Dict[str, Any] = {"environment": environment}
    if notes:
        config["notes"] = notes
    return config


def write_configuration(path: Path, data: Dict[str, Any]) -> None:
    """Write the resolved configuration to disk with header guidance.

    Parameters
    ----------
    path : Path
        Output location for the configuration file.
    data : dict
        Configuration mapping to dump as YAML.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "# This file was generated by scripts/configure_coord2region.py\n"
    header += "# Keep it private and never commit it to version control.\n"
    yaml_text = yaml.safe_dump(data, sort_keys=True, allow_unicode=False)
    path.write_text(header + yaml_text, encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    """Script entry point, parsing arguments and driving the configuration flow.

    Parameters
    ----------
    argv : list of str, optional
        Optional argument list to parse (defaults to ``sys.argv``).

    Returns
    -------
    int
        Exit code, ``0`` on success.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template",
        default="config/coord2region-config.template.yaml",
        help="Path to the configuration template.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Custom output path. Defaults to the template's 'output' value.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip prompts and use defaults + existing values instead.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file without confirmation.",
    )
    args = parser.parse_args(argv)

    template_path = Path(args.template)
    template = _load_yaml(template_path)

    output_name = template.get("output", "coord2region-config.yaml")
    output_path = (
        Path(args.output) if args.output else template_path.parent / output_name
    )

    existing_env: Dict[str, Any] = {}
    if output_path.exists():
        try:
            existing_data = (
                yaml.safe_load(output_path.read_text(encoding="utf-8")) or {}
            )
            if isinstance(existing_data, dict):
                existing_env = existing_data.get("environment") or {}
                if not isinstance(existing_env, dict):
                    existing_env = {}
        except Exception:
            if not args.force:
                raise
            existing_env = {}

        if not args.force and not args.non_interactive:
            reply = input(
                f"{output_path} already exists. Overwrite values while preserving "
                "existing defaults? [y/N]: "
            ).strip()
            if reply.lower() not in {"y", "yes"}:
                print("Aborting. No changes made.")
                return 0

    config = build_configuration(
        template,
        existing_values=existing_env,
        interactive=not args.non_interactive,
    )

    write_configuration(output_path, config)
    print(f"Configuration written to {output_path}")
    print("Ensure this file stays out of version control (see .gitignore).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
