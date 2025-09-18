"""Working directory helpers for Coord2Region projects."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def resolve_working_directory(base: Optional[str] = None) -> Path:
    """Return the root working directory for coord2region assets.

    Parameters
    ----------
    base : str or None, optional
        User supplied base directory.  If ``None`` (default) the path
        ``~/coord2region`` is returned.  Relative paths are interpreted
        relative to the user's home directory.

    Returns
    -------
    pathlib.Path
        Absolute path to the working directory.
    """
    if base is None:
        return Path.home() / "coord2region"

    try:
        expanded = Path(base).expanduser()
        expanded.resolve()
    except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover
        # platform dependent
        raise ValueError(f"Invalid path: {base}") from exc

    if expanded.is_absolute():
        return expanded.resolve()
    return (Path.home() / expanded).resolve()
