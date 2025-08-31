"""Path helpers for coord2region.

This module centralises logic for resolving the base data directory used by
coord2region. The :func:`resolve_data_dir` function expands a user supplied
path or falls back to ``~/coord2region`` when ``None``. Relative paths are
interpreted relative to the user's home directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def resolve_data_dir(base: Optional[str] = None) -> Path:
    """Return the directory for coord2region data.

    Parameters
    ----------
    base : str or None, optional
        User supplied base directory.  If ``None`` (default) the path
        ``~/coord2region`` is returned.  Relative paths are interpreted
        relative to the user's home directory.

    Returns
    -------
    pathlib.Path
        Absolute path to the base directory.
    """
    if base is None:
        return Path.home() / "coord2region"

    p = Path(base).expanduser()
    if p.is_absolute():
        return p
    return Path.home() / p
