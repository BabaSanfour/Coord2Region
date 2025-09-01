"""High-level path helpers for coord2region."""

from __future__ import annotations

from .utils.paths import resolve_data_dir


def get_data_directory(base: str | None = None) -> str:
    """Return absolute path to the coord2region data directory.

    Parameters
    ----------
    base : str, optional
        Base directory supplied by the user.  If ``None`` (default) the
        path ``~/coord2region`` is used.  Relative paths are interpreted
        relative to the user's home directory.

    Returns
    -------
    str
        Absolute path to the data directory.  The directory is created if
        it does not already exist.
    """
    path = resolve_data_dir(base)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
