"""High-level path helpers for coord2region."""

from __future__ import annotations

from .utils.paths import resolve_working_directory


def get_working_directory(base: str | None = None) -> str:
    """Return absolute path to the coord2region working directory.

    Parameters
    ----------
    base : str, optional
        Base directory supplied by the user.  If ``None`` (default) the
        path ``~/coord2region`` is used.  Relative paths are interpreted
        relative to the user's home directory.

    Returns
    -------
    str
        Absolute path to the working directory.  The directory is created if
        it does not already exist.
    """
    path = resolve_working_directory(base)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
