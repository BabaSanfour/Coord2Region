"""Working directory helpers for Coord2Region projects."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, os.PathLike, None]


def resolve_working_directory(base: PathLike = None) -> Path:
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


def ensure_mne_data_directory(base: PathLike = None) -> Path:
    """Ensure the MNE data directory exists and is registered with MNE."""
    import mne

    base_dir = resolve_working_directory(base)

    candidates = []
    env_candidate = os.environ.get("MNE_DATA")
    if env_candidate:
        candidates.append(env_candidate)
    try:
        config_candidate = mne.get_config("MNE_DATA", None)
    except Exception:  # pragma: no cover - defensive
        config_candidate = None
    if config_candidate:
        candidates.append(config_candidate)

    target: Optional[Path] = None
    for candidate in candidates:
        candidate_path = Path(candidate).expanduser()
        if not candidate_path.is_absolute():
            candidate_path = (base_dir / candidate_path).resolve()
        target = candidate_path
        break

    if target is None:
        target = (base_dir / "mne_data").resolve()

    try:
        os.makedirs(target, exist_ok=True)
    except OSError as exc:  # pragma: no cover - filesystem permissions
        raise ValueError(f"Cannot create MNE data directory at {target}: {exc}")

    current_env = os.environ.get("MNE_DATA")
    if current_env != str(target):
        os.environ["MNE_DATA"] = str(target)

    try:
        current_config = mne.get_config("MNE_DATA", None)
    except Exception:  # pragma: no cover - defensive
        current_config = None

    if current_config != str(target):
        mne.utils.set_config("MNE_DATA", str(target), set_env=True)

    return target
