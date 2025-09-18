"""Coordinate-to-region mapping utilities.

This module provides classes and helper functions for converting between
MNI coordinates, voxel indices, and anatomical region labels. It enables
lookups and transformations across multiple brain atlases.
"""

import logging
import pickle
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import mne
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .fetching import AtlasFetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _mni_to_tal(coords: Union[List[float], np.ndarray]) -> np.ndarray:
    """Convert MNI coordinates to Talairach space.

    Parameters
    ----------
    coords : array-like, shape (..., 3)
        Coordinates defined in MNI space. The input is reshaped to ``(-1, 3)``
        for transformation and the original shape is restored in the output.

    Returns
    -------
    np.ndarray
        Coordinates in Talairach space with the same shape as ``coords``.

    Examples
    --------
    >>> _mni_to_tal([0, 0, 0])
    array([0., 0., 0.])
    """
    coords = np.asarray(coords, dtype=float)
    orig_shape = coords.shape
    coords = coords.reshape(-1, 3)
    A = np.array(
        [[0.99, 0, 0, 0], [0, 0.9688, 0.0460, 0], [0, -0.0485, 0.9189, 0], [0, 0, 0, 1]]
    )
    B = np.array(
        [[0.99, 0, 0, 0], [0, 0.9688, 0.0420, 0], [0, -0.0485, 0.8390, 0], [0, 0, 0, 1]]
    )
    out = np.empty_like(coords)
    for i, c in enumerate(coords):
        vec = np.append(c, 1)
        out[i] = (A @ vec)[:3] if c[2] >= 0 else (B @ vec)[:3]
    return out.reshape(orig_shape)


def _tal_to_mni(coords: Union[List[float], np.ndarray]) -> np.ndarray:
    """Convert Talairach coordinates to MNI space.

    Parameters
    ----------
    coords : array-like, shape (..., 3)
        Coordinates defined in Talairach space. The input is reshaped to
        ``(-1, 3)`` for transformation and the original shape is restored in
        the output.

    Returns
    -------
    np.ndarray
        Coordinates in MNI space with the same shape as ``coords``.

    Examples
    --------
    >>> _tal_to_mni([0, 0, 0])
    array([0., 0., 0.])
    """
    coords = np.asarray(coords, dtype=float)
    orig_shape = coords.shape
    coords = coords.reshape(-1, 3)
    A = np.array(
        [[0.99, 0, 0, 0], [0, 0.9688, 0.0460, 0], [0, -0.0485, 0.9189, 0], [0, 0, 0, 1]]
    )
    B = np.array(
        [[0.99, 0, 0, 0], [0, 0.9688, 0.0420, 0], [0, -0.0485, 0.8390, 0], [0, 0, 0, 1]]
    )
    A_inv = np.linalg.inv(A)
    B_inv = np.linalg.inv(B)
    out = np.empty_like(coords)
    for i, c in enumerate(coords):
        vec = np.append(c, 1)
        out[i] = (A_inv @ vec)[:3] if c[2] >= 0 else (B_inv @ vec)[:3]
    return out.reshape(orig_shape)


_TRANSFORMS = {
    ("mni", "tal"): _mni_to_tal,
    ("tal", "mni"): _tal_to_mni,
}


def _get_numeric_hemi(hemi: Union[str, int]) -> int:
    """Convert hemisphere string to numeric code (0 or 1)."""
    if isinstance(hemi, int):
        return hemi
    if hemi is None:
        return None
    if isinstance(hemi, str):
        if hemi.lower() in ("l", "lh", "left"):
            return 0
        if hemi.lower() in ("r", "rh", "right"):
            return 1
    raise ValueError("Invalid hemisphere value. Use 'L', 'R', 'LH', 'RH', 0, or 1.")


class AtlasMapper:
    """Stores a single atlas and provides coordinate conversions.

    The atlas may be volumetric (a 3D numpy array with an associated 4x4 affine)
    or surface-based (a vertices array). In either case, the mapper supports
    conversions between coordinates, voxel indices, and region labels.

    Parameters
    ----------
    name : str
        Identifier for the atlas (e.g., "aal" or "brodmann").
    vol : np.ndarray
        A 3D numpy array representing the volumetric atlas.
    hdr : np.ndarray
        A 4x4 affine transform mapping voxel indices to MNI/world coordinates.
    labels : dict or list or None, optional
        Region labels. If a dict, keys should be strings for numeric indices and
        values are region names. If a list/array, it should match ``indexes``.
    indexes : list or np.ndarray or None, optional
        Region indices corresponding to ``labels``. Not needed if ``labels`` is
        a dict.
    regions : dict or None, optional
        For surface atlases, mapping of region names to vertex indices.
    system : str, optional
        The anatomical coordinate space (e.g., "mni" or "tal").

    Attributes
    ----------
    name : str
        Atlas identifier.
    vol : np.ndarray
        Volumetric atlas array.
    hdr : np.ndarray
        Affine transform mapping voxel indices to MNI/world coordinates.
    labels : dict or list or None
        Region labels.
    indexes : list or np.ndarray or None
        Region indices corresponding to labels.
    regions : dict or None
        Mapping of region names to vertex indices for surface atlases.
    system : str
        Anatomical coordinate space.
    shape : tuple
        Shape of the volumetric atlas.
    """

    def __init__(
        self,
        name: str,
        vol: np.ndarray,
        hdr: np.ndarray,
        labels: Optional[Union[Dict[str, str], List[str], np.ndarray]] = None,
        indexes: Optional[Union[List[int], np.ndarray]] = None,
        subject: Optional[str] = "fsaverage",
        regions: Optional[Dict[str, np.ndarray]] = None,
        subjects_dir: Optional[str] = None,
        system: str = "mni",
    ) -> None:
        self.name = name
        self.labels = labels
        self.indexes = indexes
        # Ensure region->vertex mapping uses integer vertex indices
        if regions is not None:
            self.regions = {
                key: np.asarray(vals, dtype=int).ravel()
                for key, vals in regions.items()
            }
        else:
            self.regions = None
        self.vertex_to_region = None
        self.system = system

        # Basic shape checks
        if isinstance(vol, np.ndarray):
            self.vol = np.asarray(vol)
            # volumetric atlas
            if hdr is not None and self.vol.ndim == 3:
                self.hdr = np.asarray(hdr)
                if self.hdr.shape != (4, 4):
                    raise ValueError("`hdr` must be a 4x4 transform matrix.")
                self.shape = self.vol.shape
                self.atlas_type = "volume"
            # coordinate atlas (list of region centroids)
            elif self.vol.ndim == 2 and self.vol.shape[1] == 3:
                self.hdr = None
                self.atlas_type = "coords"
                if self.indexes is None:
                    self.indexes = np.arange(self.vol.shape[0])
            else:
                raise ValueError("Unsupported array format for `vol`.")
        elif isinstance(vol, list):
            arr = np.asarray(vol)
            if arr.ndim == 2 and arr.shape[1] == 3:
                # coordinate atlas provided as list
                self.vol = arr.astype(float)
                self.hdr = None
                self.atlas_type = "coords"
                if self.indexes is None:
                    self.indexes = np.arange(self.vol.shape[0])
            else:
                # For surface atlases, `vol` is a list of vertex arrays per hemisphere
                self.vol = [np.asarray(v, dtype=int) for v in vol]
                self.hdr = None
                self.atlas_type = "surface"
                self.subject = subject
                self.subjects_dir = subjects_dir
                self.vertex_to_region = {
                    int(v): k
                    for k, verts in (regions or {}).items()
                    for v in np.asarray(verts).ravel()
                }

        # If labels is a dict, prepare an inverse mapping:
        #   region_name -> region_index
        if isinstance(self.labels, dict):
            self._label2index = {v: k for k, v in self.labels.items()}
        else:
            self._label2index = None

        # Cache for region centroids (used by nearest-region queries)
        self._centroids_cache: Optional[Dict[int, np.ndarray]] = None

        # Cached KD-tree for voxel center lookup (volume atlases)
        self._voxel_kdtree: Optional[cKDTree] = None
        self._voxel_indices: Optional[np.ndarray] = None

    # -------------------------------------------------------------------------
    # Internal lookups (private)
    # -------------------------------------------------------------------------

    def _lookup_region_name(self, value: Union[int, str]) -> str:
        """
        Return the region name corresponding to the given region index (int/str).

        Returns "Unknown" if not found.
        """
        if not isinstance(value, (int, str)):
            raise ValueError("value must be int or str")

        if self.atlas_type == "surface" and self.vertex_to_region is not None:
            try:
                return self.vertex_to_region.get(int(value), "Unknown")
            except ValueError:
                return "Unknown"

        value_str = str(value)
        if isinstance(self.labels, dict):
            return self.labels.get(value_str, "Unknown")

        if self.indexes is not None and self.labels is not None:
            try:
                if isinstance(self.indexes, list):
                    pos = self.indexes.index(int(value))
                else:
                    pos = int(np.where(self.indexes == int(value))[0][0])
                return self.labels[pos]
            except (ValueError, IndexError):
                return "Unknown"
        elif self.labels is not None:
            try:
                return self.labels[int(value)]
            except (ValueError, IndexError):
                return "Unknown"
        return "Unknown"

    def _lookup_region_index(self, label: str) -> Union[int, str]:
        """
        Return the numeric region index corresponding to the given region name.

        Returns "Unknown" if not found.
        """
        if not isinstance(label, str):
            raise ValueError("label must be a string")

        if self.atlas_type == "surface" and self.regions is not None:
            return np.asarray(self.regions.get(label, []))

        if self._label2index is not None:
            return self._label2index.get(label, "Unknown")

        if self.indexes is not None and self.labels is not None:
            try:
                if isinstance(self.labels, list):
                    pos = self.labels.index(label)
                else:
                    pos = int(np.where(np.array(self.labels) == label)[0][0])
                # Return the corresponding numeric index from self.indexes
                if isinstance(self.indexes, list):
                    return self.indexes[pos]
                else:
                    return int(self.indexes[pos])
            except (ValueError, IndexError):
                return "Unknown"
        elif self.labels is not None:
            # If self.labels is just a list of strings
            try:
                return int(np.where(np.array(self.labels) == label)[0][0])
            except (ValueError, IndexError):
                return "Unknown"
        return "Unknown"

    # -------------------------------------------------------------------------
    # Region name / index
    # -------------------------------------------------------------------------

    def region_name_from_index(self, region_idx: Union[int, str]) -> str:
        """Return region name from numeric region index."""
        return self._lookup_region_name(region_idx)

    def region_index_from_name(self, region_name: str) -> Union[int, str, np.ndarray]:
        """Return region index from region name."""
        return self._lookup_region_index(region_name)

    def list_all_regions(self) -> List[str]:
        """Return a list of all unique region names in this atlas."""
        if self.regions is not None:
            return list(self.regions.keys())
        if self.labels is None:
            return []
        regions = self.labels.values() if isinstance(self.labels, dict) else self.labels
        return list(dict.fromkeys(regions))

    def infer_hemisphere(self, region: Union[int, str]) -> Optional[str]:
        """
        Return the hemisphere ('L' or 'R') inferred from ``region``.

        Returns None if not found or not applicable.
        """
        # Convert numeric region to string name, if needed:
        region_name = (
            region if isinstance(region, str) else self._lookup_region_name(region)
        )

        if isinstance(region, str):
            # If a string is actually an index, resolve it to the label first.
            resolved_name = self._lookup_region_name(region)
            if resolved_name != "Unknown":
                region_name = resolved_name

        if region_name in (None, "Unknown"):
            return None

        # Ensure the region actually belongs to the current atlas.
        if isinstance(region_name, str):
            idx = self._lookup_region_index(region_name)
            missing = isinstance(idx, str) and idx == "Unknown"
            if isinstance(idx, np.ndarray) and idx.size == 0:
                missing = True
            if missing:
                warnings.warn(
                    f"Region '{region_name}' is not part of the '{self.name}' atlas.",
                    UserWarning,
                    stacklevel=2,
                )
                return None

        if self.name.lower() == "schaefer":
            parts = region_name.split("_", 1)
            lower = parts[-1].lower()
            return (
                "L"
                if lower.startswith(("lh"))
                else "R"
                if lower.startswith(("rh"))
                else None
            )

        lower = region_name.lower()
        return (
            "L"
            if lower.endswith(("_lh", "-lh"))
            else "R"
            if lower.endswith(("_rh", "-rh"))
            else None
        )

    # -------------------------------------------------------------------------
    # Coordinate system conversions
    # -------------------------------------------------------------------------

    def convert_system(
        self,
        coord: Union[List[float], np.ndarray],
        source_system: str,
        target_system: str,
    ) -> np.ndarray:
        """Convert coordinates between anatomical systems."""
        source = source_system.lower()
        target = target_system.lower()
        if source == target:
            return np.asarray(coord, dtype=float)
        try:
            func = _TRANSFORMS[(source, target)]
        except KeyError:
            raise ValueError(
                f"Unsupported system conversion: {source_system} -> {target_system}"
            )
        return func(coord)

    # -------------------------------------------------------------------------
    # MNI <--> voxel conversions
    # -------------------------------------------------------------------------

    def _build_voxel_kdtree(self) -> None:
        """Build a KD-tree of voxel centers for nearest-neighbor queries.

        Lazy initialization of the KD-tree for efficient nearest voxel lookups.
        The tree is built once on first use and cached for subsequent queries.
        """
        if self.atlas_type != "volume" or self._voxel_kdtree is not None:
            return

        grid = np.indices(self.vol.shape).reshape(3, -1).T
        mni_coords = grid @ self.hdr[:3, :3].T + self.hdr[:3, 3]

        self._voxel_indices = grid.astype(int)
        self._voxel_kdtree = cKDTree(mni_coords)

    def mni_to_voxel(
        self, mni_coord: Union[List[float], np.ndarray]
    ) -> Tuple[int, int, int]:
        """Convert an MNI coordinate to the nearest voxel indices.

        The coordinate is transformed using the atlas affine. If it does not
        exactly match a voxel center, the voxel whose MNI coordinates are
        closest in Euclidean distance is returned.
        """
        if not isinstance(mni_coord, (list, np.ndarray)):
            raise ValueError("`mni_coord` must be a list or numpy array.")
        pos_arr = np.asarray(mni_coord)
        if pos_arr.shape != (3,):
            raise ValueError("`mni_coord` must be a 3-element (x,y,z).")

        # MNI coordinates are 3D (x, y, z). For affine transforms we use
        # homogeneous coordinates (x, y, z, 1)
        homogeneous = np.append(pos_arr, 1)
        voxel = np.linalg.inv(self.hdr) @ homogeneous
        # self.hdr is a 4×4 affine matrix mapping voxel indices to MNI
        # coordinates. Its inverse maps MNI back to voxel space. The @
        # applies the matrix multiplication.
        rounded = np.round(voxel[:3]).astype(int)

        # Check if this voxel maps back exactly to the MNI coordinate
        back = (self.hdr @ np.append(rounded, 1))[:3]
        if np.allclose(back, pos_arr, atol=1e-6):
            return tuple(rounded)

        # Otherwise search for the voxel with minimal distance in MNI space
        self._build_voxel_kdtree()
        if self._voxel_kdtree is None or self._voxel_indices is None:
            raise RuntimeError(
                f"Failed to construct voxel KD-tree for atlas '{self.name}'. "
                "This may indicate memory issues or invalid volume data."
            )
        _, idx = self._voxel_kdtree.query(pos_arr)
        nearest = self._voxel_indices[idx]
        return tuple(int(v) for v in nearest)

    def mni_to_vertex(
        self,
        mni_coord: Union[List[float], np.ndarray],
        hemi: Optional[Union[List[int], int]] = None,
    ) -> Union[np.ndarray, int]:
        """Convert an MNI coordinate to the nearest vertex index.

        Parameters
        ----------
        mni_coord : list | ndarray
            The target MNI coordinate ``[x, y, z]``.
        hemi : int | list[int] | None
            Hemisphere(s) to restrict the search to. ``0`` for left,
            ``1`` for right. If ``None`` (default) both hemispheres are
            searched.

        Returns
        -------
        int | ndarray
            Index/indices of the matching vertex. If no vertex matches
            exactly, the closest vertex is returned.
        """
        mni_coord = np.asarray(mni_coord)

        # Determine which hemispheres to search
        if hemi is None:
            hemis = [0, 1]
        elif isinstance(hemi, (list, tuple, np.ndarray)):
            hemis = [_get_numeric_hemi(h) for h in hemi]
        else:
            hemis = [_get_numeric_hemi(hemi)]

        all_vertices: List[np.ndarray] = []
        all_coords: List[np.ndarray] = []
        for h in hemis:
            verts = np.asarray(self.vol[h])
            if verts.size == 0:
                continue
            coords = mne.vertex_to_mni(verts, h, self.subject, self.subjects_dir)
            all_vertices.append(verts)
            all_coords.append(coords)

        if not all_vertices:
            return np.array([])

        vertices = np.concatenate(all_vertices)
        coords = np.vstack(all_coords)

        dists = np.linalg.norm(coords - mni_coord, axis=1)
        exact = np.where(dists == 0)[0]
        if exact.size:
            matches = vertices[exact]
            return matches if matches.size > 1 else int(matches[0])

        closest_vertex = vertices[int(np.argmin(dists))]
        return int(closest_vertex)

    def convert_to_source(
        self,
        target: Union[List[float], np.ndarray],
        hemi: Optional[Union[List[int], int]] = None,
        source_system: str = "mni",
    ) -> np.ndarray:
        """Convert a coordinate to the atlas source space.

        Parameters
        ----------
        target : list | ndarray
            The coordinate to convert.
        hemi : int | list[int] | None
            Hemisphere(s) to search when using surface atlases. ``0`` for
            left and ``1`` for right. If ``None`` (default) both hemispheres
            are searched.
        source_system : str, optional
            Coordinate system of ``target``. Defaults to ``"mni"``.
        """
        if source_system.lower() != self.system.lower():
            target = self.convert_system(target, source_system, self.system)
        if self.atlas_type == "volume":
            return self.mni_to_voxel(target)
        if self.atlas_type == "surface":
            return self.mni_to_vertex(target, hemi)
        if self.atlas_type == "coords":
            arr = np.asarray(self.vol, dtype=float)
            tgt = np.asarray(target, dtype=float).reshape(1, 3)
            mask = np.all(np.isclose(arr, tgt), axis=1)
            if not mask.any():
                return np.array([], dtype=int)
            inds = np.where(mask)[0]
            if self.indexes is not None:
                return np.array([self.indexes[i] for i in inds])
            return inds

    def voxel_to_mni(self, voxel_ijk: Union[List[int], np.ndarray]) -> np.ndarray:
        """
        Convert voxel indices (i, j, k) to MNI/world coordinates.

        Returns an array of shape (3,).
        """
        if not isinstance(voxel_ijk, (list, np.ndarray)):
            raise ValueError("`voxel_ijk` must be list or numpy array.")
        src_arr = np.atleast_2d(voxel_ijk)
        ones = np.ones((src_arr.shape[0], 1))
        homogeneous = np.hstack([src_arr, ones])
        transformed = homogeneous @ self.hdr.T
        coords = transformed[:, :3] / transformed[:, 3, np.newaxis]
        if src_arr.shape[0] == 1:
            return coords[0]
        return coords

    def vertex_to_mni(
        self, vertices: Union[List[int], np.ndarray], hemi: Union[List[int], int]
    ) -> np.ndarray:
        """
        Convert vertices to MNI coordinates.

        Returns an array of shape (3,).
        """
        # use mne.vertex_to_mni
        coords = mne.vertex_to_mni(vertices, hemi, self.subject, self.subjects_dir)
        return coords

    def _vertices_to_mni(self, vertices: np.ndarray) -> np.ndarray:
        """Convert vertices from both hemispheres to MNI coordinates."""
        vertices = np.atleast_1d(vertices).astype(int)
        if vertices.size == 0:
            return np.empty((0, 3))
        lh_vertices, rh_vertices = self.vol
        lh_mask = np.in1d(vertices, lh_vertices)
        coords = []
        if lh_mask.any():
            coords.append(
                mne.vertex_to_mni(vertices[lh_mask], 0, self.subject, self.subjects_dir)
            )
        if (~lh_mask).any():
            coords.append(
                mne.vertex_to_mni(
                    vertices[~lh_mask], 1, self.subject, self.subjects_dir
                )
            )
        return np.vstack(coords) if coords else np.empty((0, 3))

    def convert_to_mni(
        self,
        source: Union[List[int], np.ndarray],
        hemi: Optional[Union[List[int], int]] = None,
    ) -> np.ndarray:
        """Convert source space coordinates to MNI."""
        if self.atlas_type == "volume":
            return self.voxel_to_mni(source)
        if self.atlas_type == "surface":
            if hemi is None:
                raise ValueError("hemi must be provided for surface atlases")
            return self.vertex_to_mni(source, hemi)
        if self.atlas_type == "coords":
            return np.asarray(source, dtype=float)

    # -------------------------------------------------------------------------
    # MNI <--> region
    # -------------------------------------------------------------------------

    def mni_to_region_index(
        self,
        mni_coord: Union[List[float], np.ndarray],
        max_distance: Optional[float] = None,
        hemi: Optional[Union[List[int], int]] = None,
        return_distance: bool = False,
    ) -> Union[int, str, Tuple[Union[int, str], float]]:
        """Return the region index for a given MNI coordinate.

        Parameters
        ----------
        mni_coord : list | ndarray
            Target MNI coordinate.
        max_distance : float | None
            If provided, fall back to the nearest region and apply this distance
            threshold. Distances greater than ``max_distance`` return
            ``"Unknown"``.
        hemi : int | list[int] | None
            Hemisphere restriction for surface atlases.
        return_distance : bool
            Whether to also return the distance to the reported region.
        """
        coord = np.asarray(mni_coord, dtype=float)

        result: Union[int, str, np.ndarray]
        dist = 0.0

        if self.atlas_type == "volume":
            ind = np.asarray(self.convert_to_source(coord))
            if ind.size == 0 or np.any((ind < 0) | (ind >= np.array(self.shape))):
                result, dist = self._nearest_region_index(coord, hemi)
            else:
                result = int(self.vol[tuple(ind)])
                if result == 0:
                    result, dist = self._nearest_region_index(coord, hemi)
        elif self.atlas_type == "surface":
            if hemi is not None:
                verts = np.atleast_1d(self.convert_to_source(coord, hemi))
                hemis = (
                    [_get_numeric_hemi(h) for h in hemi]
                    if isinstance(hemi, (list, tuple, np.ndarray))
                    else [_get_numeric_hemi(hemi)]
                )
            else:
                verts = np.atleast_1d(self.convert_to_source(coord))
                hemis = [0, 1]
            exact_matches: List[int] = []
            for v in verts:
                v_int = int(v)
                hemi_v = next(
                    (h for h in hemis if v_int in np.asarray(self.vol[h])), None
                )
                if hemi_v is not None:
                    v_mni = mne.vertex_to_mni(
                        [v_int], hemi_v, self.subject, self.subjects_dir
                    )[0]
                    if np.allclose(v_mni, coord):
                        exact_matches.append(v_int)
                elif self.vertex_to_region and v_int in self.vertex_to_region:
                    exact_matches.append(v_int)
            if exact_matches:
                result = (
                    np.array(exact_matches)
                    if len(exact_matches) > 1
                    else int(exact_matches[0])
                )
            else:
                result, dist = self._nearest_region_index(coord, hemi)
        elif self.atlas_type == "coords":
            exact = np.atleast_1d(self.convert_to_source(coord))
            if exact.size > 0:
                result = exact if exact.size > 1 else int(exact[0])
            else:
                result, dist = self._nearest_region_index(coord, hemi)
        else:
            result, dist = self._nearest_region_index(coord, hemi)

        if max_distance is not None and dist > max_distance:
            result = "Unknown"

        return (result, dist) if return_distance else result

    def mni_to_region_name(
        self,
        mni_coord: Union[List[float], np.ndarray],
        max_distance: Optional[float] = None,
        hemi: Optional[Union[List[int], int]] = None,
        return_distance: bool = False,
    ) -> Union[str, Tuple[str, float]]:
        """Return the region name for a given MNI coordinate."""
        idx, dist = self.mni_to_region_index(
            mni_coord,
            max_distance=max_distance,
            hemi=hemi,
            return_distance=True,
        )
        if isinstance(idx, np.ndarray):
            names = {self._lookup_region_name(int(i)) for i in idx}
            name = names.pop() if len(names) == 1 else "Unknown"
        else:
            name = "Unknown" if idx == "Unknown" else self._lookup_region_name(idx)
        return (name, dist) if return_distance else name

    # ------------------------------------------------------------------
    # Nearest region helpers
    # ------------------------------------------------------------------

    def _compute_centroids(self) -> None:
        """Compute and cache centroids for all regions (volume atlases)."""
        if self.atlas_type != "volume" or self._centroids_cache is not None:
            return
        centroids = {}
        for idx in np.unique(self.vol):
            if idx == 0:
                continue
            coords = self.region_index_to_mni(int(idx))
            # Ensure 2D shape even for singleton regions (1x3). Without this,
            # mean(axis=0) on a 1D array can yield a scalar and later stacking
            # of centroids would fail (shape mismatch), as seen in iEEG case.
            coords = np.atleast_2d(coords)
            if coords.size == 0:
                continue
            centroids[int(idx)] = coords.mean(axis=0)
        self._centroids_cache = centroids

    def _nearest_region_index(
        self,
        mni_coord: Union[List[float], np.ndarray],
        hemi: Optional[Union[List[int], int]] = None,
    ) -> Tuple[Union[int, str], float]:
        """Return (nearest region index, distance) to ``mni_coord``."""
        coord = np.asarray(mni_coord, dtype=float)

        if self.atlas_type == "volume":
            self._compute_centroids()
            if not self._centroids_cache:
                return "Unknown", float("inf")
            ids = np.array(list(self._centroids_cache.keys()))
            cents = np.vstack(list(self._centroids_cache.values()))
            dists = np.linalg.norm(cents - coord, axis=1)
            min_idx = np.argmin(dists)
            return int(ids[min_idx]), float(dists[min_idx])

        if self.atlas_type == "surface":
            if hemi is None:
                hemis = [0, 1]
            elif isinstance(hemi, (list, tuple, np.ndarray)):
                hemis = [_get_numeric_hemi(h) for h in hemi]
            else:
                hemis = [_get_numeric_hemi(hemi)]

            all_vertices: List[np.ndarray] = []
            all_coords: List[np.ndarray] = []
            for h in hemis:
                verts = np.asarray(self.vol[h])
                if verts.size == 0:
                    continue
                coords = mne.vertex_to_mni(verts, h, self.subject, self.subjects_dir)
                all_vertices.append(verts)
                all_coords.append(coords)
            if not all_vertices:
                return "Unknown", float("inf")
            vertices = np.concatenate(all_vertices)
            coords = np.vstack(all_coords)
            dists = np.linalg.norm(coords - coord, axis=1)
            min_idx = int(np.argmin(dists))
            return int(vertices[min_idx]), float(dists[min_idx])

        if self.atlas_type == "coords":
            coords = np.asarray(self.vol, dtype=float)
            dists = np.linalg.norm(coords - coord, axis=1)
            min_idx = int(np.argmin(dists))
            idx = self.indexes[min_idx] if self.indexes is not None else min_idx
            return int(idx), float(dists[min_idx])

        return "Unknown", float("inf")

    # -------------------------------------------------------------------------
    # region index/name <--> all voxel coords
    # -------------------------------------------------------------------------

    def region_index_to_mni(
        self,
        region_idx: Union[int, str, List[int], np.ndarray],
        hemi: Optional[int] = None,
    ) -> np.ndarray:
        """
        Return MNI coordinates for voxels or vertices in ``region_idx``.

        Returns an Nx3 array or an empty array if none found.
        """
        # Make sure region_idx is an integer:
        if self.atlas_type == "volume":
            try:
                idx_val = int(region_idx)
            except (ValueError, TypeError):
                return np.empty((0, 3))
            coords = np.argwhere(self.vol == idx_val)
            if coords.size == 0:
                return np.empty((0, 3))
            return self.convert_to_mni(coords, hemi)
        elif self.atlas_type == "surface":
            try:
                verts = np.atleast_1d(region_idx).astype(int)
            except (ValueError, TypeError):
                return np.empty((0, 3))
            return self._vertices_to_mni(verts)
        elif self.atlas_type == "coords":
            try:
                idx_val = int(region_idx)
            except (ValueError, TypeError):
                return np.empty((0, 3))
            if self.indexes is not None:
                try:
                    pos = list(self.indexes).index(idx_val)
                except ValueError:
                    return np.empty((0, 3))
            else:
                pos = idx_val
                if pos < 0 or pos >= len(self.vol):
                    return np.empty((0, 3))
            return np.atleast_2d(self.vol[pos])

    def region_name_to_mni(self, region_name: str) -> np.ndarray:
        """Return MNI coordinates for voxels matching ``region_name``.

        Returns an Nx3 array or an empty array if no matches are found.
        """
        region_idx = self.region_index_from_name(region_name)
        if isinstance(region_idx, str) and region_idx == "Unknown":
            return np.empty((0, 3))
        if isinstance(region_idx, np.ndarray) and region_idx.size == 0:
            return np.empty((0, 3))
        return self.region_index_to_mni(
            region_idx, _get_numeric_hemi(self.infer_hemisphere(region_name))
        )

    def region_centroid(self, region: Union[int, str]) -> np.ndarray:
        """Return the centroid MNI coordinate for a region or vertex index."""
        if isinstance(region, str):
            coords = self.region_name_to_mni(region)
        else:
            coords = self.region_index_to_mni(region)
        # Some regions can contain exactly one voxel/vertex; keep (1, 3)
        # to make mean/distances robust and consistent with batch cases.
        coords = np.atleast_2d(coords)
        if coords.size == 0:
            return np.empty((0,))
        return coords.mean(axis=0)

    def distance_to_region_centroid(
        self, mni_coord: Union[List[float], np.ndarray], region: Union[int, str]
    ) -> float:
        """Return Euclidean distance from ``mni_coord`` to a region centroid."""
        centroid = self.region_centroid(region)
        if centroid.size == 0:
            return float("inf")
        coord = np.asarray(mni_coord, dtype=float)
        return float(np.linalg.norm(coord - centroid))

    def distance_to_region_boundary(
        self, mni_coord: Union[List[float], np.ndarray], region: Union[int, str]
    ) -> float:
        """Return distance from ``mni_coord`` to the nearest point in ``region``."""
        if isinstance(region, str):
            coords = self.region_name_to_mni(region)
        else:
            coords = self.region_index_to_mni(region)
        # Guard against 1D arrays so pairwise distances work reliably.
        coords = np.atleast_2d(coords)
        if coords.size == 0:
            return float("inf")
        coord = np.asarray(mni_coord, dtype=float)
        dists = np.linalg.norm(coords - coord, axis=1)
        return float(dists.min())

    def membership_scores(
        self,
        mni_coord: Union[List[float], np.ndarray],
        method: str = "centroid",
    ) -> Dict[Union[int, str], float]:
        """Return normalized membership probabilities for all regions."""
        coord = np.asarray(mni_coord, dtype=float)

        # Determine region identifiers
        if self.atlas_type == "volume":
            region_ids = (
                [int(i) for i in (self.indexes or [])]
                if self.indexes is not None
                else [int(i) for i in np.unique(self.vol) if int(i) != 0]
            )
        elif self.atlas_type == "coords":
            if self.indexes is not None:
                region_ids = [int(i) for i in self.indexes]
            else:
                region_ids = list(range(len(self.vol)))
        elif self.atlas_type == "surface" and self.regions is not None:
            region_ids = list(self.regions.keys())
        else:
            return {}

        dists = []
        for rid in region_ids:
            if method == "boundary":
                d = self.distance_to_region_boundary(coord, rid)
            else:
                d = self.distance_to_region_centroid(coord, rid)
            dists.append(d)

        dists_arr = np.array(dists, dtype=float)
        scores = np.exp(-dists_arr)
        total = float(scores.sum())
        if total > 0:
            scores /= total

        # Map region identifiers to names if possible
        if isinstance(self.labels, dict):
            names = [self.labels.get(str(r), str(r)) for r in region_ids]
        elif isinstance(self.labels, (list, np.ndarray)):
            names = list(self.labels)
        else:
            names = region_ids

        return dict(zip(names, scores))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    _SERIAL_VERSION = 1

    def _get_state(self) -> Dict[str, Any]:
        """Return minimal state necessary to recreate this mapper."""
        state: Dict[str, Any] = {
            "name": self.name,
            "vol": self.vol,
            "hdr": self.hdr,
            "labels": self.labels,
            "indexes": self.indexes,
            "regions": self.regions,
            "system": self.system,
        }
        if hasattr(self, "subject"):
            state["subject"] = getattr(self, "subject")
        if hasattr(self, "subjects_dir"):
            state["subjects_dir"] = getattr(self, "subjects_dir")
        return state

    def save(self, filename: str) -> None:
        """Serialize this ``AtlasMapper`` to ``filename`` using pickle."""
        data = {
            "metadata": {
                "class": self.__class__.__name__,
                "version": self._SERIAL_VERSION,
            },
            "state": self._get_state(),
        }
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filename: str) -> "AtlasMapper":
        """Load an ``AtlasMapper`` from ``filename``.

        Parameters
        ----------
        filename : str
            Path to the serialized mapper.

        Returns
        -------
        AtlasMapper
            A reconstructed mapper instance.

        Raises
        ------
        ValueError
            If the file metadata is incompatible.
        """
        with open(filename, "rb") as f:
            data = pickle.load(f)
        meta = data.get("metadata", {})
        if meta.get("class") != cls.__name__:
            raise ValueError("File does not contain AtlasMapper data")
        if meta.get("version") != cls._SERIAL_VERSION:
            raise ValueError("Incompatible AtlasMapper version")
        state = data.get("state", {})
        return cls(**state)


class BatchAtlasMapper:
    """Provide batch (vectorized) conversions for a single atlas mapper.

    Parameters
    ----------
    mapper : AtlasMapper
        The atlas mapper to wrap for vectorized operations.

    Attributes
    ----------
    mapper : AtlasMapper
        Wrapped atlas mapper used for transformations.

    Examples
    --------
    >>> mapper = AtlasMapper(...)
    >>> batch = BatchAtlasMapper(mapper)
    >>> regions = batch.batch_mni_to_region_name([[0, 0, 0], [10, -20, 30]])
    """

    def __init__(self, mapper: AtlasMapper) -> None:
        if not isinstance(mapper, AtlasMapper):
            raise ValueError("mapper must be an instance of AtlasMapper")
        self.mapper = mapper

    # ---- region name <-> index (batch) ---------------------------------------
    def batch_region_name_from_index(self, values: List[Union[int, str]]) -> List[str]:
        """Return the region name for each index in ``values``."""
        return [self.mapper.region_name_from_index(val) for val in values]

    def batch_region_index_from_name(self, labels: List[str]) -> List[Union[int, str]]:
        """Return the region index for each name in ``labels``."""
        return [self.mapper.region_index_from_name(label) for label in labels]

    # ---- MNI <-> voxel (batch) -----------------------------------------------
    def batch_mni_to_voxel(
        self, positions: Union[List[List[float]], np.ndarray]
    ) -> List[tuple]:
        """Convert MNI coordinates to voxel indices (i, j, k)."""
        positions_arr = np.atleast_2d(positions)
        return [self.mapper.mni_to_voxel(pos) for pos in positions_arr]

    def batch_voxel_to_mni(
        self, sources: Union[List[List[int]], np.ndarray]
    ) -> np.ndarray:
        """
        Convert a batch of voxel indices (i, j, k) to MNI coordinates.

        Returns an Nx3 array.
        """
        sources_arr = np.atleast_2d(sources)
        return np.array([self.mapper.voxel_to_mni(s) for s in sources_arr])

    # ---- MNI -> region (batch) -----------------------------------------------
    def batch_mni_to_region_index(
        self,
        positions: Union[List[List[float]], np.ndarray],
        max_distance: Optional[float] = None,
        hemi: Optional[Union[List[int], int]] = None,
    ) -> List[Union[int, str]]:
        """Return region index for each coordinate, using nearest lookup if needed."""
        positions_arr = np.atleast_2d(positions)
        return [
            self.mapper.mni_to_region_index(pos, max_distance=max_distance, hemi=hemi)
            for pos in positions_arr
        ]

    def batch_mni_to_region_name(
        self,
        positions: Union[List[List[float]], np.ndarray],
        max_distance: Optional[float] = None,
        hemi: Optional[Union[List[int], int]] = None,
    ) -> List[str]:
        """Return region name for each coordinate, using nearest lookup if needed."""
        positions_arr = np.atleast_2d(positions)
        return [
            self.mapper.mni_to_region_name(pos, max_distance=max_distance, hemi=hemi)
            for pos in positions_arr
        ]

    # ---- region index/name -> MNI coords (batch) -----------------------------
    def batch_region_index_to_mni(
        self, indices: List[Union[int, str]]
    ) -> List[np.ndarray]:
        """Return MNI coordinates (Nx3) for each region index."""
        return [self.mapper.region_index_to_mni(idx) for idx in indices]

    def batch_region_name_to_mni(self, regions: List[str]) -> List[np.ndarray]:
        """Return MNI coordinates (Nx3) for each region name."""
        return [self.mapper.region_name_to_mni(r) for r in regions]


class MultiAtlasMapper:
    """Manage multiple atlases and provide batch queries across them.

    Parameters
    ----------
    data_dir : str
        Directory for atlas data.
    atlases : dict
        Dictionary mapping atlas names to keyword arguments passed to
        :class:`AtlasFetcher` in order to retrieve each atlas.

    Attributes
    ----------
    mappers : dict
        Mapping of atlas names to :class:`BatchAtlasMapper` instances.
    """

    def __init__(self, data_dir: str, atlases: Dict[str, Dict[str, Any]]) -> None:
        self.mappers = {}

        atlas_fetcher = AtlasFetcher(data_dir=data_dir)
        for name, kwargs in atlases.items():
            atlas_data = atlas_fetcher.fetch_atlas(name, **kwargs)
            vol = atlas_data["vol"]
            hdr = atlas_data["hdr"]
            labels = atlas_data.get("labels")
            indexes = atlas_data.get("indexes")
            subject = kwargs.get("subject", "fsaverage")
            subjects_dir = kwargs.get("subjects_dir")

            # Handle coordinate atlases represented as DataFrames or lists
            if isinstance(vol, pd.DataFrame):
                df = vol
                if {"x", "y", "z"}.issubset(df.columns):
                    vol = df[["x", "y", "z"]].to_numpy()
                else:
                    vol = df.iloc[:, :3].to_numpy()
                if labels is None:
                    for col in ["label", "labels", "name", "region", "roi"]:
                        if col in df.columns:
                            labels = df[col].astype(str).tolist()
                            break
                if indexes is None:
                    indexes = df.index.to_list()
            else:
                arr = np.asarray(vol)
                if hdr is None and arr.ndim == 2 and arr.shape[1] == 3:
                    vol = arr
                    if indexes is None:
                        indexes = np.arange(vol.shape[0])

            single_mapper = AtlasMapper(
                name=name,
                vol=vol,
                hdr=hdr,
                labels=labels,
                indexes=indexes,
                regions=atlas_data.get("regions"),
                subject=subject,
                subjects_dir=subjects_dir,
                system="mni",  # or read from atlas_data if you store that
            )
            batch_mapper = BatchAtlasMapper(single_mapper)
            self.mappers[name] = batch_mapper

    def batch_mni_to_region_names(
        self,
        coords: Union[List[List[float]], np.ndarray],
        max_distance: Optional[float] = None,
        hemi: Optional[Union[List[int], int]] = None,
    ) -> Dict[str, List[str]]:
        """
        Convert a batch of MNI coordinates to region names for all atlases.

        Returns a dict {atlas_name: [region_name, region_name, ...], ...}.
        """
        results = {}
        for atlas_name, mapper in self.mappers.items():
            results[atlas_name] = mapper.batch_mni_to_region_name(
                coords, max_distance=max_distance, hemi=hemi
            )
        return results

    def batch_region_name_to_mni(
        self, region_names: List[str]
    ) -> Dict[str, List[np.ndarray]]:
        """
        Convert a list of region names to MNI coordinates for all atlases.

        Returns a dict {atlas_name: [np.array_of_coords_per_region, ...], ...}.
        """
        results = {}
        for atlas_name, mapper in self.mappers.items():
            results[atlas_name] = mapper.batch_region_name_to_mni(region_names)
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    _SERIAL_VERSION = 1

    def save(self, filename: str) -> None:
        """Serialize all contained mappers to ``filename`` using pickle."""
        mapper_states = {
            name: mapper.mapper._get_state() for name, mapper in self.mappers.items()
        }
        data = {
            "metadata": {
                "class": self.__class__.__name__,
                "version": self._SERIAL_VERSION,
            },
            "state": {"mappers": mapper_states},
        }
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filename: str) -> "MultiAtlasMapper":
        """Load a ``MultiAtlasMapper`` instance from ``filename``."""
        with open(filename, "rb") as f:
            data = pickle.load(f)
        meta = data.get("metadata", {})
        if meta.get("class") != cls.__name__:
            raise ValueError("File does not contain MultiAtlasMapper data")
        if meta.get("version") != cls._SERIAL_VERSION:
            raise ValueError("Incompatible MultiAtlasMapper version")
        mapper_states = data.get("state", {}).get("mappers", {})
        obj = cls.__new__(cls)
        obj.mappers = {}
        for name, mstate in mapper_states.items():
            atlas = AtlasMapper(**mstate)
            obj.mappers[name] = BatchAtlasMapper(atlas)
        return obj
