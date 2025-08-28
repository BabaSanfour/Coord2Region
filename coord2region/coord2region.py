"""Coordinate-to-region mapping utilities.

This module provides classes and helper functions for converting between
MNI coordinates, voxel indices, and anatomical region labels. It enables
lookups and transformations across multiple brain atlases.
"""

import numpy as np
import mne
from typing import Any, Dict, List, Optional, Union, Tuple
from .fetching import AtlasFetcher


# TODO: Add getting region with the shortest distance to a given coordinate
# TODO: Add save/load methods for AtlasMapper and MultiAtlasMapper
# TODO: Add support for surface atlases
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
    """
    Stores a single atlas (a 3D numpy array + 4x4 affine for volumetric
    atlases or a vertices array for surface atlases) and provides
    coordinate <-> voxel <-> region lookups.

    Parameters
    ----------
    :param name: Identifier for the atlas (e.g. "aal" or "brodmann").
    :vol: A 3D numpy array representing the volumetric atlas.
    :hdr: A 4x4 affine transform mapping voxel indices -> MNI/world coordinates.
    :labels: Region labels. If a dict, keys should be strings for numeric
        indices and values are region names. If a list/array, it should
        match `indexes`.
    :indexes: Region indices (numeric) corresponding to the labels list or
        array. Not needed if `labels` is a dict.
    :regions: For surface atlases, mapping of region names to vertex indices.
    :system: The anatomical coordinate space (e.g. "mni", "tal").

    Attributes
    ----------
    :attrib: name: str
    :attrib: vol: np.ndarray
    :attrib: hdr: np.ndarray
    :attrib: labels: dict or list or None
    :attrib: indexes: list or np.ndarray or None
    :attrib: regions: dict or None
    :attrib: system: str
    :attrib: shape: tuple
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
        if isinstance(vol, list):
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
        """
        Public method: Return region name from numeric region index.
        """
        return self._lookup_region_name(region_idx)

    def region_index_from_name(self, region_name: str) -> Union[int, str, np.ndarray]:
        """
        Public method: Return region index from region name.
        """
        return self._lookup_region_index(region_name)

    def list_all_regions(self) -> List[str]:
        """
        Return a list of all unique region names in this atlas.
        """
        if self.regions is not None:
            return list(self.regions.keys())
        if self.labels is None:
            return []
        regions = self.labels.values() if isinstance(self.labels, dict) else self.labels
        return list(dict.fromkeys(regions))

    def infer_hemisphere(self, region: Union[int, str]) -> Optional[str]:
        """
        Return the hemisphere ('L' or 'R') inferred from the region name,
        or None if not found or not applicable.
        """
        # Convert numeric region to string name, if needed:
        region_name = (
            region if isinstance(region, str) else self._lookup_region_name(region)
        )
        if region_name in (None, "Unknown"):
            return None

        if self.name.lower() == "schaefer":
            parts = region_name.split("_", 1)
            lower = parts[-1].lower()
            return (
                "L"
                if lower.startswith(("lh"))
                else "R" if lower.startswith(("rh")) else None
            )

        lower = region_name.lower()
        return (
            "L"
            if lower.endswith(("_lh", "-lh"))
            else "R" if lower.endswith(("_rh", "-rh")) else None
        )

    # -------------------------------------------------------------------------
    # MNI <--> voxel conversions
    # -------------------------------------------------------------------------

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
        grid = np.indices(self.vol.shape).reshape(3, -1).T
        ones = np.ones((grid.shape[0], 1))
        mni_coords = (np.hstack((grid, ones)) @ self.hdr.T)[:, :3]
        dists = np.linalg.norm(mni_coords - pos_arr, axis=1)
        nearest = grid[np.argmin(dists)]
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
    ) -> np.ndarray:
        """Convert an MNI coordinate to the atlas source space.

        Parameters
        ----------
        target : list | ndarray
            The MNI coordinate to convert.
        hemi : int | list[int] | None
            Hemisphere(s) to search when using surface atlases. ``0`` for
            left and ``1`` for right. If ``None`` (default) both hemispheres
            are searched.
        """
        if self.atlas_type == "volume":
            return self.mni_to_voxel(target)
        if self.atlas_type == "surface":
            return self.mni_to_vertex(target, hemi)

    def voxel_to_mni(self, voxel_ijk: Union[List[int], np.ndarray]) -> np.ndarray:
        """
        Convert voxel indices (i,j,k) to MNI/world coordinates.
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
                mne.vertex_to_mni(vertices[~lh_mask], 1, self.subject, self.subjects_dir)
            )
        return np.vstack(coords) if coords else np.empty((0, 3))

    
    def convert_to_mni(
        self,
        source: Union[List[int], np.ndarray],
        hemi: Optional[Union[List[int], int]] = None,
    ) -> np.ndarray:
        """
        Convert source space to MNI.
        """
        if self.atlas_type == "volume":
            return self.voxel_to_mni(source)
        if self.atlas_type == "surface":
            if hemi is None:
                raise ValueError("hemi must be provided for surface atlases")
            return self.vertex_to_mni(source, hemi)
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

        result: Union[int, str]
        dist = 0.0

        if self.atlas_type == "volume":
            ind = np.asarray(self.convert_to_source(coord))
            if ind.size == 0 or np.any((ind < 0) | (ind >= np.array(self.shape))):
                result, dist = self._nearest_region_index(coord, hemi)
            else:
                result = int(self.vol[tuple(ind)])
                if result == 0:
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
        self, region_idx: Union[int, str, List[int], np.ndarray], hemi: Optional[int] = None
    ) -> np.ndarray:
        """
        Return an Nx3 array of MNI coordinates for all voxels or vertices
        matching ``region_idx``. Returns an empty array if none found.
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

    def region_name_to_mni(self, region_name: str) -> np.ndarray:
        """
        Return an Nx3 array of MNI coordinates for all voxels matching the
        specified region name. Returns an empty array if none found.
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
        if coords.size == 0:
            return np.empty((0,))
        return coords.mean(axis=0)


    def region_centroid(self, region: Union[int, str]) -> np.ndarray:
        """Return the centroid MNI coordinate for a region or vertex index."""
        if isinstance(region, str):
            coords = self.region_name_to_mni(region)
        else:
            coords = self.region_index_to_mni(region)
        if coords.size == 0:
            return np.empty((0,))
        return coords.mean(axis=0)

class BatchAtlasMapper:
    """
    Provides batch (vectorized) conversions over many coordinates for a
    single AtlasMapper.

    Example:
    --------
    mapper = AtlasMapper(...)
    batch = BatchAtlasMapper(mapper)

    regions = batch.batch_mni_to_region_name([[0, 0, 0], [10, -20, 30]])
    """

    def __init__(self, mapper: AtlasMapper) -> None:
        if not isinstance(mapper, AtlasMapper):
            raise ValueError("mapper must be an instance of AtlasMapper")
        self.mapper = mapper

    # ---- region name <-> index (batch) ---------------------------------------
    def batch_region_name_from_index(self, values: List[Union[int, str]]) -> List[str]:
        """
        For each region index in `values`, return the corresponding region name.
        """
        return [self.mapper.region_name_from_index(val) for val in values]

    def batch_region_index_from_name(self, labels: List[str]) -> List[Union[int, str]]:
        """
        For each region name in `labels`, return the corresponding region index.
        """
        return [self.mapper.region_index_from_name(label) for label in labels]

    # ---- MNI <-> voxel (batch) -----------------------------------------------
    def batch_mni_to_voxel(
        self, positions: Union[List[List[float]], np.ndarray]
    ) -> List[tuple]:
        """
        Convert a batch of MNI coordinates to voxel indices (i,j,k).
        """
        positions_arr = np.atleast_2d(positions)
        return [self.mapper.mni_to_voxel(pos) for pos in positions_arr]

    def batch_voxel_to_mni(
        self, sources: Union[List[List[int]], np.ndarray]
    ) -> np.ndarray:
        """
        Convert a batch of voxel indices (i,j,k) to MNI coords.
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
            self.mapper.mni_to_region_index(
                pos, max_distance=max_distance, hemi=hemi
            )
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
            self.mapper.mni_to_region_name(
                pos, max_distance=max_distance, hemi=hemi
            )
            for pos in positions_arr
        ]

    def batch_mni_to_nearest_region_index(
        self,
        positions: Union[List[List[float]], np.ndarray],
        max_distance: Optional[float] = None,
        hemi: Optional[Union[List[int], int]] = None,
    ) -> List[Union[int, str]]:
        """Return nearest region index for each coordinate."""

        positions_arr = np.atleast_2d(positions)
        return [
            self.mapper.mni_to_nearest_region_index(
                pos, max_distance=max_distance, hemi=hemi
            )
            for pos in positions_arr
        ]

    def batch_mni_to_nearest_region_name(
        self,
        positions: Union[List[List[float]], np.ndarray],
        max_distance: Optional[float] = None,
        hemi: Optional[Union[List[int], int]] = None,
    ) -> List[str]:
        """Return nearest region name for each coordinate."""

        positions_arr = np.atleast_2d(positions)
        return [
            self.mapper.mni_to_nearest_region_name(
                pos, max_distance=max_distance, hemi=hemi
            )
            for pos in positions_arr
        ]

    # ---- region index/name -> MNI coords (batch) -----------------------------
    def batch_region_index_to_mni(
        self, indices: List[Union[int, str]]
    ) -> List[np.ndarray]:
        """
        For each region index, return an array of MNI coords (Nx3) for that region.
        """
        return [self.mapper.region_index_to_mni(idx) for idx in indices]

    def batch_region_name_to_mni(self, regions: List[str]) -> List[np.ndarray]:
        """
        For each region name, return an array of MNI coords (Nx3) for that region.
        """
        return [self.mapper.region_name_to_mni(r) for r in regions]


class MultiAtlasMapper:
    """
    Manages multiple atlases by name, providing batch MNI->region or
    region->MNI queries across all atlases at once.

    Parameters
    ----------
    :params data_dir: Directory for atlas data.
    :params atlases: Dictionary of {atlas_name: fetch_kwargs}, used by
        AtlasFetcher to retrieve each atlas.

    Attributes
    :attrib: mappers: dict
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
            # system = atlas_data.get("system", "mni")

            single_mapper = AtlasMapper(
                name=name,
                vol=vol,
                hdr=hdr,
                labels=labels,
                indexes=indexes,
                system="mni",  # or read from atlas_data if you store that
            )
            batch_mapper = BatchAtlasMapper(single_mapper)
            self.mappers[name] = batch_mapper

    def batch_mni_to_region_names(
        self, coords: Union[List[List[float]], np.ndarray]
    ) -> Dict[str, List[str]]:
        """
        Convert a batch of MNI coordinates to region names for ALL atlases.
        Returns a dict {atlas_name: [region_name, region_name, ...], ...}.
        """
        results = {}
        for atlas_name, mapper in self.mappers.items():
            results[atlas_name] = mapper.batch_mni_to_region_name(coords)
        return results

    def batch_region_name_to_mni(
        self, region_names: List[str]
    ) -> Dict[str, List[np.ndarray]]:
        """
        Convert a list of region names to MNI coordinates for ALL atlases.
        Returns a dict {atlas_name: [np.array_of_coords_per_region, ...], ...}.
        """
        results = {}
        for atlas_name, mapper in self.mappers.items():
            results[atlas_name] = mapper.batch_region_name_to_mni(region_names)
        return results
