import numpy as np
import mne
from typing import Any, Dict, List, Optional, Union, Tuple
from fetching import AtlasFetcher

# TODO: Add getting region with the shortest distance to a given coordinate
# TODO: Add save/load methods for AtlasMapper and MultiAtlasMapper
# TODO: Add support for surface atlases
def _get_numeric_hemi(hemi: Union[str, int]) -> int:
    """
    Convert hemisphere string to numeric code (0 or 1).
    """
    if isinstance(hemi, int):
        return hemi
    if hemi.lower() in ('l', 'lh', 'left'):
        return 0
    if hemi.lower() in ('r', 'rh', 'right'):
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
    :labels: Region labels. If a dict, keys should be strings for numeric indices, and values are region names. If a list/array, it should match `index`.
    :index: Region indices (numeric) corresponding to the labels list/array. Not needed if `labels` is a dict.
    :system: The anatomical coordinate space (e.g. "mni", "tal").

    Attributes
    ----------
    :attrib: name: str
    :attrib: vol: np.ndarray
    :attrib: hdr: np.ndarray
    :attrib: labels: dict or list or None
    :attrib: index: list or np.ndarray or None
    :attrib: system: str
    :attrib: shape: tuple
    """

    def __init__(self,
                 name: str,
                 vol: np.ndarray,
                 hdr: np.ndarray,
                 labels: Optional[Union[Dict[str, str], List[str], np.ndarray]] = None,
                 index: Optional[Union[List[int], np.ndarray]] = None,
                 subject: Optional[str] = "fsaverage",
                 subjects_dir: Optional[str] = None,
                 system: str = 'mni') -> None:

        self.name = name
        self.labels = labels
        self.index = index
        self.system = system

        # Basic shape checks
        if isinstance(vol, np.ndarray) and isinstance(hdr, np.ndarray):
            self.vol = np.asarray(vol)
            self.hdr = np.asarray(hdr)
            if self.vol.ndim != 3:
                raise ValueError("`vol` must be a 3D numpy array.")
            if self.hdr.shape != (4, 4):
                raise ValueError("`hdr` must be a 4x4 transform matrix.")
            self.shape = self.vol.shape
            self.atlas_type = 'volume'
        if isinstance(vol, list):
            self.vol = vol
            self.hdr = None
            self.atlas_type = 'surface'
            self.subject = subject
            self.subjects_dir = subjects_dir

        # If labels is a dict, prepare an inverse mapping:
        #   region_name -> region_index
        # TODO: Add support for surface atlases: sometimes a region has many indexes
        if isinstance(self.labels, dict):
            # Here we assume keys are index strings, values are region names
            self._label2index = {v: k for k, v in self.labels.items()}
        else:
            self._label2index = None

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

        value_str = str(value)
        if isinstance(self.labels, dict):
            return self.labels.get(value_str, "Unknown")

        if self.index is not None and self.labels is not None:
            try:
                # If the index array is a list, we use index(); if np.ndarray, we do np.where
                if isinstance(self.index, list):
                    pos = self.index.index(int(value))
                else:
                    pos = int(np.where(self.index == int(value))[0][0])
                return self.labels[pos]
            except (ValueError, IndexError):
                return "Unknown"
        elif self.labels is not None:
            # labels might just be a list or array with no separate index
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

        if self._label2index is not None:
            return self._label2index.get(label, "Unknown")

        if self.index is not None and self.labels is not None:
            try:
                if isinstance(self.labels, list):
                    pos = self.labels.index(label)
                else:
                    pos = int(np.where(np.array(self.labels) == label)[0][0])
                # Return the corresponding numeric index from self.index
                if isinstance(self.index, list):
                    return self.index[pos]
                else:
                    return int(self.index[pos])
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

    def region_index_from_name(self, region_name: str) -> Union[int, str]:
        """
        Public method: Return region index from region name.
        """
        return self._lookup_region_index(region_name)

    def list_all_regions(self) -> List[str]:
        """
        Return a list of all unique region names in this atlas.
        """
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
        region_name = region if isinstance(region, str) else self._lookup_region_name(region)
        if region_name in (None, "Unknown"):
            return None

        if self.name.lower() == 'schaefer':
            parts = region.split('_', 1)
            if len(parts) > 1:
                return {'LH': 'L', 'RH': 'R'}.get(parts[1])
            return None

        lower = region_name.lower()
        return 'L' if lower.endswith(('_lh', '-lh')) else 'R' if lower.endswith(('_rh', '-rh')) else None

    # -------------------------------------------------------------------------
    # MNI <--> voxel conversions
    # -------------------------------------------------------------------------

    def mni_to_voxel(self, mni_coord: Union[List[float], np.ndarray]) -> Tuple[int, int, int]:
        """
        Convert an (x,y,z) MNI/world coordinate to voxel indices (i,j,k).
        Returns (i, j, k) as integers (rounded).
        """
        if not isinstance(mni_coord, (list, np.ndarray)):
            raise ValueError("`mni_coord` must be a list or numpy array.")
        pos_arr = np.asarray(mni_coord)
        if pos_arr.shape != (3,):
            raise ValueError("`mni_coord` must be a 3-element (x,y,z).")

        # MNI coordinates are usually in 3D (x, y, z), but to apply affine transformations, we need homogeneous coordinates (x, y, z, 1)
        homogeneous = np.append(pos_arr, 1)
        voxel = np.linalg.inv(self.hdr) @ homogeneous
        #self.hdr is a 4×4 affine transformation matrix that maps voxel indices ↔ MNI coordinates.
        #np.linalg.inv(self.hdr) computes the inverse of the affine matrix, which transforms MNI back to voxel space.
        #@ homogeneous applies the matrix multiplication.
        ijk = tuple(map(int, np.round(voxel[:3])))
        return ijk
    
    def mni_to_vertex(self, mni_coord: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Convert MNI coordinates to vertices.
        Returns an array of vertex indices from both hemispheres that match the given coordinate.
        """
        mni = mne.vertex_to_mni(self.vol, [0, 1], self.subject, self.subjects_dir)
        mni_coord_round = np.round(mni_coord, decimals=5)
        mni_rounded = np.round(mni, decimals=5)
        matches = np.all(mni_rounded == mni_coord_round, axis=2)
        vertex = np.nonzero(matches[0])[0] if matches[0].any() else np.nonzero(matches[1])[0]
        return self.index[vertex]
    
    def convert_to_source(self, target: Union[List[float], np.ndarray], hemi: Optional[Union[List[int], int]] = None) -> np.ndarray:
        """
        Convert target mni to the source space.
        """
        if self.atlas_type == 'volume':
            return self.mni_to_voxel(target)
        if self.atlas_type == 'surface':
            return self.mni_to_vertex(target)

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
    
    def vertex_to_mni(self, vertices: Union[List[int], np.ndarray], hemi: Union[list[int], int]) -> np.ndarray:
        """
        Convert vertices to MNI coordinates.
        Returns an array of shape (3,).
        """
        # use mne.vertex_to_mni
        coords = mne.vertex_to_mni(vertices, hemi, self.subject, self.subjects_dir)
        return coords
    
    def convert_to_mni(self, source: Union[List[int], np.ndarray], hemi: Optional[Union[List[int], int]] = None) -> np.ndarray:
        """
        Convert source space to MNI.
        """
        if self.atlas_type == 'volume':
            return self.voxel_to_mni(source)
        if self.atlas_type == 'surface':
            if hemi is None:
                raise ValueError("hemi must be provided for surface atlases")
            return self.vertex_to_mni(source, hemi)
    # -------------------------------------------------------------------------
    # MNI <--> region
    # -------------------------------------------------------------------------

    def mni_to_region_index(self, mni_coord: Union[List[float], np.ndarray]) -> Union[int, str]:
        """
        Return the region index for a given MNI coordinate.
        """
        ind = self.convert_to_source(mni_coord)
        if self.atlas_type == 'volume':
            if any(i < 0 or i >= s for i, s in zip(ind, self.shape)):
                return "Unknown"
            return int(self.vol[tuple(ind)])
        elif self.atlas_type == 'surface':
            if ind < 0 or ind >= len(self.index):
                return "Unknown"
            return ind[0]

    def mni_to_region_name(self, mni_coord: Union[List[float], np.ndarray]) -> str:
        """
        Return the region name for a given MNI coordinate.
        """
        region_idx = self.mni_to_region_index(mni_coord)
        if region_idx == "Unknown":
            return "Unknown"
        return self._lookup_region_name(int(region_idx))

    # -------------------------------------------------------------------------
    # region index/name <--> all voxel coords
    # -------------------------------------------------------------------------
    
    def region_index_to_mni(self, region_idx: Union[int, str], hemi: Optional[int] = None) -> np.ndarray:
        """
        Return an Nx3 array of MNI coords for all voxels matching the specified region index.
        Returns an empty array if none found.
        """
        # Make sure region_idx is an integer:
        try:
            idx_val = int(region_idx)
        except (ValueError, TypeError):
            return np.empty((0, 3))
        if self.atlas_type == 'volume':
            coords = np.argwhere(self.vol == idx_val)
        elif self.atlas_type == 'surface':
            coords = np.argwhere(self.index == idx_val)
        if coords.size == 0:
            return np.empty((0, 3))
        return self.convert_to_mni(coords, hemi) # Gets all mnicoords for the region if surface atlas

    def region_name_to_mni(self, region_name: str) -> np.ndarray:
        """
        Return an Nx3 array of MNI coords for all voxels matching the specified region name.
        Returns an empty array if none found.
        """
        region_idx = self.region_index_from_name(region_name)
        if region_idx == "Unknown":
            return np.empty((0, 3))
        return self.region_index_to_mni(region_idx, _get_numeric_hemi(self.infer_hemisphere(region_name)))

class BatchAtlasMapper:
    """
    Provides batch (vectorized) conversions over many coordinates for a single AtlasMapper.

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
    def batch_mni_to_voxel(self, positions: Union[List[List[float]], np.ndarray]) -> List[tuple]:
        """
        Convert a batch of MNI coordinates to voxel indices (i,j,k).
        """
        positions_arr = np.atleast_2d(positions)
        return [self.mapper.mni_to_voxel(pos) for pos in positions_arr]

    def batch_voxel_to_mni(self, sources: Union[List[List[int]], np.ndarray]) -> np.ndarray:
        """
        Convert a batch of voxel indices (i,j,k) to MNI coords.
        Returns an Nx3 array.
        """
        sources_arr = np.atleast_2d(sources)
        return np.array([self.mapper.voxel_to_mni(s) for s in sources_arr])

    # ---- MNI -> region (batch) -----------------------------------------------
    def batch_mni_to_region_index(self, positions: Union[List[List[float]], np.ndarray]) -> List[Union[int, str]]:
        """
        For each MNI coordinate, return the corresponding region index.
        """
        positions_arr = np.atleast_2d(positions)
        return [self.mapper.mni_to_region_index(pos) for pos in positions_arr]

    def batch_mni_to_region_name(self, positions: Union[List[List[float]], np.ndarray]) -> List[str]:
        """
        For each MNI coordinate, return the corresponding region name.
        """
        positions_arr = np.atleast_2d(positions)
        return [self.mapper.mni_to_region_name(pos) for pos in positions_arr]

    # ---- region index/name -> MNI coords (batch) -----------------------------
    def batch_region_index_to_mni(self, indices: List[Union[int, str]]) -> List[np.ndarray]:
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
    Manages multiple atlases by name, providing batch MNI->region or region->MNI queries
    across all atlases at once.

    Parameters
    ----------
    :params data_dir: Directory for atlas data.
    :params atlases: Dictionary of {atlas_name: fetch_kwargs}, used by AtlasFetcher to retrieve each atlas.

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
            index = atlas_data.get("index")
            # system = atlas_data.get("system", "mni")

            single_mapper = AtlasMapper(
                name=name,
                vol=vol,
                hdr=hdr,
                labels=labels,
                index=index,
                system="mni"   # or read from atlas_data if you store that
            )
            batch_mapper = BatchAtlasMapper(single_mapper)
            self.mappers[name] = batch_mapper

    def batch_mni_to_region_names(self, coords: Union[List[List[float]], np.ndarray]) -> Dict[str, List[str]]:
        """
        Convert a batch of MNI coordinates to region names for ALL atlases.
        Returns a dict {atlas_name: [region_name, region_name, ...], ...}.
        """
        results = {}
        for atlas_name, mapper in self.mappers.items():
            results[atlas_name] = mapper.batch_mni_to_region_name(coords)
        return results

    def batch_region_name_to_mni(self, region_names: List[str]) -> Dict[str, List[np.ndarray]]:
        """
        Convert a list of region names to MNI coordinates for ALL atlases.
        Returns a dict {atlas_name: [np.array_of_coords_per_region, ...], ...}.
        """
        results = {}
        for atlas_name, mapper in self.mappers.items():
            results[atlas_name] = mapper.batch_region_name_to_mni(region_names)
        return results

if __name__ == '__main__':
# Fetch an atlas
    af = AtlasFetcher()
    # fetch aparc.a2009s
    aparc = af.fetch_atlas('aparc.a2009s')
    # fetch nilearn harvard-oxford
    aparc_mapper = AtlasMapper(
        name='aparc.a2009s',
        vol=aparc['vol'],
        hdr=aparc['hdr'],
        labels=aparc['labels'],
        index=aparc['indexes'],
        subjects_dir=af.subjects_dir,

    )
    print(aparc_mapper.region_name_from_index(32))
    print(aparc_mapper.region_index_from_name('S_oc_sup_and_transversal-lh'))
    # print(aparc_mapper.list_all_regions())
    print(aparc_mapper.infer_hemisphere("Lat_Fis-post-rh"))
    print(aparc_mapper.convert_to_mni(32, hemi=0))
    print(aparc_mapper.convert_to_source([-23.91684151, -78.48731995,  16.8182888]))
    print(aparc_mapper.mni_to_region_name([-23.91684151, -78.48731995,  16.8182888]))
    print(aparc_mapper.region_index_to_mni(32, 0))
    print(aparc_mapper.region_name_to_mni("S_oc_sup_and_transversal-lh"))







