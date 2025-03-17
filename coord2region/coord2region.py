import numpy as np
from typing import Any, Dict, List, Optional, Union

class AtlasRegionMapper:
    """
    Holds a single atlas and its metadata, and provides methods to query region labels and coordinates.

    Parameters
    ----------
    :param name: Human-readable identifier of the atlas (e.g., "aal" or "brodmann").
    :param vol: 3D array (I, J, K) representing the atlas, where each voxel contains a region index.
    :param hdr: 4x4 affine transform for voxel-to-world (MNI) coordinates.
    :param labels: Mapping of region indices to region names. Can be a dictionary (keys as strings) or an array-like.
    :param index: Explicit numeric indices corresponding to `labels`. If `labels` is a dict, this can be left as None.
    :param system: The anatomical coordinate space (e.g., "mni", "tal", "unknown").

    Attributes
    ----------
    :attr name: Human-readable identifier of the atlas.
    :attr vol: 3D array representing the atlas.
    :attr hdr: 4x4 affine transform for voxel-to-world (MNI) coordinates.
    :attr labels: Mapping of region indices to region names.
    :attr index: Explicit numeric indices corresponding to `labels`.
    :attr system: The anatomical coordinate space.
    """

    def __init__(self,
                 name: str,
                 vol: np.ndarray,
                 hdr: np.ndarray,
                 labels: Optional[Union[Dict[str, str], List[str], np.ndarray]] = None,
                 index: Optional[Union[List[int], np.ndarray]] = None,
                 system: str = 'mni') -> None:
        self.name = name
        self.vol = np.asarray(vol)
        self.hdr = np.asarray(hdr)
        self.labels = labels
        self.index = index
        self.system = system
        # TODO: Add support for surf based atlases (mne)
        # Basic checks
        # TODO: check for vol and hdr only when working with volume based atlases and addd support for surf atlases
        if self.vol.ndim != 3:
            raise ValueError("`vol` must be a 3D numpy array.")
        if self.hdr.shape != (4, 4):
            raise ValueError("`hdr` must be a 4x4 transform matrix.")

        self.shape = self.vol.shape

        # If labels is a dict, prepare an inverse mapping.
        if isinstance(self.labels, dict):
            self._label2index = {v: k for k, v in self.labels.items()}
        else:
            self._label2index = None

    # --- Region Mapping Methods ---

    def _get_region_name(self, value: Union[int, str]) -> str:
        """
        Get the region label corresponding from an index.
        
        :param value: The region index or label. Must be an int or str.
        :return: The region name.

        If `labels` is a dict, lookup is performed using string keys.
        If `labels` is array-like with a corresponding `index` array, the position is found.
        Returns "Unknown" if not found.
        """
        if not isinstance(value, (int, str)):
            raise ValueError("value must be an int or str")
        value_str = str(value)
        if isinstance(self.labels, dict):
            return self.labels.get(value_str, "Unknown")
        
        if self.index is not None and self.labels is not None:
            try:
                # Support both list and numpy array for index.
                if isinstance(self.index, list):
                    pos = self.index.index(int(value))
                else:
                    pos = int(np.where(self.index == int(value))[0][0])
                return self.labels[pos]
            except (ValueError, IndexError):
                return "Unknown"
        return "Unknown"

    def get_region_name(self, value: Union[int, str]) -> str:
        """
        Return the region name corresponding to the given atlas value.
        """
        return self._get_region_name(value)

    def _get_region_index(self, label: str) -> Union[int, str]:
        """
        Return the numeric region index corresponding to the given label.
        
        :param label: The region name.
        :return: The region index.

        If `labels` is a dict, lookup is performed using string keys.
        If `labels` is array-like with a corresponding `index` array, the position is found.
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
                if isinstance(self.index, list):
                    return self.index[pos]
                else:
                    return self.index[pos]
            except (ValueError, IndexError):
                return "Unknown"
        return "Unknown"
    
    def get_region_index(self, label: str) -> Union[int, str]:
        """
        Return the region index corresponding to the given label.
        """
        return self._get_region_index(label)

    def get_list_of_regions(self) -> List[str]:
        """
        Return a list of all unique region names in the atlas.

        :return: A list of region names.
        """
        if isinstance(self.labels, dict):
            return list(self.labels.values())
        elif self.labels is not None:
            return list(self.labels)
        else:
            return []

    def get_hemisphere(self, region: Union[int, str]) -> Optional[str]:
        """
        Return the hemisphere ('L' or 'R') inferred from the region name.
        
        :param region: The region index or name.
        :return: The hemisphere ('L' or 'R') or None if not found.
        """
        region_name = region if isinstance(region, str) else self.get_region_name(region)
        if region_name in (None, "Unknown"):
            return None
        region_lower = region_name.lower()
        if region_lower.endswith('_l'):
            return 'L'
        elif region_lower.endswith('_r'):
            return 'R'
        return None

    # --- Coordinate Conversion Methods ---

    def pos_to_source(self, pos: Union[List[float], np.ndarray]) -> tuple[int, int, int]:
        """
        Convert an MNI coordinate (x, y, z) to voxel indices using the inverse of the affine transform.

        :param pos: The MNI coordinate (x, y, z).
        :return: The voxel indices (i, j, k).
        """
        if not isinstance(pos, (list, np.ndarray)):
            raise ValueError("`pos` must be a list or numpy array.")
        pos_arr = np.asarray(pos)
        if pos_arr.shape != (3,):
            raise ValueError("`pos` must be a 3-element coordinate (x, y, z).")
        homogeneous = np.append(pos_arr, 1)
        voxel = np.linalg.inv(self.hdr) @ homogeneous
        return tuple(map(int, np.round(voxel[:3])))

    def pos_to_index(self, pos: Union[List[float], np.ndarray]) -> Union[int, str]:
        """
        Return the atlas region index for a given MNI coordinate.

        :param pos: The MNI coordinate (x, y, z).
        :return: The region index or "Unknown".
        """
        if not isinstance(pos, (list, np.ndarray)):
            raise ValueError("`pos` must be a list or numpy array.")
        ijk = self.pos_to_source(pos)
        if any(i < 0 or i >= s for i, s in zip(ijk, self.shape)):
            return "Unknown"
        return int(self.vol[ijk])

    def pos_to_region(self, pos: Union[List[float], np.ndarray]) -> str:
        """
        Return the region name for a given MNI coordinate.

        :param pos: The MNI coordinate (x, y, z).
        :return: The region name or "Unknown".
        """
        if not isinstance(pos, (list, np.ndarray)):
            raise ValueError("`pos` must be a list or numpy array.")
        idx = self.pos_to_index(pos)
        if idx == "Unknown":
            return "Unknown"
        return self.get_region_name(idx)

    def source_to_pos(self, source: Union[List[int], np.ndarray]) -> Union[np.ndarray, np.ndarray]:
        """
        Convert voxel indices (i, j, k) to MNI coordinates using the affine transform.

        :param source: The voxel indices (i, j, k).
        :return: The MNI coordinates
        """
        if not isinstance(source, (list, np.ndarray)):
            raise ValueError("`source` must be a list or numpy array.")
        src_arr = np.atleast_2d(source)
        ones = np.ones((src_arr.shape[0], 1))
        homogeneous = np.hstack([src_arr, ones])
        transformed = homogeneous @ self.hdr.T
        coords = transformed[:, :3] / transformed[:, 3, np.newaxis]
        return coords if src_arr.shape[0] > 1 else coords[0]

    def index_to_pos(self, index: Union[int, str]) -> np.ndarray:
        """
        Return MNI coordinates for all voxels that have the specified atlas region index.

        :param index: The region index or name.
        :return: The MNI coordinates.
        """
        if not isinstance(index, (int, str)):
            raise ValueError("`index` must be an int or str.")
        try:
            idx = int(index)
        except ValueError:
            return np.empty((0, 3))
        coords = np.argwhere(self.vol == idx)
        if coords.size == 0:
            return np.empty((0, 3))
        return self.source_to_pos(coords)

    def region_to_pos(self, region: str) -> np.ndarray:
        """
        Return MNI coordinates for all voxels corresponding to the given region name.

        :param region: The region name.
        :return: The MNI coordinates
        """
        if not isinstance(region, str):
            raise ValueError("`region` must be a string.")
        idx = self.get_region_index(region)
        if idx == "Unknown":
            return np.empty((0, 3))
        return self.index_to_pos(idx)