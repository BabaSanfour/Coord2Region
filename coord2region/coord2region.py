import numpy as np

class AtlasRegionMapper:
    """
    Holds a single atlas and its metadata.

    Parameters
    ----------
    name : str
        Human-readable identifier of the atlas (e.g., "aal" or "brodmann").
    vol : np.ndarray, shape (I, J, K)
        The 3D integer/float array representing the atlas, where each voxel 
        corresponds to a labeled region index (e.g., 0, 1, 2, ...).
    hdr : np.ndarray, shape (4, 4)
        The affine transform for voxel->world coordinates.
    labels : array-like or dict, optional
        A structure mapping region indices to region names. 
        Could be a list, a numpy array, or a dict {index: label}.
        By default, None if not provided.
    index : np.ndarray or list, optional
        Explicit numeric indices that correspond to `labels`.
        For example, if `index = [1,2,3]` and `labels = ["Area1","Area2","Area3"]`,
        then a voxel labeled '2' in `vol` => "Area2".
        If you supply a dict to `labels` keyed by these index values, you may leave
        `index` as None. Default is None.
    system : {"mni", "tal", "unknown"}, default "mni"
        The anatomical coordinate space or reference system.
        Commonly 'mni' or 'tal'. Use "unknown" if not sure.

    Attributes
    ----------
    name : str
    vol : np.ndarray
    hdr : np.ndarray
    labels : array-like or dict, optional
    index : np.ndarray or list, optional
    system : str
    shape : tuple of int
        The shape (I, J, K) of the atlas data.

    Methods
    -------
    get_label(value):
        Returns the label corresponding to a given index (int) value in `vol`.
    """

    def __init__(self,
                 name,
                 vol,
                 hdr,
                 labels=None,
                 index=None,
                 system='mni'):
        self.name = name
        self.vol = np.asarray(vol)
        self.hdr = hdr
        self.labels = labels
        self.index = index
        self.system = system

        # Basic checks
        if not isinstance(vol, np.ndarray) or vol.ndim != 3:
            raise ValueError("`vol` must be a 3D numpy array.")
        if not isinstance(hdr, np.ndarray) or hdr.shape != (4, 4):
            raise ValueError("`hdr` must be a 4x4 transform matrix.")

        self.shape = vol.shape  # convenience
        if isinstance(self.labels, dict):
            self._label2index = {v: k for k, v in self.labels.items()}
        else:
            self._label2index = None

    def _get_region_name(self, value):
        """
        Return the label corresponding to the integer `value` in the volume.

        If `labels` is:
          - a dict {region_index: region_name}, we use `labels.get(value, 'Unknown')`.
          - a list or np.ndarray, we find where `index == value`.
          - None, returns 'Unknown'.

        Returns
        -------
        str
            The label (region name), or 'Unknown' if not found.
        """
        value = str(value) #TODO: Check if this is necessary/could be problematic
        if isinstance(self.labels, dict):
            return self.labels.get(value, "Unknown")

        # Otherwise, if we have an array-like `labels` plus a separate `index` array,
        if self.index is not None and len(self.index) == len(self.labels):
            try:
                idx_pos = self.index.index(value) if isinstance(self.index, list) else np.where(self.index == value)[0][0]
                return self.labels[idx_pos]
            except (ValueError, IndexError):
                return "Unknown"
        return "Unknown"
    
    def get_region_name(self, value):
        """
        Return the clean region name for a given index value in the volume.
        """
        #TODO: Implement this method, for now just return the raw label
        return self._get_region_name(value)

    def get_region_index(self, label):
        """
        Return the index corresponding to the label in the volume.

        If `labels` is:
          - a dict {region_index: region_name}, we use `labels.get(value, 'Unknown')`.
          - a list or np.ndarray, we find where `index == value`.
          - None, returns 'Unknown'.

        Returns
        -------
        str
            The index (region index), or 'Unknown' if not found.
        """
        if self._label2index is not None:
            return self._label2index.get(label, "Unknown")
        
        
        if self.index is not None and len(self.index) == len(self.labels):
            try:
                idx_pos = self.labels.index(label) if isinstance(self.labels, list) else np.where(self.labels == label)[0][0]
                return self.index[idx_pos]
            except (ValueError, IndexError):
                return "Unknown"
            
        return "Unknown"
    
    def _get_hemisphere(self, region):
        """
        Return the hemisphere (left/right) of the region name/index.
        """
        if not isinstance(region, str):
            region = self.get_region_name(region)
        if region is None or region == "Unknown":
            return None
        region_lower = region.lower()
        if region_lower.endswith('_l'):
            return 'L'
        elif region_lower.endswith('_r'):
            return 'R'
        return None
    
    def get_list_of_regions(self):
        """
        Return a list of all unique region names in the atlas.
        """
        #TODO: Implement this method
        return None
    
    def pos_to_source(self, pos):
        """
        Return the source indices (i, j, k) for a given MNI coordinate using hdr.
        """
        pos = np.asarray(pos)
        if pos.shape != (3,):
            raise ValueError("pos must be a 3-element coordinate (x,y,z).")
        xyz = np.linalg.inv(self.hdr) @ np.array([*pos, 1])
        return tuple(map(int, np.round(xyz)[:3]))
    
    def pos_to_index(self, pos):
        """
        Return the region index for a given MNI coordinate using hdr.
        """
        ijk = self.pos_to_source(pos)
        # Check bounds
        if any(i < 0 or i >= s for i, s in zip(ijk, self.shape)):
            return "Unknown"  # or None
        return int(self.vol[ijk])
    
    def pos_to_region(self, pos):
        """
        Return the region name for a given MNI coordinate using hdr.
        """
        index = self.pos_to_index(pos)
        if index == "Unknown":
            return "Unknown"
        return self.get_region_name(index)
    
    def source_to_pos(self, source):
        """
        Return the MNI coordinate for a given source indices (i, j, k) using hdr.
        """
        source = np.atleast_2d(source)  # Ensure shape is (N, 3) even if (3,)
        source = np.hstack([source, np.ones((source.shape[0], 1))])  # (N, 4)
        transformed = source @ self.hdr.T 
        xyz = transformed[:, :3] / transformed[:, 3, np.newaxis]
        return xyz if len(source) > 1 else xyz[0]  
        
    def index_to_pos(self, index):
        """
        Return the MNI coordinate for a given region index.
        """
        index = int(index) #TODO: Check if this is necessary/could be problematic
        coords = np.argwhere(self.vol == index)  # shape (N,3)
        if coords.size == 0:
            return np.empty((0, 3))  # empty array if none found
        return self.source_to_pos(coords)
    
    def region_to_pos(self, region):
        """
        Return the MNI coordinate for a given region name.
        """
        index = self.get_region_index(region)
        if index == "Unknown":
            return np.empty((0, 3))
        return self.index_to_pos(index)