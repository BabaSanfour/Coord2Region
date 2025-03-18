# --------------------------------------------------------
# mne_mapper.py
# --------------------------------------------------------
"""
MNE-based mapper: loads annotation files, builds region lookups,
handles coordinate <-> region conversions for a single atlas.
"""
import os
import warnings
import numpy as np
import mne

try:
    from sklearn.neighbors import KDTree
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class MNEAtlasLoader:
    """
    Loads a surface-based atlas (annotation) via MNE/FreeSurfer from subject's label folder,
    builds vertex->label lookup, optionally builds a KDTree for fast MNI-based queries.
    """

    def __init__(self,
                 atlas_name: str,
                 subject: str,
                 subjects_dir: str,
                 build_kdtree: bool = True,
                 verbose: bool = False,
                 src_spacing: str = 'ico-5'):
        """
        :param atlas_name: e.g. 'aparc', 'aparc_sub', 'HCPMMP1', etc.
        :param subject: e.g. 'fsaverage'
        :param subjects_dir: path to FreeSurfer subjects_dir
        :param build_kdtree: if True, build a KD-tree of MNI coords for nearest-vertex queries
        :param verbose: if True, print logs
        :param src_spacing: the spacing used when building a source space file name (e.g. ico-5).
                            The default fsaverage-ico-5 is ~20k vertices per hemisphere.
        """
        self.atlas_name = atlas_name
        self.subject = subject
        self.subjects_dir = os.path.abspath(subjects_dir)
        self.build_kdtree = build_kdtree
        self.verbose = verbose
        self.src_spacing = src_spacing

        # Outputs
        self.labels_ = None  # list of mne.Label
        self.label_names_ = []  # sorted list of unique label names
        self.vertex_to_label_ = {}  # vertex_id -> label_name
        self.verts_coords_ = None  # Nx3 array of MNI coords (LH + RH)
        self.kdtree_ = None

        self._load_annot()
        self._prepare_lookup()
        if self.build_kdtree:
            self._build_kdtree()

    def _load_annot(self):
        """
        Load the annotation files (both hemis) using mne.read_labels_from_annot.
        This populates self.labels_ (list of mne.Label).
        """
        if self.verbose:
            print(f"[MNEAtlasLoader] Loading atlas '{self.atlas_name}' for subject '{self.subject}'")

        # load annotation (labels) for both hemis
        self.labels_ = mne.read_labels_from_annot(
            subject=self.subject,
            parc=self.atlas_name,
            hemi='both',  # read LH and RH
            subjects_dir=self.subjects_dir,
            verbose=self.verbose
        )
        if self.verbose:
            print(f"[MNEAtlasLoader] Loaded {len(self.labels_)} labels from annot.")

    def _prepare_lookup(self):
        """
        Build label_names_ and vertex_to_label_ from self.labels_.
        """
        label_name_set = set()
        for lab in self.labels_:
            name = lab.name
            label_name_set.add(name)
            for vtx in lab.vertices:
                self.vertex_to_label_[vtx] = name

        self.label_names_ = sorted(label_name_set)
        if self.verbose:
            print(f"[MNEAtlasLoader] Found {len(self.label_names_)} unique label names.")

    def _build_kdtree(self):
        """
        Build a KDTree of MNI coordinates for all vertices in LH+RH (according to a source space).
        Then queries can do nearest-vertex to get region.
        """
        # We'll attempt to read a source space file to get the vertex->MNI transform
        src_fname = os.path.join(self.subjects_dir, self.subject, 'bem',
                                 f"{self.subject}-{self.src_spacing}-src.fif")
        if not os.path.exists(src_fname):
            # fallback or raise
            msg = (f"No source space file found at {src_fname}. "
                   "Please generate or specify a different spacing.")
            raise FileNotFoundError(msg)

        src = mne.read_source_spaces(src_fname, verbose=self.verbose)
        if len(src) < 2:
            warnings.warn("Source space does not have both hemispheres?")

        # gather LH and RH
        lh_vertno = src[0]['vertno']
        rh_vertno = src[1]['vertno'] if len(src) > 1 else np.array([])

        # convert to MNI
        mni_lh = mne.vertex_to_mni([lh_vertno], [0], subject=self.subject, subjects_dir=self.subjects_dir)
        coords_lh = mni_lh[0]
        coords_rh = np.zeros((0, 3))
        if rh_vertno.size > 0:
            mni_rh = mne.vertex_to_mni([rh_vertno], [1], subject=self.subject, subjects_dir=self.subjects_dir)
            coords_rh = mni_rh[0]

        total_verts = len(lh_vertno) + len(rh_vertno)
        self.verts_coords_ = np.zeros((total_verts, 3), dtype=float)
        self.verts_coords_[:len(lh_vertno), :] = coords_lh
        self.verts_coords_[len(lh_vertno):, :] = coords_rh

        if not HAS_SKLEARN:
            warnings.warn("scikit-learn not installed; can't build KDTree. Nearest-vertex search will be manual.")
            self.kdtree_ = None
            return

        self.kdtree_ = KDTree(self.verts_coords_)
        if self.verbose:
            print(f"[MNEAtlasLoader] Built KDTree with {total_verts} vertices for {self.atlas_name}.")

    def pos_to_region(self, mni_xyz: np.ndarray) -> str:
        """
        Given an MNI coordinate (x, y, z), find the nearest vertex in fsaverage space,
        look up that vertex's label. Return the label name or "Unknown" if not found.

        :param mni_xyz: shape (3,) or (3x1)
        :return: region label name
        """
        if mni_xyz.shape != (3,):
            raise ValueError("pos_to_region expects a 3-element coordinate.")
        if self.kdtree_ is None or self.verts_coords_ is None:
            # fallback: naive search
            dist = np.linalg.norm(self.verts_coords_ - mni_xyz[None, :], axis=1)
            idx = np.argmin(dist)
        else:
            dist, idx = self.kdtree_.query(mni_xyz.reshape(1, -1), k=1)
            idx = idx[0][0]

        # check if that idx is in the vertex_to_label_ dict
        if idx not in self.vertex_to_label_:
            return "Unknown"
        labname = self.vertex_to_label_[idx]
        if 'unknown' in labname.lower():
            return "Unknown"
        return labname

    def region_to_pos(self, region_name: str, method: str = 'center') -> np.ndarray:
        """
        Given a region name, return a single MNI coordinate representing that region
        (center or c-o-m). If not found, returns an empty array.

        :param region_name: e.g. 'G_precentral-lh'
        :param method: 
            'center' => average the MNI coords of all vertices,
            'c-o-m'  => use label.center_of_mass() on spherical surface (more 'true' center)
        :return: shape (3,) coordinate in MNI, or empty if region not found.
        """
        matching_labels = [lab for lab in self.labels_ if lab.name == region_name]
        if len(matching_labels) == 0:
            # attempt partial match ignoring -lh/-rh, or just return empty
            return np.array([])

        label = matching_labels[0]
        vtxs = label.vertices

        # gather MNI coords of all vertices in this label:
        coords = self._get_mni_coords_for_vertices(vtxs)

        if coords.size == 0:
            return np.array([])

        if method == 'center':
            # Direct average in MNI space
            return coords.mean(axis=0)

        elif method == 'c-o-m':
            # Use the spherical center_of_mass from MNE
            # => returns a vertex index in local hemisphere numbering
            # We must figure out which hemisphere:
            hemi_local = 0 if label.hemi == 'lh' else 1
            # center_of_mass returns an int: vertex index in that hemisphere's space
            # i.e. 0..(n_lh-1) for LH, 0..(n_rh-1) for RH
            # So we need to offset if it's the right hemisphere in the global array.

            idx_cofm_local = label.center_of_mass(
                subject=self.subject,
                hemi=hemi_local,
                subjects_dir=self.subjects_dir,
                restrict_label=True  # ensures c-o-m is restricted to this label
            )

            # Now let's find how many LH vertices we have in the source space so we can offset if necessary:
            # If we've not stored that, let's do it or retrieve it from the source spaces:
            if self._lh_count is None or self._rh_count is None:
                self._get_lh_rh_counts()

            if hemi_local == 0:
                # left hemisphere => global_index = idx_cofm_local
                global_vtx = idx_cofm_local
            else:
                # right hemisphere => global_index = idx_cofm_local + #LH vertices
                global_vtx = idx_cofm_local + self._lh_count

            # Safety check:
            if global_vtx < 0 or global_vtx >= len(self.verts_coords_):
                return np.array([])

            # Now we can return that vertex's MNI coordinate:
            return self.verts_coords_[global_vtx]

        else:
            raise ValueError("Unknown method. Use 'center' or 'c-o-m'.")


    def _get_lh_rh_counts(self):
        """
        Helper to figure out how many vertices are in the LH vs RH for our chosen source space.
        We store them in self._lh_count, self._rh_count.
        This is run once if needed.
        """
        import os

        src_fname = os.path.join(
            self.subjects_dir,
            self.subject,
            'bem',
            f"{self.subject}-{self.src_spacing}-src.fif"
        )
        if not os.path.exists(src_fname):
            raise FileNotFoundError(f"Missing source space file: {src_fname}")

        src = mne.read_source_spaces(src_fname, verbose=self.verbose)
        lh_vertno = src[0]['vertno']
        rh_vertno = src[1]['vertno'] if len(src) > 1 else []
        self._lh_count = len(lh_vertno)
        self._rh_count = len(rh_vertno)
    def list_regions(self):
        """Return the sorted list of region names."""
        return self.label_names_

    def _get_mni_coords_for_vertices(self, vert_idxs: np.ndarray) -> np.ndarray:
        """
        Return Nx3 MNI coords for the given global vertex indices (LH+RH).
        If a vertex index is out of range, skip it.
        """
        valid_mask = (vert_idxs >= 0) & (vert_idxs < len(self.verts_coords_))
        valid_verts = vert_idxs[valid_mask]
        return self.verts_coords_[valid_verts, :] if len(valid_verts) > 0 else np.zeros((0, 3), dtype=float)


class MNEVectorizedAtlasRegionMapper:
    """
    Provides batch methods for MNEAtlasLoader's pos_to_region, region_to_pos, etc.
    """
    def __init__(self, loader: MNEAtlasLoader):
        self.loader = loader

    def batch_pos_to_region(self, positions: np.ndarray) -> List[str]:
        """
        Convert a batch of MNI coordinates to region names.
        :param positions: shape (N, 3)
        :return: list of region names
        """
        results = []
        for pos in positions:
            reg = self.loader.pos_to_region(pos)
            results.append(reg)
        return results

    def batch_region_to_pos(self, region_names: List[str], method: str = 'center') -> List[np.ndarray]:
        """
        Convert a list of region names to MNI coordinates.
        :param region_names: list of str
        :param method: 'center' or 'c-o-m'
        :return: list of shape(3,) arrays
        """
        coords_list = []
        for rn in region_names:
            coord = self.loader.region_to_pos(rn, method=method)
            coords_list.append(coord)
        return coords_list
