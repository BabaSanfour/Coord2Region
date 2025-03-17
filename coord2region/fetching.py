import os
import logging
import urllib.parse
import numpy as np
import nibabel as nib
import requests
from typing import Optional
  
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AtlasFileHandler:
    """
    Handles file operations for atlas fetching.
    
    This class provides utility functions to:
      - Load local atlas files.
      - Download atlases from a URL.
      - Package files into a standardized dictionary with keys:
          'vol', 'hdr', 'labels', 'description', and 'file'.
    """
    def __init__(self, data_dir: Optional[str] = None):
        """
        :param data_dir: Directory to store/download atlas files.
             Defaults to a 'data' folder within a hidden '.coord2region' folder in the user's home directory.
        """
        home_dir = os.path.expanduser("~")
        if data_dir is None:
            self.data_dir = os.path.join(home_dir, 'coord2region_data')
        else:
            self.data_dir = os.path.join(home_dir, data_dir)
        os.makedirs(self.data_dir, exist_ok=True)
        self.nilearn_data = os.path.join(home_dir, 'nilearn_data')
        self.mne_data = os.path.join(home_dir, 'mne_data')

    def _fetch_labels(self, fname: str):
        """
        Attempt to fetch labels from a corresponding XML or TXT file.
        
        :param fname: The file name of the atlas image.
        :return: A dictionary of labels if found, else None.
        """
        base, _ = os.path.splitext(fname)        
        fname_xml = base + '.xml'

        # get parent directory
        base_dir = os.path.dirname(fname)
        if "HarvardOxford" in base_dir:
            fname_xml = base_dir + "-Cortical.xml"
        if "Juelich" in base_dir:
            fname_xml = base_dir + ".xml"
        if os.path.exists(fname_xml):
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(fname_xml)
                root = tree.getroot()
                labels = {}
                for label in root.find("data").findall("label"):
                    index = label.find("index").text
                    name = label.find("name").text
                    labels[index] = name
                return labels
            except Exception as e:
                logger.warning(f"Failed to parse XML labels: {e}")
        else:
            fname_txt = base + '.txt'
            if "schaefer" in base_dir:
                fname_txt = os.path.join(base_dir, "Schaefer2018_400Parcels_7Networks_order.txt")
            if os.path.exists(fname_txt):
                with open(fname_txt, 'r') as f:
                    lines = f.readlines()
                labels = {str(idx): line.strip() for idx, line in enumerate(lines)}
                return labels
        logger.warning(f"Failed to fetch labels")
        return None

    def pack_vol_output(self, fname: str, desc: str = None):
        """
        Load an atlas file into a nibabel image (or numpy archive) and package it.
        
        :param fname: Path to the atlas file.
        :param desc: Short description.
        :return: A dictionary with keys: 'vol', 'hdr', 'labels', 'description', 'file'.
        :raises ValueError: If file format is unrecognized.
        """
        path = os.path.abspath(fname)
        _, ext = os.path.splitext(fname)
        ext = ext.lower()

        if ext in ['.nii', '.gz', '.nii.gz']:
            img = nib.load(fname)
            vol_data = img.get_fdata(dtype=np.float32)
            hdr_matrix = img.affine
            labels = self._fetch_labels(path)
            return {
                'vol': vol_data,
                'hdr': hdr_matrix,
                'labels': labels,
                'description': desc,
                'file': fname
            }
        elif ext == '.npz':
            arch = np.load(path, allow_pickle=True)
            vol_data = arch['vol']
            hdr_matrix = arch['hdr']
            labels = None
            if 'labels' in arch and 'index' in arch:
                labels = {idx: name for idx, name in zip(arch['index'], arch['labels'])}
            return {
                'vol': vol_data,
                'hdr': hdr_matrix,
                'labels': labels,
                'description': desc,
                'file': fname
            }
        else:
            raise ValueError(f"Unrecognized file format '{ext}' for path: {path}")

    def pack_surf_output(self, subject: str, subjects_dir: str, parc: str = 'aparc', **kwargs):
        """
        Load surface-based atlas using MNE from FreeSurfer annotation files.

        :param subject: The subject identifier (e.g., 'fsaverage').
        :param subjects_dir: Path to the FreeSurfer subjects directory.
        :param parc: The parcellation name (e.g., 'aparc', 'aparc.a2009s').
        :param kwargs: Additional keyword arguments.
        :return: A dictionary with keys: 'vmap', 'labmap', 'mni'.
                """
        import mne
        src = mne.read_source_spaces(os.path.join(subjects_dir, subject, 'bem', f'{subject}-ico-5-src.fif'), verbose=False)
        labels = mne.read_labels_from_annot(subject, parc=parc, subjects_dir=subjects_dir, verbose=False)
        lh_vert = src[0]['vertno']
        rh_vert = src[1]['vertno']
    
        cortex_dict = {
            label.name: (np.searchsorted(lh_vert, np.intersect1d(lh_vert, label.vertices))
                        if label.hemi == 'lh'
                        else len(lh_vert) + np.searchsorted(rh_vert, np.intersect1d(rh_vert, label.vertices)))
            for label in labels
        }

        labmap = {v: lab for lab, verts in cortex_dict.items() for v in np.atleast_1d(verts)}

        # Compute MNI coordinates for the cortical parts (assuming first two hemispheres)
        mni_list = mne.vertex_to_mni(vertno[:2], [0, 1], subject, subjects_dir=subjects_dir)
        mni_coords = np.concatenate(mni_list, axis=0)
        return {
            'vmap': cortex_dict,
            'labmap': labmap,
            'mni': mni_coords
        }

    def fetch_from_local(self, atlas_path: str):
        """
        Load an atlas from a local file.
        
        :param atlas_path: Path to the local atlas file.
        :return: The standardized atlas dictionary.
        """
        logger.info(f"Loading local atlas file: {atlas_path}")
        return self.pack_vol_output(atlas_path, desc="Local file")

    def fetch_from_url(self, atlas_url: str, **kwargs):
        """
        Download an atlas from a URL (if not already present) and load it.
        
        :param atlas_url: The URL of the atlas.
        :param kwargs: Additional parameters.
        :return: The standardized atlas dictionary.
        :raises RuntimeError: if the download fails.
        """
        parsed = urllib.parse.urlparse(atlas_url)
        file_name = os.path.basename(parsed.path)
        if not file_name:
            file_name = "atlas_download.nii.gz"
        local_path = os.path.join(self.data_dir, file_name)

        if not os.path.exists(local_path):
            logger.info(f"Downloading atlas from {atlas_url}...")
            try:
                with requests.get(atlas_url, stream=True) as r:
                    r.raise_for_status()
                    with open(local_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                logger.info(f"Atlas downloaded to {local_path}")
            except Exception as e:
                if os.path.exists(local_path):
                    os.remove(local_path)
                raise RuntimeError(f"Failed to download from {atlas_url}") from e
        else:
            logger.info(f"Atlas already exists: {local_path}. Skipping download.")

        return local_path


class AtlasFetcher:
    """
    Fetches neuroimaging atlases using various methods.
    
    This class uses an AtlasFileHandler instance for file operations and provides atlas-specific
    fetchers. Supported atlas identifiers include volumetric atlases such as:
      - "aal", "brodmann", "harvard-oxford", "juelich", "schaefer", "yeo", "aparc2009"
    
    In addition, this module now supports MNE-based, surface annotation atlases via:
      - "mne-annot" (generic annotation; requires keyword arguments 'subject' and 'subjects_dir')
      - "mne-aparc2009" (a convenience key that sets parc to 'aparc.a2009s')
      
    Each fetcher returns a standardized dictionary. For volumetric atlases the keys are:
      'vol', 'hdr', 'labels', 'description', 'file'.
    For MNE annotation atlases, additional keys include:
      'vmap' (label to vertex mapping), 'labmap' (vertex-to-label mapping), and 'mni' (MNI coordinates).
    """

    # Fallback URL for Talairach atlas (used by the Brodmann fetcher).
    ATLAS_URLS = {
        'talairach': 'https://www.talairach.org/talairach.nii',
        'aal': 'https://www.gin.cnrs.fr/wp-content/uploads/AAL3v2_for_SPM12.tar.gz',
    }

    def __init__(self, data_dir: str = None):
        """
        :param data_dir: Directory to store/download atlas files.
        """

        self.file_handler = AtlasFileHandler(data_dir=data_dir)
        self.data_dir = self.file_handler.data_dir
        self.nilearn_data = self.file_handler.nilearn_data
        self.mne_data = self.file_handler.mne_data
        self._atlas_fetchers = {
            "aal": self._fetch_atlas_aal,
            "brodmann": self._fetch_atlas_brodmann,
            "harvard-oxford": self._fetch_atlas_harvard_oxford,
            "juelich": self._fetch_atlas_juelich,
            "schaefer": self._fetch_atlas_schaefer,
            "yeo": self._fetch_atlas_yeo,
            # MNE-based atlases:
            "aparc2009": self._fetch_atlas_aparc2009,
        }

    # ---- Volumetric atlas fetchers using Nilear ----
    def _fetch_atlas(self, fetcher, **kwargs):
        try:
            return fetcher(data_dir=self.file_handler.data_dir, **kwargs)
        except:
            return fetcher(data_dir=self.file_handler.nilearn_data, **kwargs)

    def _fetch_atlas_aal(self, **kwargs):
        try:
            from nilearn.datasets import fetch_atlas_aal
            version = kwargs.get('version', '3v2')
            fetched = self._fetch_atlas(fetch_atlas_aal, version=version, **kwargs)
            return self.file_handler.pack_vol_output(fetched["maps"], desc="AAL Atlas")
        except:
            # Fallback URL for AAL atlas.
            return self.file_handler.fetch_from_url(self.ATLAS_URLS.get('aal'))
    
    def _fetch_atlas_brodmann(self, **kwargs):
        try:
            from nilearn.datasets import fetch_atlas_talairach
            fetched = self._fetch_atlas(fetch_atlas_talairach, level_name="ba", **kwargs)
            return self.file_handler.pack_vol_output(fetched["maps"], desc="Talairach Atlas")
        except:
            # Fallback URL for Talairach atlas.
            return self.file_handler.fetch_from_url(self.ATLAS_URLS.get('talairach'))

    def _fetch_atlas_harvard_oxford(self, **kwargs):
        from nilearn.datasets import fetch_atlas_harvard_oxford
        atlas_name = kwargs.get('version', 'cort-maxprob-thr25-2mm')
        fetched = self._fetch_atlas(fetch_atlas_harvard_oxford, atlas_name=atlas_name, **kwargs)
        

    def _fetch_atlas_juelich(self, **kwargs):
        from nilearn.datasets import fetch_atlas_juelich
        atlas_name = kwargs.get('version', 'maxprob-thr0-1mm')
        fetched = fetched = self._fetch_atlas(fetch_atlas_juelich, atlas_name=atlas_name, **kwargs)
        return self.file_handler.pack_vol_output(fetched["filename"], desc=f"Juelich {atlas_name} Atlas")

    def _fetch_atlas_schaefer(self, **kwargs):
        from nilearn.datasets import fetch_atlas_schaefer_2018
        fetched = self._fetch_atlas(fetch_atlas_schaefer_2018, **kwargs)
        return self.file_handler.pack_vol_output(fetched["maps"], desc="Schaefer 2018 Atlas")

    def _fetch_atlas_yeo(self, **kwargs):
        from nilearn.datasets import fetch_atlas_yeo_2011
        fetched = self._fetch_atlas(fetch_atlas_yeo_2011, **kwargs)
        version = kwargs.get('version', 'thick_17')
        return self.file_handler.pack_vol_output(fetched[version], desc=f"Yeo 2011 {version} Atlas")

    # ---- MNE-based (surface annotation) atlas fetcher ----
    
    def _fetch_atlas_aparc2009(self, **kwargs):
        return self.file_handler.pack_surf_output(parc='aparc.a2009s', **kwargs)
    
    # ---- Public method ----

    def fetch_atlas(self, atlas_name: str, atlas_url: str = None, version: str = None, **kwargs):
        """
        Fetch an atlas given an atlas identifier.
        
        The identifier can be:
            (a) A URL (starting with http:// or https://),
            (b) A local file path,
            (c) Nilearn or mne atlases atlases (e.g., "aal", "harvard-oxford", "aparc2009", "mne-annot", etc.).
        
        For MNE-based atlases (keys starting with "mne-"), additional keyword arguments are required:
            - subject: subject identifier (e.g., "fsaverage")
            - subjects_dir: path to the FreeSurfer subjects directory
        
        :param atlas_name: The atlas identifier or file path.
        :param version: Version specifier (used for certain atlases, e.g., AAL).
        :param atlas_url: (Optional) Override URL for fetching the atlas.
        :param kwargs: Additional keyword arguments for the specific fetcher.
        :return: A standardized atlas dictionary.
        :raises ValueError: if the atlas identifier is not recognized.
        """
        # Case (a): URL provided.
        if atlas_url is not None and (atlas_url.startswith('http://') or atlas_url.startswith('https://')):
            return self.file_handler.fetch_from_url(atlas_url, **kwargs)
        
        # Case (b): Local file path.
        atlas_file = kwargs.get("atlas_file")
        if atlas_file and os.path.isfile(atlas_file):
            return self.file_handler.fetch_from_local(atlas_file)
        elif os.path.isfile(os.path.join(self.data_dir, atlas_name)):
            return self.file_handler.fetch_from_local(os.path.join(self.data_dir, atlas_name))
    
        # Case (c): nilearn or mne atlases.
        key = atlas_name.lower()
        fetcher = self._atlas_fetchers.get(key)
        if fetcher:
            return fetcher(**kwargs)
    
        raise ValueError(f"Unrecognized atlas name '{atlas_name}'. Available options: {list(self._atlas_fetchers.keys())}.")


# Example usage: # remove later
if __name__ == '__main__':
    af = AtlasFetcher(data_dir="atlas_data")
    # TODO: fix fetch using url!
    # atlas = af.fetch_atlas("aal", atlas_url="https://www.gin.cnrs.fr/wp-content/uploads/AAL3v2_for_SPM12.tar.gz")
    # logger.info(f"Fetched atlas: {atlas['description']} from file: {atlas['file']}")
    # atlas = af.fetch_atlas("talairach", atlas_url="https://www.talairach.org/talairach.nii")
    # logger.info(f"Fetched atlas: {atlas['description']} from file: {atlas['file']}")

    # TODO: test fetch from local file

    # TODO add "destrieux": self._fetch_atlas_destrieux, similar to mne-annot
    # TODO brodmann: self._fetch_atlas_brodmann is not downloading the file
    # TODO harvard-oxford: self._fetch_atlas_harvard_oxford fix labels fetching for this atlas
    # TODO juilich: self._fetch_atlas_juelich fix labels fetching for this atlas
    # TODO schaefer: self._fetch_atlas_schaefer check if labels are extracted correctly
    # TODO yeo: self._fetch_atlas_yeo check label extraction from description file
    # TODO add other nibabel, nilearn, mne atlases
    atlas = af.fetch_atlas("yeo")
    print(isinstance(atlas, dict))
    print(atlas.keys())
    print(atlas["labels"])

    # TODO: test fetching a surface-based atlas
    # atlas = af.fetch_atlas("mne-annot", subject="fsaverage", subjects_dir="mne_data")
    # print(isinstance(atlas, dict))

    # TODO: add save/load methods for created atlases

    # TODO: add method to list available atlases
    # TODO: refactor to use a single fetch method for all atlases
    # TODO: add method to fetch all atlases at once
    # TODO: check for atlases that supported by both mne and nilearn if else
    