import os
import mne
import logging
import numpy as np
from typing import Optional
from nibabel.nifti1 import Nifti1Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# TODO: Raise SSL issue for URL not working suggest to download the file manually and provide the path (aal, brodmann, talairach)
# TODO: test fetch from local file
# TODO add "destrieux": self._fetch_atlas_destrieux, similar to mne-annot
# TODO add other nibabel, nilearn, mne atlases
# TODO: UPDATE fetch mne atlases
# TODO: add save/load methods for created objects!
# TODO: add method to list available atlases
# TODO: check for atlases that supported by both mne and nilearn if else

class AtlasFileHandler:
    """
    Handles file operations for atlas fetching.
    
    This class provides utility functions to:
      - Load local atlas files.
      - Download atlases from a URL.
      - Package files into a standardized dictionary with keys:
          'vol', 'hdr', 'labels', 'description', and 'file'.
    """
    def __init__(self, data_dir: Optional[str] = None, subjects_dir: Optional[str] = None):
        """
        :param data_dir: Directory to store/download atlas files.
             Defaults to a 'data' folder within a hidden 'coord2region' folder in 
             the user's home directory.
        :param subjects_dir: Directory with FreeSurfer subjects (where fsaverage is or will be downloaded).
        """
        home_dir = os.path.expanduser("~")
        if data_dir is None:
            self.data_dir = os.path.join(home_dir, 'coord2region')
        elif os.path.isabs(data_dir):
            self.data_dir = data_dir
        else:
            self.data_dir = os.path.join(home_dir, data_dir)

        try:
            os.makedirs(self.data_dir, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Could not create data directory {self.data_dir}: {e}")

        if not os.access(self.data_dir, os.W_OK):
            raise ValueError(f"Data directory {self.data_dir} is not writable")

        self.nilearn_data = os.path.join(home_dir, 'nilearn_data')
        # When working with surf based or MNE atlases
        self.subjects_dir = mne.get_config('SUBJECTS_DIR', None)
        if subjects_dir is None:
            logger.warning("Please provide a subjects_dir or set MNE's SUBJECTS_DIR in your environment.")

    def fetch_labels(self, labels: str):
        """
        :param labels: Path to the labels file or a list of labels.
        :return: A list of labels.
        """
        if isinstance(labels, str):
            raise NotImplementedError("Reading labels from file is not yet implemented.")
        elif isinstance(labels, list):
            return labels
        else:
            raise ValueError(f"Invalid labels type: {type(labels)}")

    def pack_vol_output(self, file: str):
        """
        Load an atlas file into a nibabel image (or numpy archive) and package it.
        
        :param file: Path to the atlas image file (NIfTI, NPZ) or a Nifti1Image object.
        :return: A dictionary with keys: 'vol', 'hdr'.
        :raises ValueError: If file format is unrecognized.
        """

        if isinstance(file, str):
            path = os.path.abspath(file)
            _, ext = os.path.splitext(file)
            ext = ext.lower()

            if ext in ['.nii', '.gz', '.nii.gz']:
                import nibabel as nib
                img = nib.load(file)
                vol_data = img.get_fdata(dtype=np.float32)
                hdr_matrix = img.affine
                return {
                    'vol': vol_data,
                    'hdr': hdr_matrix,
                }
 
            elif ext == '.npz':
                arch = np.load(path, allow_pickle=True)
                vol_data = arch['vol']
                hdr_matrix = arch['hdr']
                return {
                    'vol': vol_data,
                    'hdr': hdr_matrix,
                }
            else:
                raise ValueError(f"Unrecognized file format '{ext}' for path: {path}")
        else:
            if isinstance(file, Nifti1Image):
                vol_data = file.get_fdata(dtype=np.float32)
                hdr_matrix = file.affine
                return {
                    'vol': vol_data,
                    'hdr': hdr_matrix,
                }

    def pack_surf_output(self, fetcher: mne.datasets, subject: str='fsaverage', subjects_dir: str=None, parc: str = 'aparc', **kwargs):
        """
        Load surface-based atlas using MNE from FreeSurfer annotation files.

        :param subject: The subject identifier (e.g., 'fsaverage').
        :param subjects_dir: Path to the FreeSurfer subjects directory.
        :param parc: The parcellation name (e.g., 'aparc', 'aparc.a2009s').
        :param kwargs: Additional keyword arguments.
        :return: A dictionary with keys: 'vmap', 'labmap', 'mni'.
        """
        import mne

        if self.subjects_dir is None:
            subjects_dir = mne.datasets.sample.data_path() / "subjects"

        # Read annotation labels
        try:
            labels = mne.read_labels_from_annot(
                subject, parc, subjects_dir=subjects_dir, **kwargs
            )
        except Exception as e:
            #use fetcher to get the atlas
            fetcher(subjects_dir=subjects_dir)
            labels = mne.read_labels_from_annot(
                subject, parc, subjects_dir=subjects_dir, **kwargs
            )
        # Set up source space to retrieve vertices information
        src = mne.setup_source_space(subject, spacing='oct6', subjects_dir=subjects_dir, add_dist=False)
        lh_vert = src[0]['vertno']
        rh_vert = src[1]['vertno']
    
        cortex_dict = {
            label.name: (np.searchsorted(lh_vert, np.intersect1d(lh_vert, label.vertices))
                        if label.hemi == 'lh'
                        else len(lh_vert) + np.searchsorted(rh_vert, np.intersect1d(rh_vert, label.vertices)))
            for label in labels
        }

        labmap = {v: lab for lab, verts in cortex_dict.items() for v in np.atleast_1d(verts)}

        return {
            'vol': [lh_vert, rh_vert],
            'hdr': None,
            'labels': labmap,
        }

    def fetch_from_local(self, atlas: str, labels: str):
        """
        Load an atlas from a local file.
        
        :param atlas: Path to the atlas file or a Nifti1Image object.
        :return: The standardized atlas dictionary.
        """
        logger.info(f"Loading local atlas file: {atlas}")
        output = self.pack_vol_output(atlas)
        output['labels'] = self.fetch_labels(labels)
        return output

    def fetch_from_url(self, atlas_url: str, **kwargs):
        """
        Download an atlas from a URL (if not already present) and load it.
        
        :param atlas_url: The URL of the atlas.
        :param kwargs: Additional parameters.
        :return: The standardized atlas dictionary.
        :raises RuntimeError: if the download fails.
        """
        import warnings
        warnings.warn("The file name is expected to be in the URL", UserWarning)
        import urllib.parse
        import requests
        #requests.packages.urllib3.disable_warnings()
        parsed = urllib.parse.urlparse(atlas_url)
        file_name = os.path.basename(parsed.path)
        local_path = os.path.join(self.data_dir, file_name)

        if not os.path.exists(local_path):
            logger.info(f"Downloading atlas from {atlas_url}...")
            try:
                with requests.get(atlas_url, stream=True, timeout=30, verify=False) as r:
                    r.raise_for_status()
                    with open(local_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                logger.info(f"Atlas downloaded to {local_path}")
            except Exception as e:
                if os.path.exists(local_path):
                    os.remove(local_path)
                logger.exception(f"Failed to download from {atlas_url}")
                raise RuntimeError(f"Failed to download from {atlas_url}") from e
        else:
            logger.info(f"Atlas already exists: {local_path}. Skipping download.")

        return local_path


class AtlasFetcher:
    """
    This class uses an AtlasFileHandler instance for file operations and provides atlas-specific
    fetchers. You can either download atlases using a URL, specify your atlas file, or use one of the
    supported atlas identifiers including volumetric and surface atlases such as:
      - "aal", "brodmann", "harvard-oxford", "juelich", "schaefer", "yeo", "aparc2009"
          
    Each fetcher returns a standardized dictionary. For volumetric atlases the keys are:
      'vol', 'hdr', 'labels', 'description', 'file'.
    For MNE annotation atlases, additional keys include:
      'vmap' (label-to-vertex mapping), 'labmap' (vertex-to-label mapping), and 'mni' (MNI coordinates).
    """

    # Fallback URL for Talairach atlas .
    ATLAS_URLS = {
        'talairach': 'https://www.talairach.org/talairach.nii',
        'aal': 'http://www.gin.cnrs.fr/wp-content/uploads/AAL3v2_for_SPM12.tar.gz',
    }

    def __init__(self, data_dir: str = None):
        """
        :param data_dir: Directory to store/download atlas files.
        """

        self.file_handler = AtlasFileHandler(data_dir=data_dir)
        self.data_dir = self.file_handler.data_dir
        self.nilearn_data = self.file_handler.nilearn_data
        self.subjects_dir = self.file_handler.subjects_dir
        from nilearn.datasets import fetch_atlas_aal, fetch_atlas_talairach, fetch_atlas_harvard_oxford, fetch_atlas_juelich, fetch_atlas_schaefer_2018, fetch_atlas_yeo_2011

        def _fetch_atlas_yeo_version(version='thick_17', **kwargs):
            from nilearn.datasets import fetch_atlas_yeo_2011
            fetched = self._fetch_atlas(fetch_atlas_yeo_2011, **kwargs)
            version = kwargs.get('version', 'thick_17')

            # Needs special care
            # thin/thick keys are the images
            # colors are the labels
            num = version.split('_')[-1]
            labels_file = fetched[f'colors_{num}']
            # read the labels file
            with open(labels_file, 'r') as f:
                lines = f.readlines()
            import re
            # replace any number of spaces with a single space in all lines
            lines = [re.sub(' +', ' ', line) for line in lines]
            labels = [line.strip().split(' ')[1] for idx, line in enumerate(lines)]
            output = {}
            output['labels'] = labels
            output['description'] = fetched['description']
            output['file'] = fetched[version]
            output['maps']=fetched[version] # this will be taken care of to make it an array later
            return output

        self._atlas_fetchers_nilearn = {
            'aal':  {'fetcher':fetch_atlas_aal,'default_kwargs': {'version': 'SPM12'}},
            'brodmann': {'fetcher':fetch_atlas_talairach,'default_kwargs': {'level_name': 'ba'}},
            'harvard-oxford': {'fetcher':fetch_atlas_harvard_oxford, 'default_kwargs': {'atlas_name': 'cort-maxprob-thr25-2mm'}},
            'juelich': {'fetcher':fetch_atlas_juelich, 'default_kwargs': {'atlas_name': 'maxprob-thr0-1mm'}},
            'schaefer': {'fetcher':fetch_atlas_schaefer_2018, 'default_kwargs': {}},
            'yeo': {'fetcher':_fetch_atlas_yeo_version, 'default_kwargs': {'version': 'thick_17'}},
        }

        self._atlas_fetchers_mne = {
            'aparc.a2009s': {
            'aliases': ['aparc.a2009s', 'destrieux', 'a2009s'],
            'fetcher': mne.datasets.fetch_fsaverage,
            'default_kwargs': {
                'verbose': True
            },
            'new_atlas_name': 'aparc.a2009s'
            },
            'aparc_sub': {
                'aliases': ['aparc_sub', 'aparc_subs'],
                'fetcher': mne.datasets.fetch_aparc_sub_parcellation,
                'default_kwargs': {
                    'subjects_dir': self.subjects_dir,
                    'verbose': True
                },
                'new_atlas_name': 'aparc_sub'
            },
            'aparc': {
                'aliases': ['aparc', 'desikan', 'dk'],
                'fetcher': mne.datasets.fetch_fsaverage,
                'default_kwargs': {
                    'verbose': True
                },
                'new_atlas_name': 'aparc'
            },
            'hcpmmp1': {
                'aliases': ['hcp', 'hcpmmp', 'hcpmmp1'],
                'fetcher': mne.datasets.fetch_hcp_mmp_parcellation,
                'default_kwargs': {
                    'subjects_dir': self.subjects_dir,
                    'accept': True,
                    'verbose': True
                },
                'new_atlas_name': 'HCPMMP1',
                'validation': lambda kwargs: kwargs.get('accept_hcp', False) # Validation: require user to pass accept_hcp=True
            },
            'aparc.DKTatlas': {
            'aliases': ['dkt', 'aparc.dktatlas'],
            'fetcher': mne.datasets.fetch_fsaverage,
            'default_kwargs': {
                'verbose': True
            },
            'new_atlas_name': 'aparc.DKTatlas'
            }
        }

    # ---- atlas fetchers using nilearn/mne ----

    def _fetch_atlas(self, fetcher, **kwargs):
        try:
            return fetcher(data_dir=self.file_handler.data_dir, **kwargs)
        except Exception as e:
            logger.error(f"Failed to fetch atlas using primary data_dir: {self.file_handler.data_dir}", e, exc_info=True)
            logger.info(f"Attempting to fetch atlas using nilearn_data: {self.file_handler.nilearn_data}")
            return fetcher(data_dir=self.file_handler.nilearn_data, **kwargs)
        except Exception as e:
            logger.error(f"Failed to fetch atlas using nilearn_data: {self.file_handler.nilearn_data}", e, exc_info=True)
            logger.info(f"Attempting to fetch atlas using subject's data_dir: {self.subjects_dir}")
            return fetcher(data_dir=self.subjects_dir, **kwargs)
    
    # ---- Public method ----

    def fetch_atlas(self, atlas_name: str, atlas_url: str = None, **kwargs):
        """
        Fetch an atlas given an atlas identifier.
        
        The identifier can be:
            (a) A URL (starting with http:// or https://),
            (b) A local file path 
            (c) Nifti1Image/NPZ object,
            (d) Nilearn or mne atlases atlases (e.g., "aal", "harvard-oxford", "aparc2009", "mne-annot", etc.).
        
        For MNE-based atlases (keys starting with "mne-"), additional keyword arguments are required:
            - subject: subject identifier (e.g., "fsaverage")
            - subjects_dir: path to the FreeSurfer subjects directory
        
        :param atlas_name: The atlas identifier.
        :param atlas_url: (Optional) Override URL for fetching the atlas.
        :param kwargs: Additional keyword arguments for the specific fetcher.
        :return: A standardized atlas dictionary.
        :raises ValueError: if the atlas identifier is not recognized.
        """
        # Case (a): URL provided.
        if atlas_url is not None and (atlas_url.startswith('http://') or atlas_url.startswith('https://')):
            return self.file_handler.fetch_from_local(self.file_handler.fetch_from_url(atlas_url, **kwargs))

        labels = kwargs.get("labels") or kwargs.get("label_file")

        # Case (b/c): Local file path or Nifti1Image object.
        atlas_file = kwargs.get("atlas_file", "None")
        if atlas_file is not None:
            if os.path.isfile(atlas_file):
                return self.file_handler.fetch_from_local(atlas_file, labels)
            elif os.path.isfile(os.path.join(self.data_dir, atlas_file)):
                return self.file_handler.fetch_from_local(os.path.join(self.data_dir, atlas_file), labels)
        
        atlas_image = kwargs.get("atlas_image")
        if isinstance(atlas_image, (Nifti1Image, np.ndarray)):
            output = self.file_handler.pack_vol_output(atlas_image)
            output['labels'] = self.file_handler.fetch_labels(labels)
            return output
    
        # Case (d): nilearn or mne atlases.
        key = atlas_name.lower()
        fetcher_nilearn = self._atlas_fetchers_nilearn.get(key, None)
        if fetcher_nilearn:
            try:
                this_kwargs = fetcher_nilearn['default_kwargs']
                this_kwargs.update(kwargs)
                if atlas_name != 'yeo':
                    fetched = self._fetch_atlas(fetcher_nilearn['fetcher'],**this_kwargs)
                else:
                    fetched = fetcher_nilearn['fetcher'](**this_kwargs)
                maphdr = self.file_handler.pack_vol_output(fetched["maps"])
                fetched.update(maphdr)
                fetched['vol']=np.squeeze(fetched['vol'])
                fetched['kwargs'] = this_kwargs
                if fetched.get('labels', None) is not None and isinstance(fetched['labels'], np.ndarray):
                    labels = fetched['labels'].tolist()
                    if isinstance(labels[0], bytes):
                        labels = [label.decode('utf-8') for label in labels]
                    fetched['labels'] = labels
                return fetched
            except Exception as e:
                logger.error(f"Failed to fetch atlas {key} using nilearn", e, exc_info=True)
                logger.warning(f"Attempting to fetch atlas {key} using url")
                if key in self.ATLAS_URLS:
                    return self.file_handler.fetch_from_url(self.ATLAS_URLS[key])
                else:
                    logger.error(f"Atlas {key} not found in available atlas urls")
        fetcher_mne = self._atlas_fetchers_mne.get(key, None)
        if fetcher_mne:
            try:
                this_kwargs = fetcher_mne['default_kwargs'].copy()
                this_kwargs.update(kwargs)
                if 'validation' in fetcher_mne and not fetcher_mne['validation'](this_kwargs):
                    raise ValueError("To fetch HCPMMP atlas, you must pass accept_hcp=True.")
                # Instead of calling the raw fetcher, we now use pack_surf_output
                subject = this_kwargs.get('subject', 'fsaverage')
                fetched = self.file_handler.pack_surf_output(
                    fetcher = fetcher_mne['fetcher'],
                    subject=subject,
                    parc=fetcher_mne['new_atlas_name'],
                    **this_kwargs
                )
                # fetched['kwargs'] = this_kwargs
                return fetched
            except Exception as e:
                logger.error(f"Failed to fetch atlas {key} using mne", e, exc_info=True)
        raise ValueError(f"Unrecognized atlas name '{atlas_name}'.") # ADD Available options: {list(self._atlas_fetchers.keys())