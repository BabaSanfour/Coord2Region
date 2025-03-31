import os
import logging
import numpy as np
import mne
from nibabel.nifti1 import Nifti1Image
from .file_handler import AtlasFileHandler
from .utils import pack_vol_output, pack_surf_output

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AtlasFetcher:
    """
    Fetches atlases from various sources (local files, URLs, Nilearn datasets, or MNE annotations).
    
    Supported identifiers include volumetric atlases (e.g., "aal", "harvard-oxford") and
    surface-based atlases (e.g., "aparc", "brodmann"). A fallback mechanism is provided
    when an atlas is available via both Nilearn and MNE.
    """
    ATLAS_URLS = {
        'talairach': 'https://www.talairach.org/talairach.nii',
        'aal': {
            'atlas_url': 'http://www.gin.cnrs.fr/wp-content/uploads/AAL3v2_for_SPM12.tar.gz',
            'atlas_file': 'AAL3v1.nii.gz',
            'labels': 'AAL3v1.xml',
        },
    }

    def __init__(self, data_dir: str = None):
        self.file_handler = AtlasFileHandler(data_dir=data_dir)
        self.data_dir = self.file_handler.data_dir
        self.nilearn_data = self.file_handler.nilearn_data
        self.subjects_dir = self.file_handler.subjects_dir

        from nilearn.datasets import (fetch_atlas_destrieux_2009, fetch_atlas_aal,
                                      fetch_atlas_talairach, fetch_atlas_harvard_oxford,
                                      fetch_atlas_juelich, fetch_atlas_schaefer_2018,
                                      fetch_atlas_yeo_2011, fetch_atlas_pauli_2017,
                                      fetch_atlas_basc_multiscale_2015)
        self._atlas_fetchers_nilearn = {
            'aal':  {'fetcher': fetch_atlas_aal, 'default_kwargs': {'version': '3v2'}},
            'brodmann': {'fetcher': fetch_atlas_talairach, 'default_kwargs': {'level_name': 'ba'}},
            'harvard-oxford': {'fetcher': fetch_atlas_harvard_oxford, 'default_kwargs': {'atlas_name': 'cort-maxprob-thr25-2mm'}},
            'juelich': {'fetcher': fetch_atlas_juelich, 'default_kwargs': {'atlas_name': 'maxprob-thr0-1mm'}},
            'schaefer': {'fetcher': fetch_atlas_schaefer_2018, 'default_kwargs': {'n_rois': 400, 'yeo_networks': 7, 'resolution_mm': 1}},
            'yeo': {'fetcher': fetch_atlas_yeo_2011, 'default_kwargs': {'n_networks': 7, 'thickness': 'thick'}},
            'destrieux': {'fetcher': fetch_atlas_destrieux_2009, 'default_kwargs': {'lateralized': True}},
            'pauli': {'fetcher': fetch_atlas_pauli_2017, 'default_kwargs': {'atlas_type': 'deterministic'}},
            'basc': {'fetcher': fetch_atlas_basc_multiscale_2015, 'default_kwargs': {'resolution': 444, 'version': 'sym'}},
        }

        from nilearn.datasets import (fetch_coords_dosenbach_2010,
                                      fetch_coords_power_2011,
                                      fetch_coords_seitzman_2018)
        self._coords_fetchers_nilearn = {
            'dosenbach': { 'fetcher': fetch_coords_dosenbach_2010, 'default_kwargs': {}},
            'power': { 'fetcher': fetch_coords_power_2011, 'default_kwargs': {}},
            'seitzman': { 'fetcher': fetch_coords_seitzman_2018, 'default_kwargs': {}},
        }

        self._atlas_fetchers_mne = {
            'brodmann': {'fetcher': None, 'default_kwargs': {'version': 'PALS_B12_Brodmann'}},
            'human-connectum project': {'fetcher': mne.datasets.fetch_hcp_mmp_parcellation, 'default_kwargs': {'version': 'HCPMMP1_combined'}},
            'pals_b12_lobes': {'fetcher': None, 'default_kwargs': {'version': 'PALS_B12_Lobes'}},
            'pals_b12_orbitofrontal': {'fetcher': None, 'default_kwargs': {'version': 'PALS_B12_OrbitoFrontal'}},
            'pals_b12_visuotopic': {'fetcher': None, 'default_kwargs': {'version': 'PALS_B12_Visuotopic'}},
            'aparc_sub': {'fetcher': mne.datasets.fetch_aparc_sub_parcellation, 'default_kwargs': {}},
            'aparc': {'fetcher': None, 'default_kwargs': {}},
            'aparc.a2009s': {'fetcher': None, 'default_kwargs': {}},
            'aparc.a2005s': {'fetcher': None, 'default_kwargs': {}},
            'oasis.chubs': {'fetcher': None, 'default_kwargs': {}},
            'yeo2011': {'fetcher': None, 'default_kwargs': {'version': 'Yeo2011_17Networks_N1000'}},
        }

    def _get_description(self, atlas_name, fetched, kwargs):
        description = {}
        description.update(kwargs)
        description["atlas_name"] = atlas_name
        description.update({k: v for k, v in {
            'atlas_type': fetched.get('atlas_type'),
            'atlas_template': fetched.get('template'),
            'networks': fetched.get('networks'),
            'radius': fetched.get('radius'),
        }.items() if v is not None})
        version = kwargs.get('atlas_name') or kwargs.get('version')
        template = fetched.get('template', '')
        description['coordinate system'] = 'MNI' if 'MNI' in template else kwargs.get('coordinate system', 'Unknown')
        description['type'] = kwargs.get('type', 'volumetric')
        if version is not None:
            description['version'] = version
        return description

    def _fetch_coords_nilearn(self, atlas_name, fetcher_nilearn, **kwargs):
        this_kwargs = fetcher_nilearn['default_kwargs'].copy()
        this_kwargs.update(kwargs)
        fetched = fetcher_nilearn['fetcher'](**this_kwargs)
        description = self._get_description(atlas_name, fetched, {"type": "coords", 'coordinate system': 'MNI'})
        labels = fetched.get('labels') or fetched.get('regions') 
        if labels is None: 
            labels = fetched['rois']['roi'].tolist()
        return {
            'vol': fetched['rois'],
            'hdr': None,
            'labels': labels,
            'description': description,
        }
    
    def _fetch_atlas_nilearn(self, atlas_name, fetcher_nilearn, **kwargs):
        this_kwargs = fetcher_nilearn['default_kwargs'].copy()
        this_kwargs.update(kwargs)
        fetched = fetcher_nilearn['fetcher'](**this_kwargs)
        maphdr = pack_vol_output(fetched["maps"])
        fetched.update(maphdr)
        fetched['vol'] = np.squeeze(fetched['vol'])
        fetched['description'] = self._get_description(atlas_name, fetched, this_kwargs)
        if fetched.get('labels', None) is not None and isinstance(fetched['labels'], np.ndarray):
            labels = fetched['labels'].tolist()
            if labels and isinstance(labels[0], bytes):
                labels = [label.decode('utf-8') for label in labels]
            fetched['labels'] = labels
        return {
            'vol': fetched['vol'],
            'hdr': fetched['hdr'],
            'labels': fetched['labels'],
            'description': fetched['description'],
        }
    
    def _fetch_atlas_mne(self, atlas_name, fetcher_mne, **kwargs):
        kwargs['subject'] = kwargs.get('subject', 'fsaverage')
        this_kwargs = fetcher_mne['default_kwargs'].copy()
        this_kwargs.update(kwargs)
        atlas_name_mne = this_kwargs.pop('version', atlas_name)
        fetched = pack_surf_output(
            atlas_name=atlas_name_mne,
            fetcher=fetcher_mne['fetcher'],
            **this_kwargs
        )
        this_kwargs.update({'type': 'surface'})
        this_kwargs['coordinate system'] = 'MNI'
        this_kwargs['version'] = atlas_name_mne
        description = self._get_description(atlas_name, fetcher_mne, this_kwargs)
        fetched['description'] = description
        return fetched

    def _fetch_from_url(self, atlas_name, atlas_url, **kwargs):
        """
        Fetch an atlas from a URL.
        """
        local_path = self.file_handler.fetch_from_url(atlas_url, **kwargs)
        output = self.file_handler.fetch_from_local(kwargs.get("atlas_file"), local_path, kwargs.get("labels"))
        output['description'] = self._get_description(atlas_name, output, kwargs)
        return output

    def list_available_atlases(self):
        """
        Returns a sorted list of available atlas identifiers.
        """
        atlases_nilearn = list(self._atlas_fetchers_nilearn.keys())
        atlases_coords = list(self._coords_fetchers_nilearn.keys())
        atlases_mne = list(self._atlas_fetchers_mne.keys())
        atlases_urls = list(self.ATLAS_URLS.keys())
        all_atlases = set(atlases_nilearn + atlases_coords + atlases_mne + atlases_urls)
        return sorted(all_atlases)

    def fetch_atlas(self, atlas_name: str, atlas_url: str = None, prefer: str = "nilearn", **kwargs):
        """
        Fetch an atlas given an atlas identifier.

        The identifier can be a URL, local file path, or a known atlas name.
        The 'prefer' flag allows the user to choose the primary source ("nilearn" or "mne").
        """
        key = atlas_name.lower()
        if atlas_url is not None and atlas_url.startswith(('http://', 'https://')):
            return self._fetch_from_url(key, atlas_url, **kwargs)
        if key in self.ATLAS_URLS:
            kwargs.update(self.ATLAS_URLS[key])
            return self._fetch_from_url(key, **kwargs)

        # Local file or image cases.
        atlas_file = kwargs.get("atlas_file", None)
        if atlas_file is not None:
            if os.path.isfile(atlas_file):
                return self.file_handler.fetch_from_local(atlas_file, kwargs.get("labels"))
            elif os.path.isfile(os.path.join(self.data_dir, atlas_file)):
                return self.file_handler.fetch_from_local(os.path.join(self.data_dir, atlas_file), kwargs.get("labels"))

        atlas_image = kwargs.get("atlas_image")
        if isinstance(atlas_image, (Nifti1Image, np.ndarray)):
            output = pack_vol_output(atlas_image)
            output['labels'] = kwargs.get("labels")
            return output

        fetcher_nilearn = self._atlas_fetchers_nilearn.get(key, None)
        fetcher_coords = self._coords_fetchers_nilearn.get(key, None)
        fetcher_mne = self._atlas_fetchers_mne.get(key, None)

        if prefer == "nilearn" and fetcher_nilearn:
            try:
                return self._fetch_atlas_nilearn(key, fetcher_nilearn, **kwargs)
            except Exception as e:
                logger.warning(f"Nilearn fetcher failed for atlas {key}: {e}")
                if fetcher_mne:
                    return self._fetch_atlas_mne(key, fetcher_mne, **kwargs)
        elif prefer == "mne" and fetcher_mne:
            try:
                return self._fetch_atlas_mne(key, fetcher_mne, **kwargs)
            except Exception as e:
                logger.warning(f"MNE fetcher failed for atlas {key}: {e}")
                if fetcher_nilearn:
                    return self._fetch_atlas_nilearn(key, fetcher_nilearn, **kwargs)

        if fetcher_nilearn:
            return self._fetch_atlas_nilearn(key, fetcher_nilearn, **kwargs)
        if fetcher_coords:
            return self._fetch_coords_nilearn(key, fetcher_coords, **kwargs)
        if fetcher_mne:
            return self._fetch_atlas_mne(key, fetcher_mne, **kwargs)

        raise ValueError(f"Unrecognized atlas name '{atlas_name}'. Available options: {self.list_available_atlases()}")