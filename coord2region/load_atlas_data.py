import os
import logging
import requests
import numpy as np
import nibabel as nib
from coord2region.utils import _fetch_labels
logger = logging.getLogger(__name__)

ATLAS_URLS = {
    'talairach': 'https://www.talairach.org/talairach.nii',
}

def fetch_atlas(atlas_input, data_dir=None, version=None, **kwargs):
    """
    Fetch an atlas in one of three ways:
      (a) By known atlas name -> use Nilearn fetchers or custom URLs.
      (b) By URL -> download if not present, then load.
      (c) By local file path -> load directly.

    Parameters
    ----------
    atlas_input : str
        Either a known atlas name (e.g. 'aal', 'brodmann'),
        a URL (http://... or https://...),
        or a local file path (.nii, .nii.gz, .npz).
    data_dir : str or None
        Where to store or look for the atlas file.
        If None, uses Nilearn's default data directory for known atlas fetchers,
        or the current working directory for direct URL downloads.
    version : str or None
        Optional version info for some fetchers (e.g. 'SPM12' for AAL).
    **kwargs :
        Additional arguments that might be passed to certain fetchers.

    Returns
    -------
    atlas_dict : dict
        A dictionary with the following keys
            - 'vol': np.ndarray, shape (i, j, k)
            - 'hdr': np.ndarray, shape (4, 4); affine transform to get to world coordinates
            - 'labels': dict or None
            - 'description': str
            - 'file': str (path to the loaded file)

    Raises
    ------
    ValueError:
        If atlas_input is unrecognized as a name and not a valid file/URL.
    RuntimeError:
        If a download fails.
    """
    if data_dir is not None:
        os.makedirs(data_dir, exist_ok=True)
    else:
        data_dir = data_dir or os.getcwd()

    def _pack_output(fname, desc=None):
        """
        Load the file into a nibabel image and return
        a dictionary like Nilearn's fetchers do.
        """
        path = os.path.abspath(fname)
        _, ext = os.path.splitext(fname)
        ext = ext.lower()
        
        if ext in ['.nii', '.gz', '.nii.gz']:
            img = nib.load(fname)
            vol_data = img.get_fdata(dtype=np.float32)  # or float64 if you prefer
            hdr_matrix = img.affine
            labels = _fetch_labels(path)
            return {'vol': vol_data, 'hdr': hdr_matrix, 'labels': labels, "description": desc, "file": fname}

        elif ext == '.npz':
            arch = np.load(path, allow_pickle=True)
            vol_data = arch['vol']
            hdr_matrix = arch['hdr']
            labels = arch['labels'] if 'labels' in arch else None
            index = arch['index'] if 'index' in arch else None
            labels = {idx: name for idx, name in zip(index, labels)} if labels is not None else None
            return {'vol': vol_data, 'hdr': hdr_matrix, 'labels': labels, "description": desc, "file": fname}        
        else:
            raise ValueError(f"Unrecognized file format '{ext}' for path: {path}")


    # 1. Check if atlas_input is a local file path (scenario c)
    if os.path.isfile(atlas_input):
        logger.info(f"Loading local atlas file: {atlas_input}")
        return _pack_output(atlas_input, desc="Local file")

    # 2. Check if atlas_input is a URL (scenario b)
    if atlas_input.startswith('http://') or atlas_input.startswith('https://'):
        import urllib.parse
        url_path = urllib.parse.urlparse(atlas_input).path
        if kwargs.get('file_name'):
            file_name = kwargs['file_name']
        else:
            file_name = os.path.basename(url_path)
        if not file_name:  # fallback if URL doesn't have a basename
            file_name = 'atlas_download.nii.gz'
        local_path = os.path.join(data_dir, file_name)

        # If not present, download
        if not os.path.exists(local_path):
            logger.info(f"Downloading atlas from {atlas_input}...")
            try:
                with requests.get(atlas_input, stream=True) as r:
                    r.raise_for_status()
                    with open(local_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                logger.info(f"Atlas downloaded to {local_path}")
            except Exception as e:
                # Clean up partial download
                if os.path.exists(local_path):
                    os.remove(local_path)
                raise RuntimeError(f"Failed to download from {atlas_input}") from e
        else:
            logger.info(f"Atlas already exists: {local_path}. Skipping download.")

        return _pack_output(local_path, desc=f"Downloaded from {atlas_input}")

    # 3. Otherwise, treat atlas_input as a known atlas name (scenario a)
    #    We'll first handle some that Nilearn provides direct fetchers for.
    #    If not found, we fall back to custom ATLAS_URLS or raise an error.

    # (a) Example: 'aal'
    if atlas_input.lower() == "aal":
        from nilearn.datasets import fetch_atlas_aal
        version_ = version or "SPM12"  # default
        fetched = fetch_atlas_aal(version=version_, data_dir=data_dir, **kwargs)
        return _pack_output(fetched["maps"], desc="AAL")
    # (b) Example: 'destrieux' (surface atlas)
    if atlas_input.lower() == "destrieux":
        from nilearn.datasets import fetch_atlas_surf_destrieux
        fetched = fetch_atlas_surf_destrieux(data_dir=data_dir, **kwargs)
        atlas_dict = {
            'maps': None,
            'labels': fetched['labels'],
            'description': 'Destrieux surface atlas',
            'file': fetched['annot_file']
        }
        return atlas_dict

    # (c) Example: 'brodmann'
    if atlas_input.lower() == "brodmann":
        try:
            from nilearn.datasets import fetch_atlas_talairach
            fetched = fetch_atlas_brodmann(level_name='ba', data_dir=data_dir)
            atlas_dict = {
                'maps': nib.load(fetched['maps']),
                'labels': None,  # or something custom
                'description': 'Brodmann (Nilearn)',
                'file': fetched['maps']
            }
            return atlas_dict
        except Exception as e:
            logger.warning("Nilearn fetch_atlas_brodmann failed; attempting manual URL download.")
            # If you want a fallback custom URL approach, do it here:
            # For example:
            url = ATLAS_URLS.get('talairach')
            local_fname = os.path.join(data_dir, "talairach.nii")
            if not os.path.exists(local_fname):
                logger.info("Downloading Talairach brodmann atlas (fallback for 'brodmann')")
                try:
                    with requests.get(url, stream=True) as r:
                        r.raise_for_status()
                        with open(local_fname, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                except:
                    raise RuntimeError("Download failed, please check your internet connection")
            return _pack_output(local_fname, desc="brodmann fallback")

    # If we get here, the atlas_input was not recognized
    raise ValueError(f"Unrecognized atlas_input '{atlas_input}'.\n"
                     f"Possible reasons:\n"
                     f"- Not a valid local file.\n"
                     f"- Not a recognized URL.\n"
                     f"- Not in known fetchers or ATLAS_URLS.\n"
                     f"Available custom keys in ATLAS_URLS: {list(ATLAS_URLS.keys())}")