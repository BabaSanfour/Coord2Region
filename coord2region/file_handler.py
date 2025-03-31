import os
import logging
import pickle
from typing import Optional
import mne
from .utils import pack_vol_output, pack_surf_output, fetch_labels

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AtlasFileHandler:
    """
    Handles file operations for atlas fetching.

    Provides utilities for:
      - Loading local atlas files.
      - Downloading atlas files from a URL.
      - Caching objects to reduce repeated computation.
    """
    def __init__(self, data_dir: Optional[str] = None, subjects_dir: Optional[str] = None):
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
        self.subjects_dir = mne.get_config('SUBJECTS_DIR', None)
        if subjects_dir is None:
            logger.warning("Please provide a subjects_dir or set MNE's SUBJECTS_DIR in your environment.")

    def save(self, obj, filename: str):
        """
        Save an object to the data directory using pickle.
        """
        filepath = os.path.join(self.data_dir, filename)
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(obj, f)
            logger.info(f"Object saved to {filepath}")
        except Exception as e:
            logger.exception(f"Error saving object to {filepath}: {e}")
            raise

    def load(self, filename: str):
        """
        Load an object from the data directory.
        Returns None if the file is not found.
        """
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    obj = pickle.load(f)
                logger.info(f"Object loaded from {filepath}")
                return obj
            except Exception as e:
                logger.exception(f"Error loading object from {filepath}: {e}")
                raise
        else:
            return None

    def fetch_from_local(self, atlas: str, labels: str):
        """
        Load an atlas from a local file.
        """
        logger.info(f"Loading local atlas file: {atlas}")
        output = pack_vol_output(atlas)
        output['labels'] = fetch_labels(labels)
        return output

    def fetch_from_url(self, atlas_url: str, **kwargs):
        """
        Download an atlas from a URL (if not already present) and return the local file path.
        """
        import warnings
        warnings.warn("The file name is expected to be in the URL", UserWarning)
        import urllib.parse
        import requests

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
