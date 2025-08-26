import os
import logging
import pickle
from typing import Optional, Union, List
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
    
    Attributes:
    :attr data_dir: Directory for storing downloaded atlas files.
    :attr subjects_dir: Directory for MNE data. Provided during
        initialization or inferred from ``mne.get_config('SUBJECTS_DIR')``.
    :attr nilearn_data: Directory for storing Nilearn data.
    """
    def __init__(self, data_dir: Optional[str] = None, subjects_dir: Optional[str] = None):
        """Initialize the file handler.

        Parameters
        ----------
        data_dir : str | None
            Directory for storing downloaded atlas files. If ``None``,
            ``~/coord2region`` is used. Relative paths are resolved relative
            to the user's home directory.
        subjects_dir : str | None
            Path to the FreeSurfer ``SUBJECTS_DIR``. When provided, this value
            is stored in :attr:`subjects_dir`. Otherwise, the value is looked
            up via :func:`mne.get_config` and a warning is emitted if no value
            can be found.
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
        if subjects_dir is not None:
            self.subjects_dir = subjects_dir
        else:
            self.subjects_dir = mne.get_config('SUBJECTS_DIR', None)
            if self.subjects_dir is None:
                logger.warning("Please provide a subjects_dir or set MNE's SUBJECTS_DIR in your environment.")

    def save(self, obj, filename: str):
        """
        Save an object to the data directory using pickle.

        :param obj: The object to save.
        :param filename: The name of the file to save the object to.
        :raises ValueError: If the data directory is not writable.
        :raises Exception: If there is an error during saving.
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

        :param filename: The name of the file to load the object from.
        :raises Exception: If there is an error during loading.
        :return: The loaded object or None if the file does not exist.
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

    def fetch_from_local(self, atlas_file: str, atlas_dir: str, labels: Union[str, List]):
        """
        Load an atlas from a local file.

        :param atlas_file: The name of the atlas file.
        :param atlas_dir: The directory where the atlas file is located.
        :param labels: The labels file or a list of labels.
        :raises FileNotFoundError: If the atlas file or labels file is not found.
        :raises Exception: If there is an error during loading.
        :return: A dictionary containing the atlas data.
        """
        logger.info(f"Loading local atlas file: {atlas_file}")
        found_path = next(
            (os.path.join(root, atlas_file) for root, _, files in os.walk(atlas_dir) if atlas_file in files),
            None
        )
        if found_path is None:
            raise FileNotFoundError(f"Atlas file {atlas_file} not found in {atlas_dir} or its subdirectories")
        logger.info(f"Atlas file found at {found_path}")

        output = pack_vol_output(found_path)
        if isinstance(labels, str):
            found_path = next(
                (os.path.join(root, labels) for root, _, files in os.walk(atlas_dir) if labels in files),
                None
            )
            if found_path is None:
                raise FileNotFoundError(f"Labels file {labels} not found in {atlas_dir} or its subdirectories")
            logger.info(f"Labels file found at {found_path}")
            output['labels'] = fetch_labels(found_path)
        elif isinstance(labels, list):
            output['labels'] = fetch_labels(labels)
        return output

    def fetch_from_url(self, atlas_url: str, **kwargs):
        """
        Download an atlas from a URL (if not already present) and return the local file path.

        :param atlas_url: The URL of the atlas file.
        :param kwargs: Additional arguments for the download.
        :raises RuntimeError: If the download fails.
        :return: The local path to the downloaded atlas file.
        :raises ValueError: If the data directory is not writable.
        :raises Exception: If there is an error during downloading.
        """
        import warnings
        warnings.warn("The file name is expected to be in the URL", UserWarning)
        import urllib.parse
        import requests
        import zipfile
        import tarfile
        import gzip
        import shutil

        parsed = urllib.parse.urlparse(atlas_url)
        file_name = os.path.basename(parsed.path)
        local_path = os.path.join(self.data_dir, file_name)

        if not os.path.exists(local_path):
            logger.info(f"Downloading atlas from {atlas_url}...")
            try:
                with requests.get(atlas_url, stream=True, timeout=30, verify=True) as r:
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

        # Check if the downloaded file is compressed and decompress if necessary.
        decompressed_path = local_path
        if zipfile.is_zipfile(local_path):
            logger.info(f"Extracting zip file {local_path}")
            extract_dir = os.path.join(self.data_dir, file_name.rstrip('.zip'))
            with zipfile.ZipFile(local_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                decompressed_path = extract_dir
        elif tarfile.is_tarfile(local_path):
            logger.info(f"Extracting tar archive {local_path}")
            # Remove possible extensions to form the extract directory name
            base_name = file_name
            for ext in ['.tar.gz', '.tgz', '.tar']:
                if base_name.endswith(ext):
                    base_name = base_name[:-len(ext)]
                    break
            extract_dir = os.path.join(self.data_dir, base_name)
            with tarfile.open(local_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_dir)
                decompressed_path = extract_dir
        elif local_path.endswith('.gz') and not local_path.endswith('.tar.gz'):
            logger.info(f"Decompressing gzip file {local_path}")
            decompressed_file = local_path[:-3]
            with gzip.open(local_path, 'rb') as f_in:
                with open(decompressed_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                decompressed_path = decompressed_file

        return decompressed_path