"""Utilities for managing atlas files.

This module handles downloading, caching, and loading atlas files used by
the mapping utilities. It provides helpers for retrieving label information
and packing volumetric atlas outputs.
"""

import os
import logging
import pickle
from typing import Optional, Union, List
import mne
from .utils import fetch_labels, pack_vol_output

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AtlasFileHandler:
    """Handle file operations for atlas fetching.

    Parameters
    ----------
    data_dir : str or None, optional
        Directory for downloaded atlas files. Defaults to ``~/coord2region``.
    subjects_dir : str or None, optional
        FreeSurfer ``SUBJECTS_DIR``. If ``None``, the value is inferred from
        :func:`mne.get_config`.

    Attributes
    ----------
    data_dir : str
        Directory where atlas files are stored.
    subjects_dir : str or None
        Path to the FreeSurfer subjects directory.
    nilearn_data : str
        Directory for caching Nilearn datasets.

    Examples
    --------
    >>> handler = AtlasFileHandler()  # doctest: +SKIP
    >>> handler.data_dir  # doctest: +SKIP
    '/home/user/coord2region'
    """

    def __init__(
        self, data_dir: Optional[str] = None, subjects_dir: Optional[str] = None
    ):
        """Initialize the file handler.

        data_dir : str or None, optional
            Directory for storing downloaded atlas files. Defaults to
            ``~/coord2region``.
        subjects_dir : str or None, optional
            Path to the FreeSurfer ``SUBJECTS_DIR``. If ``None``, the value is
            looked up via :func:`mne.get_config`.

        Raises
        ------
        ValueError
            If the data directory cannot be created or is not writable.

        Examples
        --------
        >>> AtlasFileHandler()  # doctest: +SKIP
        """
        home_dir = os.path.expanduser("~")
        if data_dir is None:
            self.data_dir = os.path.join(home_dir, "coord2region")
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

        self.nilearn_data = os.path.join(home_dir, "nilearn_data")
        if subjects_dir is not None:
            self.subjects_dir = subjects_dir
        else:
            self.subjects_dir = mne.get_config("SUBJECTS_DIR", None)
            if self.subjects_dir is None:
                logger.warning(
                    "Please provide a subjects_dir or set MNE's SUBJECTS_DIR "
                    "in your environment."
                )

    def save(self, obj, filename: str):
        """Save an object to the data directory using pickle.

        Parameters
        ----------
        obj : Any
            The object to serialize.
        filename : str
            Name of the file to save the object to.

        Raises
        ------
        ValueError
            If the data directory is not writable.
        Exception
            If there is an error during saving.

        Examples
        --------
        >>> handler = AtlasFileHandler()
        >>> handler.save({'a': 1}, 'example.pkl')  # doctest: +SKIP
        """
        filepath = os.path.join(self.data_dir, filename)
        try:
            with open(filepath, "wb") as f:
                pickle.dump(obj, f)
            logger.info(f"Object saved to {filepath}")
        except Exception as e:
            logger.exception(f"Error saving object to {filepath}: {e}")
            raise

    def load(self, filename: str):
        """Load an object from the data directory.

        Parameters
        ----------
        filename : str
            Name of the file to load the object from.

        Returns
        -------
        object or None
            The loaded object, or ``None`` if the file does not exist.

        Raises
        ------
        Exception
            If there is an error during loading.

        Examples
        --------
        >>> handler = AtlasFileHandler()
        >>> handler.load('missing.pkl')  # doctest: +SKIP
        None
        """
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, "rb") as f:
                    obj = pickle.load(f)
                logger.info(f"Object loaded from {filepath}")
                return obj
            except Exception as e:
                logger.exception(f"Error loading object from {filepath}: {e}")
                raise
        else:
            return None

    def fetch_from_local(
        self, atlas_file: str, atlas_dir: str, labels: Union[str, List]
    ):
        """Load an atlas from a local file.

        Parameters
        ----------
        atlas_file : str
            The name of the atlas file.
        atlas_dir : str
            Directory where the atlas file is located.
        labels : str or list
            Labels file or a list of label names.

        Returns
        -------
        dict
            Dictionary containing the atlas data.

        Raises
        ------
        FileNotFoundError
            If the atlas or labels file is not found.
        Exception
            If there is an error during loading.

        Examples
        --------
        >>> handler = AtlasFileHandler()
        >>> handler.fetch_from_local('atlas.nii.gz', '.', ['A', 'B'])  # doctest: +SKIP
        {'vol': array(...), 'hdr': array(...), 'labels': ['A', 'B']}
        """
        logger.info(f"Loading local atlas file: {atlas_file}")
        found_path = next(
            (
                os.path.join(root, atlas_file)
                for root, _, files in os.walk(atlas_dir)
                if atlas_file in files
            ),
            None,
        )
        if found_path is None:
            raise FileNotFoundError(
                f"Atlas file {atlas_file} not found in {atlas_dir} or its "
                "subdirectories"
            )
        logger.info(f"Atlas file found at {found_path}")

        output = pack_vol_output(found_path)
        if isinstance(labels, str):
            found_path = next(
                (
                    os.path.join(root, labels)
                    for root, _, files in os.walk(atlas_dir)
                    if labels in files
                ),
                None,
            )
            if found_path is None:
                raise FileNotFoundError(
                    f"Labels file {labels} not found in {atlas_dir} or its "
                    "subdirectories"
                )
            logger.info(f"Labels file found at {found_path}")
            output["labels"] = fetch_labels(found_path)
        elif isinstance(labels, list):
            output["labels"] = fetch_labels(labels)
        return output

    def fetch_from_url(self, atlas_url: str, **kwargs):
        """Download an atlas from a URL.

        Parameters
        ----------
        atlas_url : str
            The URL of the atlas file.
        **kwargs
            Additional arguments for the download.

        Returns
        -------
        str
            Local path to the downloaded (and possibly decompressed) file.

        Raises
        ------
        RuntimeError
            If the download fails.
        ValueError
            If the data directory is not writable.
        Exception
            If there is an error during downloading.

        Examples
        --------
        >>> handler = AtlasFileHandler()
        >>> handler.fetch_from_url('http://example.com/atlas.nii.gz')  # doctest: +SKIP
        '/path/to/atlas.nii.gz'
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
                    with open(local_path, "wb") as f:
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
            extract_dir = os.path.join(self.data_dir, file_name.rstrip(".zip"))
            with zipfile.ZipFile(local_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
                decompressed_path = extract_dir
        elif tarfile.is_tarfile(local_path):
            logger.info(f"Extracting tar archive {local_path}")
            # Remove possible extensions to form the extract directory name
            base_name = file_name
            for ext in [".tar.gz", ".tgz", ".tar"]:
                if base_name.endswith(ext):
                    base_name = base_name[:-len(ext)]
                    break
            extract_dir = os.path.join(self.data_dir, base_name)
            with tarfile.open(local_path, "r:*") as tar_ref:
                tar_ref.extractall(extract_dir)
                decompressed_path = extract_dir
        elif local_path.endswith(".gz") and not local_path.endswith(".tar.gz"):
            logger.info(f"Decompressing gzip file {local_path}")
            decompressed_file = local_path[:-3]
            with gzip.open(local_path, "rb") as f_in:
                with open(decompressed_file, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                decompressed_path = decompressed_file

        return decompressed_path
