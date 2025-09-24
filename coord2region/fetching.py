"""Fetch atlases from multiple sources.

This module provides the :class:`AtlasFetcher` class to download and manage
brain atlases from various sources including Nilearn, MNE, and direct URLs.
"""

import os
import logging
from pathlib import Path
import numpy as np
import nilearn.datasets
import mne
from nibabel.nifti1 import Nifti1Image
from .utils.file_handler import AtlasFileHandler
from .utils import pack_vol_output, pack_surf_output

logger = logging.getLogger(__name__)


class AtlasFetcher:
    """Fetch atlases from multiple sources.

    Parameters
    ----------
    data_dir : str or None, optional
        Directory where downloaded files are cached. Defaults to
        ``~/coord2region``.

    Attributes
    ----------
    file_handler : AtlasFileHandler
        Helper for file operations.
    data_dir : str
        Directory for storing downloaded atlas files.
    subjects_dir : str or None
        Directory for MNE data.
    nilearn_data : str
        Directory for storing Nilearn data.
    _atlas_fetchers_nilearn : dict
        Mapping of atlas names to Nilearn fetcher functions.
    _coords_fetchers_nilearn : dict
        Mapping of atlas names to Nilearn coordinate fetchers.
    _atlas_fetchers_mne : dict
        Mapping of atlas names to MNE fetcher functions.

    Examples
    --------
    >>> fetcher = AtlasFetcher()  # doctest: +SKIP
    >>> 'aal' in fetcher.list_available_atlases()  # doctest: +SKIP
    True
    """

    ATLAS_URLS = {
        "talairach": "https://www.talairach.org/talairach.nii",
        "aal": {
            "atlas_url": (
                "http://www.gin.cnrs.fr/wp-content/uploads/" "AAL3v2_for_SPM12.tar.gz"
            ),
            "atlas_file": "AAL3v1.nii.gz",
            "labels": "AAL3v1.xml",
        },
    }

    def __init__(self, data_dir: str = None):
        """Initialize the fetcher.

        Parameters
        ----------
        data_dir : str or None, optional
            Directory where downloaded files are cached. Defaults to
            ``~/coord2region``.

        Examples
        --------
        >>> AtlasFetcher()  # doctest: +SKIP
        """
        self.file_handler = AtlasFileHandler(data_dir=data_dir)
        self.data_dir = self.file_handler.data_dir
        self.nilearn_data = self.file_handler.nilearn_data
        self.subjects_dir = self.file_handler.subjects_dir

        from nilearn.datasets import (
            fetch_atlas_destrieux_2009,
            fetch_atlas_aal,
            fetch_atlas_talairach,
            fetch_atlas_harvard_oxford,
            fetch_atlas_juelich,
            fetch_atlas_schaefer_2018,
            fetch_atlas_yeo_2011,
            fetch_atlas_pauli_2017,
            fetch_atlas_basc_multiscale_2015,
        )

        self._atlas_fetchers_nilearn = {
            "aal": {"fetcher": fetch_atlas_aal, "default_kwargs": {"version": "3v2"}},
            "brodmann": {
                "fetcher": fetch_atlas_talairach,
                "default_kwargs": {"level_name": "ba"},
            },
            "harvard-oxford": {
                "fetcher": fetch_atlas_harvard_oxford,
                "default_kwargs": {"atlas_name": "cort-maxprob-thr25-2mm"},
            },
            "juelich": {
                "fetcher": fetch_atlas_juelich,
                "default_kwargs": {"atlas_name": "maxprob-thr0-1mm"},
            },
            "schaefer": {
                "fetcher": fetch_atlas_schaefer_2018,
                "default_kwargs": {
                    "n_rois": 400,
                    "yeo_networks": 7,
                    "resolution_mm": 1,
                },
            },
            "yeo": {
                "fetcher": fetch_atlas_yeo_2011,
                "default_kwargs": {"n_networks": 7, "thickness": "thick"},
            },
            "destrieux": {
                "fetcher": fetch_atlas_destrieux_2009,
                "default_kwargs": {"lateralized": True},
            },
            "pauli": {
                "fetcher": fetch_atlas_pauli_2017,
                "default_kwargs": {"atlas_type": "deterministic"},
            },
            "basc": {
                "fetcher": fetch_atlas_basc_multiscale_2015,
                "default_kwargs": {"resolution": 444, "version": "sym"},
            },
        }

        from nilearn.datasets import (
            fetch_coords_dosenbach_2010,
            fetch_coords_power_2011,
            fetch_coords_seitzman_2018,
        )

        self._coords_fetchers_nilearn = {
            "dosenbach": {"fetcher": fetch_coords_dosenbach_2010, "default_kwargs": {}},
            "power": {"fetcher": fetch_coords_power_2011, "default_kwargs": {}},
            "seitzman": {"fetcher": fetch_coords_seitzman_2018, "default_kwargs": {}},
        }

        hcp_fetcher = getattr(mne.datasets, "fetch_hcp_mmp_parcellation", None)
        aparc_sub_fetcher = getattr(mne.datasets, "fetch_aparc_sub_parcellation", None)

        self._atlas_fetchers_mne = {
            "brodmann": {
                "fetcher": None,
                "default_kwargs": {"version": "PALS_B12_Brodmann"},
            },
            "human-connectum project": {
                "fetcher": hcp_fetcher,
                "default_kwargs": {"version": "HCPMMP1_combined"},
            },
            "pals_b12_lobes": {
                "fetcher": None,
                "default_kwargs": {"version": "PALS_B12_Lobes"},
            },
            "pals_b12_orbitofrontal": {
                "fetcher": None,
                "default_kwargs": {"version": "PALS_B12_OrbitoFrontal"},
            },
            "pals_b12_visuotopic": {
                "fetcher": None,
                "default_kwargs": {"version": "PALS_B12_Visuotopic"},
            },
            "aparc_sub": {
                "fetcher": aparc_sub_fetcher,
                "default_kwargs": {},
            },
            "aparc": {"fetcher": None, "default_kwargs": {}},
            "aparc.a2009s": {"fetcher": None, "default_kwargs": {}},
            "aparc.a2005s": {"fetcher": None, "default_kwargs": {}},
            "oasis.chubs": {"fetcher": None, "default_kwargs": {}},
            "yeo2011": {
                "fetcher": None,
                "default_kwargs": {"version": "Yeo2011_17Networks_N1000"},
            },
        }

    def _get_description(self, atlas_name: str, fetched: dict, kwargs: dict):
        """Generate a description dictionary for an atlas.

        Parameters
        ----------
        atlas_name : str
            The name of the atlas.
        fetched : dict
            Data returned by the fetcher.
        kwargs : dict
            Additional keyword arguments describing the atlas.

        Returns
        -------
        dict
            Dictionary containing atlas metadata.

        Raises
        ------
        None

        Examples
        --------
        >>> fetcher = AtlasFetcher()  # doctest: +SKIP
        >>> fetcher._get_description('aal', {}, {})['atlas_name']  # doctest: +SKIP
        'aal'
        """
        description = {}
        description.update(kwargs)
        description["atlas_name"] = atlas_name
        description.update(
            {
                k: v
                for k, v in {
                    "atlas_type": fetched.get("atlas_type"),
                    "atlas_template": fetched.get("template"),
                    "networks": fetched.get("networks"),
                    "radius": fetched.get("radius"),
                }.items()
                if v is not None
            }
        )
        version = kwargs.get("atlas_name") or kwargs.get("version")
        template = fetched.get("template", "")
        description["coordinate system"] = (
            "MNI" if "MNI" in template else kwargs.get("coordinate system", "Unknown")
        )
        description["type"] = kwargs.get("type", "volumetric")
        if version is not None:
            description["version"] = version
        return description

    def _fetch_coords_nilearn(
        self, atlas_name: str, fetcher_nilearn: nilearn.datasets, **kwargs
    ):
        """Fetch atlas coordinates using Nilearn.

        Parameters
        ----------
        atlas_name : str
            The name of the atlas.
        fetcher_nilearn : nilearn.datasets
            Nilearn fetcher function.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        dict
            Atlas coordinates and description.

        Raises
        ------
        Exception
            If fetching fails.

        Examples
        --------
        >>> fetcher._fetch_coords_nilearn(
        ...     'dosenbach', fetcher._coords_fetchers_nilearn['dosenbach']
        ... )['labels'][0]  # doctest: +SKIP
        'Anterior cingulate'
        """
        this_kwargs = fetcher_nilearn["default_kwargs"].copy()
        this_kwargs.update(kwargs)
        fetched = fetcher_nilearn["fetcher"](**this_kwargs)
        description = self._get_description(
            atlas_name, fetched, {"type": "coords", "coordinate system": "MNI"}
        )
        labels = fetched.get("labels") or fetched.get("regions")
        if labels is None:
            labels = fetched["rois"]["roi"].tolist()
        return {
            "vol": fetched["rois"],
            "hdr": None,
            "labels": labels,
            "description": description,
        }

    def _fetch_atlas_nilearn(
        self, atlas_name: str, fetcher_nilearn: nilearn.datasets, **kwargs
    ):
        """Fetch an atlas using Nilearn.

        Parameters
        ----------
        atlas_name : str
            The name of the atlas.
        fetcher_nilearn : nilearn.datasets
            Nilearn fetcher function.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        dict
            Atlas volume, header, labels, and description.

        Raises
        ------
        Exception
            If fetching fails.

        Examples
        --------
        >>> fetcher = AtlasFetcher()  # doctest: +SKIP
        >>> out = fetcher._fetch_atlas_nilearn(
        ...     'aal', fetcher._atlas_fetchers_nilearn['aal']
        ... )  # doctest: +SKIP
        >>> 'vol' in out  # doctest: +SKIP
        True
        """
        this_kwargs = fetcher_nilearn["default_kwargs"].copy()
        this_kwargs.update(kwargs)
        fetched = fetcher_nilearn["fetcher"](**this_kwargs)
        maphdr = pack_vol_output(fetched["maps"])
        fetched.update(maphdr)
        fetched["vol"] = np.squeeze(fetched["vol"])
        fetched["description"] = self._get_description(atlas_name, fetched, this_kwargs)
        if fetched.get("labels", None) is not None and isinstance(
            fetched["labels"], np.ndarray
        ):
            labels = fetched["labels"].tolist()
        else:
            labels = fetched.get("labels", None)
        return {
            "vol": fetched["vol"],
            "hdr": fetched["hdr"],
            "labels": labels,
            "description": fetched["description"],
        }

    def _fetch_atlas_mne(self, atlas_name: str, fetcher_mne, **kwargs):
        """Fetch an atlas using MNE.

        Parameters
        ----------
        atlas_name : str
            The name of the atlas.
        fetcher_mne : callable or None
            MNE fetcher function.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        dict
            Atlas volume, header, labels, and description.

        Raises
        ------
        Exception
            If fetching fails.

        Examples
        --------
        >>> fetcher = AtlasFetcher()  # doctest: +SKIP
        >>> fetcher._fetch_atlas_mne(
        ...     'aparc', fetcher._atlas_fetchers_mne['aparc']
        ... )['hdr'] is None  # doctest: +SKIP
        True
        """
        if atlas_name == "human-connectum project":
            _ensure_hcp_license()
        kwargs["subject"] = kwargs.get("subject", "fsaverage")
        this_kwargs = fetcher_mne["default_kwargs"].copy()
        this_kwargs.update(kwargs)
        atlas_name_mne = this_kwargs.pop("version", atlas_name)
        this_kwargs.setdefault("subjects_dir", self.subjects_dir)
        fetched = pack_surf_output(
            atlas_name=atlas_name_mne, fetcher=fetcher_mne["fetcher"], **this_kwargs
        )
        this_kwargs.update({"type": "surface"})
        this_kwargs["coordinate system"] = "MNI"
        this_kwargs["version"] = atlas_name_mne
        description = self._get_description(atlas_name, fetcher_mne, this_kwargs)
        fetched["description"] = description
        return fetched

    def _fetch_from_url(self, atlas_name: str, atlas_url: str, **kwargs):
        """Fetch an atlas from a URL.

        Parameters
        ----------
        atlas_name : str
            The name of the atlas.
        atlas_url : str
            URL of the atlas file.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        dict
            Atlas volume, header, labels, and description.

        Raises
        ------
        Exception
            If downloading or loading fails.

        Examples
        --------
        >>> fetcher = AtlasFetcher()  # doctest: +SKIP
        >>> fetcher._fetch_from_url(
        ...     'aal', 'http://example.com/aal.nii.gz'
        ... )  # doctest: +SKIP
        {...}
        """
        local_path = self.file_handler.fetch_from_url(atlas_url, **kwargs)
        output = self.file_handler.fetch_from_local(
            kwargs.get("atlas_file"), local_path, kwargs.get("labels")
        )
        output["description"] = self._get_description(atlas_name, output, kwargs)
        return output

    def list_available_atlases(self):
        """Return a sorted list of available atlas identifiers.

        Returns
        -------
        list of str
            Sorted list of available atlas identifiers.

        Raises
        ------
        None

        Examples
        --------
        >>> fetcher = AtlasFetcher()  # doctest: +SKIP
        >>> isinstance(fetcher.list_available_atlases(), list)  # doctest: +SKIP
        True
        """
        atlases_nilearn = list(self._atlas_fetchers_nilearn.keys())
        atlases_coords = list(self._coords_fetchers_nilearn.keys())
        atlases_mne = list(self._atlas_fetchers_mne.keys())
        atlases_urls = list(self.ATLAS_URLS.keys())
        all_atlases = set(atlases_nilearn + atlases_coords + atlases_mne + atlases_urls)
        return sorted(all_atlases)

    def fetch_atlas(
        self, atlas_name: str, atlas_url: str = None, prefer: str = "nilearn", **kwargs
    ):
        """Fetch an atlas given an identifier.

        The identifier may be a URL, local file path, or a known atlas name.
        The ``prefer`` flag allows choosing the primary source (``"nilearn"`` or
        ``"mne"``).

        Parameters
        ----------
        atlas_name : str
            The name of the atlas or a URL.
        atlas_url : str or None, optional
            Explicit URL of the atlas file.
        prefer : {"nilearn", "mne"}, optional
            Preferred source for fetching the atlas.
        **kwargs
            Additional keyword arguments for fetching.

        Returns
        -------
        dict
            Atlas volume, header, labels, and description.

        Raises
        ------
        ValueError
            If the atlas name is not recognized.
        FileNotFoundError
            If the atlas file or labels file is not found.
        Exception
            If there is an error during fetching.

        Examples
        --------
        >>> fetcher = AtlasFetcher()  # doctest: +SKIP
        >>> out = fetcher.fetch_atlas('aal')  # doctest: +SKIP
        >>> out['labels'][0]  # doctest: +SKIP
        'Precentral_L'
        """
        key = atlas_name.lower()
        if atlas_url is not None and atlas_url.startswith(("http://", "https://")):
            return self._fetch_from_url(key, atlas_url, **kwargs)
        if key in self.ATLAS_URLS:
            kwargs.update(self.ATLAS_URLS[key])
            return self._fetch_from_url(key, **kwargs)

        # Local file or image cases.
        atlas_file = kwargs.get("atlas_file", None)
        if atlas_file is not None:
            if os.path.isfile(atlas_file):
                atlas_dir = os.path.dirname(atlas_file) or "."
                atlas_fname = os.path.basename(atlas_file)
                return self.file_handler.fetch_from_local(
                    atlas_fname, atlas_dir, kwargs.get("labels")
                )
            else:
                local_path = os.path.join(self.data_dir, atlas_file)
                if os.path.isfile(local_path):
                    atlas_dir = os.path.dirname(local_path) or "."
                    atlas_fname = os.path.basename(local_path)
                    return self.file_handler.fetch_from_local(
                        atlas_fname, atlas_dir, kwargs.get("labels")
                    )

        atlas_image = kwargs.get("atlas_image")
        if isinstance(atlas_image, (Nifti1Image, np.ndarray)):
            output = pack_vol_output(atlas_image)
            output["labels"] = kwargs.get("labels")
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

        raise ValueError(
            f"Unrecognized atlas name '{atlas_name}'. Available options:"
            f" {self.list_available_atlases()}"
        )


HCP_LICENSE_ENV = "COORD2REGION_ACCEPT_HCPMMP"
HCP_LICENSE_PATH = Path.home() / ".mne" / "hcpmmp-license.txt"


def _ensure_hcp_license() -> None:
    """Raise if the HCP-MMP license was not accepted."""
    if os.getenv(HCP_LICENSE_ENV) == "1":
        return
    if HCP_LICENSE_PATH.exists() and HCP_LICENSE_PATH.stat().st_size > 0:
        return
    msg = (
        "Using the 'human-connectum project' atlas requires accepting the HCP-MMP "
        "license. Run:\n"
        '    python -c "import mne; mne.datasets.fetch_hcp_mmp_parcellation('
        'accept=True)"\n'
        "or create the file '~/.mne/hcpmmp-license.txt' after reading the terms."
    )
    raise RuntimeError(msg)
