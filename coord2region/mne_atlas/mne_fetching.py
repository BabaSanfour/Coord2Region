# --------------------------------------------------------
# mne_fetching.py
# --------------------------------------------------------
"""
Utilities to fetch or verify MNE/FreeSurfer-based atlases in subjects_dir.
"""

import os
import warnings
import mne

class MNEAtlasFetcher:
    """
    Class that handles fetching MNE-based (FreeSurfer) atlases into a subjects_dir.
    Typically used with 'fsaverage' as the subject for standard template atlases.

    Example usage:
        fetcher = MNEAtlasFetcher(subjects_dir='/path/to/subjects_dir')
        info = fetcher.fetch_atlas('HCPMMP1', accept_hcp=True)
        # => now HCPMMP1 is downloaded into /path/to/subjects_dir/fsaverage/label
    """
    def __init__(self, subjects_dir: str = None):
        """
        :param subjects_dir: Directory with FreeSurfer subjects (where fsaverage is or will be downloaded).
        """
        if subjects_dir is None:
            # fallback to MNE config
            self.subjects_dir = mne.get_config('SUBJECTS_DIR', None)
            if self.subjects_dir is None:
                raise ValueError("Please provide a subjects_dir or set MNE's SUBJECTS_DIR in your environment.")
        else:
            self.subjects_dir = os.path.abspath(subjects_dir)

        # ensure directory exists
        os.makedirs(self.subjects_dir, exist_ok=True)

    def fetch_atlas(self, atlas_name: str, subject: str = 'fsaverage', **kwargs):
        """
        Ensures the specified atlas is available for the subject in subjects_dir.

        The returned dictionary includes:
          {
            'atlas_name': ...,   # e.g. 'HCPMMP1'
            'subject': ...,      # 'fsaverage'
            'subjects_dir': ..., # path
          }

        :param atlas_name: Name of the atlas or key (e.g. 'aparc', 'aparc_sub', 'HCPMMP1', etc.).
        :param subject: Typically 'fsaverage' (the template).
        :param kwargs: Additional fetch parameters. For example, accept_hcp=True for HCP atlas license.
        :return: dict with keys 'atlas_name', 'subject', 'subjects_dir'.
        """
        atlas_name_lower = atlas_name.lower()

        # Some standard MNE fetchers:
        # 1) HCPMMP
        if atlas_name_lower in ['hcp', 'hcpmmp', 'hcpmmp1']:
            # HCP requires license acceptance
            accept = kwargs.get('accept_hcp', False)
            if not accept:
                raise ValueError("To fetch HCPMMP atlas, you must pass accept_hcp=True.")
            mne.datasets.fetch_hcp_mmp_parcellation(
                subjects_dir=self.subjects_dir,
                accept=True,
                verbose=True
            )
            # standard name is 'HCPMMP1'
            atlas_name = 'HCPMMP1'

        # 2) Subdivided aparc (448 regions)
        elif atlas_name_lower in ['aparc_sub', 'aparc_subs']:
            mne.datasets.fetch_aparc_sub_parcellation(
                subjects_dir=self.subjects_dir,
                verbose=True
            )
            atlas_name = 'aparc_sub'

        # 3) Desikan (aparc)
        elif atlas_name_lower in ['aparc', 'desikan', 'dk']:
            # Usually included with fsaverage, but let's ensure fsaverage is there
            mne.datasets.fetch_fsaverage(verbose=True)
            atlas_name = 'aparc'

        # 4) Destrieux (aparc.a2009s)
        elif atlas_name_lower in ['aparc.a2009s', 'destrieux', 'a2009s']:
            mne.datasets.fetch_fsaverage(verbose=True)
            atlas_name = 'aparc.a2009s'

        # 5) DKT (aparc.DKTatlas)
        elif atlas_name_lower in ['dkt', 'aparc.dktatlas']:
            mne.datasets.fetch_fsaverage(verbose=True)
            atlas_name = 'aparc.DKTatlas'

        else:
            # Possibly a custom .annot or Yeo, Schaefer, etc.
            # We won't do an auto-fetch. We assume user has manually placed them
            warnings.warn(f"Atlas '{atlas_name}' not recognized for auto-fetch. "
                          "Assuming it's already in subjects_dir. No fetch performed.")
            # keep atlas_name as user gave

        return {
            'atlas_name': atlas_name,
            'subject': subject,
            'subjects_dir': self.subjects_dir,
        }
