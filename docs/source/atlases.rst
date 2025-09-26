Atlases
=======

This page lists atlas families supported by :class:`coord2region.fetching.AtlasFetcher`
and provides brief notes and defaults. Use the examples below to list available
atlases and download one for use with :class:`coord2region.coord2region.AtlasMapper`.

Listing and downloading
-----------------------

.. code-block:: python

   from coord2region.fetching import AtlasFetcher

   fetcher = AtlasFetcher()
   print(fetcher.list_available_atlases())  # all identifiers

   # Download one atlas and inspect description metadata
   atlas = fetcher.fetch_atlas("harvard-oxford")
   print(atlas["description"])  # type, version, coordinate system, etc.


Volumetric (Nilearn)
--------------------

- ``aal``: Automated Anatomical Labeling (AAL3v2). Default: ``version='3v2'``. MNI space.
- ``brodmann``: Talairach Brodmann areas via nilearn (level ``ba``). MNI-mapped Talairach template.
- ``harvard-oxford``: Cortical/parcellation maxprob, default ``cort-maxprob-thr25-2mm``. MNI space.
- ``juelich``: Cytoarchitectonic atlas, default ``maxprob-thr0-1mm``. MNI space.
- ``schaefer``: 2018 Schaefer parcels. Defaults: ``n_rois=400, yeo_networks=7, resolution_mm=1``.
- ``yeo``: 2011 Yeo networks (volumetric). Defaults: ``n_networks=7, thickness='thick'``.
- ``destrieux``: Destrieux 2009 (aparc.a2009s-like) volumetric parcellation. ``lateralized=True``.
- ``pauli``: Subcortical atlas (2017). Default: ``atlas_type='deterministic'``.
- ``basc``: BASC multiscale (2015) clustering parcellations.


Coordinate sets (centroids/ROIs)
---------------------------------

These expose labeled ROI coordinates (not full volumes). Useful for
region‑of‑interest work.

- ``dosenbach``: Dosenbach 2010 coordinates.
- ``power``: Power 2011 264-node coordinates.
- ``seitzman``: Seitzman 2018 coordinates.


Surface (MNE/FreeSurfer)
------------------------

These are cortical surface parcellations (per-vertex labels). By default,
they use the ``fsaverage`` subject and can be adapted via ``subject`` and
``subjects_dir``.

- ``aparc``: FreeSurfer Desikan-Killiany parcellation.
- ``aparc.a2009s``: FreeSurfer Destrieux parcellation.
- ``aparc.a2005s``: Legacy FreeSurfer parcellation.
- ``aparc_sub``: MNE aparc subdivision parcellation.
- ``yeo2011``: 17-network surface parcellation (``Yeo2011_17Networks_N1000``).
- ``human-connectum project``: HCP MMP 1.0 (``HCPMMP1_combined``) parcellation.

  .. note::

     Accessing the HCP MMP atlas requires accepting the Human Connectome Project
     data use terms. Before using this atlas run::

         python -c "import mne; mne.datasets.fetch_hcp_mmp_parcellation(accept=True)"

     or create a non-empty ``~/.mne/hcpmmp-license.txt`` file and set the
     ``COORD2REGION_ACCEPT_HCPMMP=1`` environment variable.
- ``pals_b12_lobes`` / ``pals_b12_orbitofrontal`` / ``pals_b12_visuotopic``: PALS-B12 label sets.
- ``oasis.chubs``: OASIS CHUBS labels.


Direct URLs
-----------

These can also be fetched via direct download URLs (or overridden via
``atlas_url``). Useful if an atlas is not available through nilearn or MNE.

- ``talairach``: Talairach NIfTI (``talairach.nii``).
- ``aal``: AAL3v2 as a direct download (alternative to nilearn's copy).


Notes
-----

- The ``description`` field returned by :meth:`AtlasFetcher.fetch_atlas` includes
  the type (``volumetric``, ``surface`` or ``coords``), version/variant, and
  the coordinate system.
- For surface atlases, mapping between vertex IDs and MNI coordinates requires
  proper ``subject`` and ``subjects_dir`` (e.g., ``fsaverage`` or an MNE dataset
  like ``sample``).
- For volumetric atlases, labels are provided in MNI space and can be used with
  :class:`coord2region.coord2region.AtlasMapper` directly.
