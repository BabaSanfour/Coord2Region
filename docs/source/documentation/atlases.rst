Supported Atlases
=================

**Coord2Region** provides unified access to a wide range of neuroimaging reference
spaces. Whether you are working with volumetric NIfTI files, cortical surfaces,
or coordinate-based regions of interest (ROIs), the package handles downloading
and caching automatically.

Quick Start: Listing Atlases
----------------------------

You can check which atlases are available in your installed version using either
the command line or Python:

.. code-block:: bash
   :caption: CLI

   coord2region --list_atlases

.. code-block:: python
   :caption: Python

   from coord2region.fetching import AtlasFetcher
   print(AtlasFetcher().list_available_atlases())

Usage Guide
-----------

- **Anatomical Mapping**: Use **Harvard-Oxford** or **AAL** to label coordinates with standard macroscopic anatomical names.
- **Functional Connectivity**: Use **Schaefer** or **Yeo** networks when defining nodes for graph theory.
- **Surface Analysis**: Use **Aparc** or **HCP-MMP** if your pipeline operates in FreeSurfer vertex space.
- **Custom Data**: Provide a direct URL or a local file path if your atlas is not listed below.

.. _volumetric_atlases:

Volumetric Atlases (MNI Space)
------------------------------

These atlases are voxel-based (NIfTI) and aligned to the MNI152 template.

Anatomical & Cytoarchitectonic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Identifier
     - Description & Defaults
   * - ``harvard-oxford``
     - **Standard Probabilistic Atlas.** Maps cortical and subcortical areas.
       
       *Defaults:* ``atlas_name='cort-maxprob-thr25-2mm'``.
   * - ``aal``
     - **Automated Anatomical Labeling.** Standard macroscopic atlas (AAL3v2).
       
       *Defaults:* ``version='3v2'``.
   * - ``juelich``
     - **Cytoarchitectonic Atlas.** Microscopic cell distribution maps.
       
       *Defaults:* ``atlas_name='maxprob-thr0-1mm'``.
   * - ``brodmann``
     - **Brodmann Areas.** Classic cytoarchitectonic labels (BA1-BA52) via Talairach transform.
   * - ``pauli``
     - **Subcortical Nuclei.** High-resolution probabilistic atlas (2017).
   * - ``talairach``
     - **Talairach Daemon.** Digital version of the Talairach atlas (NIfTI).

Functional & Parcellations
~~~~~~~~~~~~~~~~~~~~~~~~~~

Derived from functional clustering (fMRI) for defining homogeneous regions.

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Identifier
     - Description & Defaults
   * - ``schaefer``
     - **Schaefer 2018.** Gradient-weighted Markov Random Fields.
       
       *Defaults:* ``n_rois=400``, ``yeo_networks=7``, ``resolution_mm=1``.
   * - ``yeo``
     - **Yeo 2011 (Volumetric).** Functional network parcellation.
       
       *Defaults:* ``n_networks=7`` (Liberal mask).
   * - ``basc``
     - **BASC Multiscale.** Bootstrap Analysis of Stable Clusters (2015).
   * - ``destrieux``
     - **Destrieux 2009.** Volumetric version of the ``aparc.a2009s`` parcellation.

Surface Atlases (FreeSurfer/MNE)
--------------------------------

Parcellations defined on the cortical surface (vertices). By default, these map to the ``fsaverage`` subject.

Standard FreeSurfer
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Identifier
     - Description
   * - ``aparc``
     - **Desikan-Killiany.** The default FreeSurfer gyral-based parcellation.
   * - ``aparc.a2005s``
     - **Desikan-Killiany (Legacy).** The older 2005 version of the atlas.
   * - ``aparc.a2009s``
     - **Destrieux.** Finer parcellation including both gyri and sulci.
   * - ``aparc_sub``
     - **MNE Subdivision.** A subdivided version of aparc for finer granularity.

PALS & OASIS
~~~~~~~~~~~~

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Identifier
     - Description
   * - ``yeo2011``
     - **Yeo 2011 (Surface).** Surface-based 17-network parcellation.
   * - ``pals_b12_lobes``
     - PALS-B12 Lobes.
   * - ``pals_b12_orbitofrontal``
     - PALS-B12 Orbitofrontal structures.
   * - ``pals_b12_visuotopic``
     - PALS-B12 Visuotopic areas.
   * - ``oasis.chubs``
     - OASIS CHUBS labels.

Connectome Project (HCP)
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Identifier
     - Description & Requirements
   * - ``human-connectum project``
     - **HCP-MMP 1.0.** Glasser 2016 multi-modal parcellation.
       
       .. important::
          **License Required:** You must accept data usage terms.
          
          Run: ``mne.datasets.fetch_hcp_mmp_parcellation(accept=True)``
          
          Or set env var: ``COORD2REGION_ACCEPT_HCPMMP=1``

Coordinate Sets (ROIs)
----------------------

Spherical ROIs or centroids used in graph theory meta-analyses.

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Identifier
     - Description
   * - ``power``
     - **Power 2011.** 264 ROIs derived from meta-analysis and functional connectivity.
   * - ``dosenbach``
     - **Dosenbach 2010.** 160 ROIs focused on sensorimotor and task control networks.
   * - ``seitzman``
     - **Seitzman 2018.** 300 ROIs including subcortical and cerebellar regions.

Custom Atlases (URLs & Local Files)
-----------------------------------

If your desired atlas is not in the built-in list, you can provide a direct link or a local file path.

**Using a Direct URL**

You can pass a downloadable URL to a NIfTI file directly to the fetcher or CLI.

.. code-block:: python

   # Python
   fetcher.fetch_atlas(atlas_url="https://example.com/my_parcellation.nii.gz")

**Using a Local File**

If you have already downloaded an atlas (or created your own parcellation), point Coord2Region to the file path.

.. code-block:: python

   # Python
   mapper = AtlasMapper(atlas_file="/path/to/my_custom_atlas.nii")

.. code-block:: bash

   # CLI
   coord2region coords-to-atlas 30 -20 50 --atlas-file /path/to/my_custom_atlas.nii

Implementation Details
----------------------

The :meth:`~coord2region.fetching.AtlasFetcher.fetch_atlas` method returns a dictionary containing:

* ``maps``: Path to the NIfTI or annotation file.
* ``labels``: List of region names.
* ``description``: Metadata string describing the atlas version.
* ``type``: One of ``volumetric``, ``surface``, or ``coords``.