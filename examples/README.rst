.. _general_examples:

Examples Gallery
================

Here we show some exemplary use cases for Coord2Region.

.. contents:: Contents
   :local:
   :depth: 2

Module examples
---------------

Demonstrations of individual modules.

.. toctree::
   :maxdepth: 1

   plot_fetching
   plot_atlas_mapping
   plot_coord2study
   plot_ai_interface
   plot_pipeline_basic
   custom_provider_example
   example_0_simple_atlas
   example_1_harvard-oxford
   example_2_batch_harvard
   example_3_multi
   example_4_study
   example_5_multi_atlas_coords_and_studies_querying
   example_6_dataset_cache
   example_7_ai_providers
   example_8_conditional_provider_activation
   example_9_output_formats
   example_10_image_providers
   example_11_local_huggingface
   example_12_nilearn_backend

End-to-end workflows
--------------------

Complete workflows that integrate multiple components. Each example checks for the required data before running.

.. toctree::
   :maxdepth: 1

   plot_fmri_coord_to_region
   plot_meg_source_localization
   plot_ieeg_electrode_localization
   example_pipeline

Data download
-------------

Some examples require datasets to be present locally.

- MEG example (MNE "sample" dataset):

  .. code-block:: python

      import mne
      mne.datasets.sample.data_path()  # downloads to the default MNE data folder

- iEEG example (MNE epilepsy ECoG dataset):

  .. code-block:: python

      import mne
      mne.datasets.epilepsy_ecog.data_path()

You can control the download location via the ``MNE_DATA`` environment variable
or by passing ``path=...`` to ``data_path()``. Examples will look for the
subjects directory (when needed) under ``<data_path>/subjects``.

Coord2Study datasets (NiMARE)
-----------------------------

Examples that query studies (e.g., coordinate → study lookup) rely on
NiMARE-compatible datasets such as Neurosynth or NeuroQuery. Use the helper
below to fetch datasets to a cache directory before running examples:

.. code-block:: python

    from coord2region.coord2study import fetch_datasets
    fetch_datasets(data_dir="~/.coord2region_examples", sources=["neurosynth"])  # or ["neuroquery"]

The first run downloads and converts datasets; subsequent runs reuse the cache.
