.. _general_examples:

Examples Gallery
================

When you have a focused question—"How do I call this provider?" or "How can I 
dump outputs as CSV?"—browse the gallery below. Each card is a runnable 
recipe showing the exact CLI flags or Python calls needed for a specific task.

.. _examples_mapping:

Anatomical Mapping
------------------

Basic recipes for querying atlases. These examples demonstrate how to retrieve 
labels from standard atlases (Harvard-Oxford, AAL) and FreeSurfer parcellations.

**Standard Atlases**

.. minigallery:: ../../examples/example_0_simple_atlas.py ../../examples/example_1_harvard-oxford.py ../../examples/example_2_batch_harvard.py ../../examples/example_3_multi.py ../../examples/plot_atlas_mapping.py
   :add-heading:

**FreeSurfer & Surface**

.. minigallery:: ../../examples/example_13_aparc.py ../../examples/example_14_batch_aparc.py ../../examples/example_15_multi_aparc.py
   :add-heading:

.. _examples_literature_ai:

Literature & AI Integration
---------------------------

Examples that query external providers. These scripts mix local atlas data with 
Neurosynth/NeuroQuery lookups and Generative AI summaries.

.. minigallery:: ../../examples/plot_coord2study.py ../../examples/plot_ai_interface.py ../../examples/example_4_study.py ../../examples/example_5_multi_atlas_coords_and_studies_querying.py ../../examples/example_7_ai_providers.py ../../examples/example_10_image_providers.py ../../examples/custom_provider_example.py
   :add-heading:

.. _examples_advanced:

Advanced Configuration
----------------------

Deep dives into caching, conditional execution, backend switching, and 
formatting outputs.

.. minigallery:: ../../examples/example_6_dataset_cache.py ../../examples/example_8_conditional_provider_activation.py ../../examples/example_9_output_formats.py ../../examples/example_11_local_huggingface.py ../../examples/example_12_nilearn_backend.py ../../examples/plot_fetching.py
   :add-heading:

.. _examples_end_to_end:

End-to-End Workflows
--------------------

Complete pipelines integrating multiple components.

.. minigallery:: ../../examples/plot_fmri_coord_to_region.py ../../examples/plot_meg_source_localization.py ../../examples/plot_ieeg_electrode_localization.py ../../examples/example_pipeline.py ../../examples/plot_pipeline_basic.py
   :add-heading:

Data Management
---------------

.. topic:: Download Requirements

    Some examples require datasets to be present locally. Coord2Region tries to 
    fetch these automatically, but you can also manage them manually.

    **Electrophysiology Datasets (MNE)**
    
    Examples using MNE (MEG/iEEG) rely on the ``sample`` and ``epilepsy_ecog`` datasets.

    .. code-block:: python

        import mne
        # Downloads to ~/mne_data by default
        mne.datasets.sample.data_path()
        mne.datasets.epilepsy_ecog.data_path()

    **Literature Datasets (NiMARE/Neurosynth)**
    
    Coordinate-to-study lookups require cached database files. Use the helper 
    function to pre-fetch them:

    .. code-block:: python

        from coord2region.coord2study import fetch_datasets
        
        # Downloads to ~/.coord2region_examples
        fetch_datasets(data_dir="~/.coord2region_examples", sources=["neurosynth"])
