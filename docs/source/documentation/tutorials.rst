.. _documentation-tutorials:

Tutorials
=========

.. note::

    **Hands-on learning:** These tutorials are generated from Python scripts.
    You can download the source code or Jupyter notebooks at the bottom of 
    each tutorial page.

The following sections cover end-to-end workflows using **Coord2Region**.


Functional Imaging (fMRI)
-------------------------

.. raw:: html

    <div class="sphx-glr-thumbnails">

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Map peak activations from an fMRI study to atlas regions and export artefacts.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_fmri_coord_to_region_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_fmri_coord_to_region.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">fMRI Peaks to Regions</div>
    </div>

.. raw:: html

    </div><div style="clear:both"></div>

Electrophysiology (MEG & iEEG)
------------------------------

.. raw:: html

    <div class="sphx-glr-thumbnails">

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Inspect how MEG sources translate into anatomical labels.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_meg_source_localization_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_meg_source_localization.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">MEG Source Localization</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Align invasive recordings with atlas metadata and prepare reports.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_ieeg_electrode_localization_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_ieeg_electrode_localization.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">iEEG Electrode Localization</div>
    </div>

.. raw:: html

    </div><div style="clear:both"></div>


.. topic:: Data management note

    To keep the documentation build lightweight, these tutorials are configured 
    to skip large dataset downloads if the files are not present locally. 
    
    When running these scripts on your own machine, Coord2Region will 
    automatically fetch the necessary templates (e.g., MNI152) and atlases via 
    ``nilearn`` or ``mne`` if they are missing.

.. toctree::
   :hidden:

   ../auto_examples/plot_fmri_coord_to_region
   ../auto_examples/plot_meg_source_localization
   ../auto_examples/plot_ieeg_electrode_localization