.. _documentation-citation:

How to Cite
===========

**Coord2Region** relies on a constellation of open-source tools and atlases. 
Proper citation ensures that credit is distributed not just to this package, 
but to the authors of the atlases, datasets, and AI models that power your results.

If you publish or present analyses built on this toolkit, please cite our 
Zenodo record.

Citing the Software
-------------------

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.15048848.svg
   :target: https://doi.org/10.5281/zenodo.15048848
   :alt: DOI

You can download the official metadata file or copy the BibTeX entry below.

* :download:`Download CITATION.cff <../CITATION.cff>` (Citation File Format)

.. code-block:: bibtex
   :caption: BibTeX Entry

   @software{coord2region,
     author       = {Abdelhedi, Hamza and Mantilla-Ramos, Yorguin-Jose and Esmaeili, Sina and Pascarella, Annalisa and Hadid, Vanessa and Jerbi, Karim},
     title        = {Coord2Region},
     year         = {2025},
     publisher    = {Zenodo},
     doi          = {10.5281/zenodo.15048848},
     url          = {https://zenodo.org/records/15048848}
   }

Citing Data & Atlases
---------------------

Your results depend heavily on the specific atlas used. **You must cite the 
atlas authors** to ensure scientific reproducibility and credit.

The :doc:`atlases` page contains the specific references for every supported 
parcellation.

* **Example Acknowledgement:**
  *"Anatomical labeling was performed using Coord2Region (Abdelhedi et al., 2025) 
  with the Harvard-Oxford Cortical Atlas (Desikan et al., 2006) and the Schaefer 
  2018 Parcellation (Schaefer et al., 2018)."*

Citing AI & Providers
---------------------

When using Generative AI features (summaries or images), it is best practice to 
cite the specific model version and the retrieval service.

* **Literature Search:** If you used `coords-to-study`, cite **Neurosynth** (Yarkoni et al., 2011) or **NeuroQuery** (Dockès et al., 2020) depending on 
  your configured source.
* **AI Models:** Specify the model ID (e.g., `gpt-4o`, `gemini-1.5-pro`) and 
  the date of access, as these models evolve over time.

Talks & Demos
-------------

Preparing a slide deck or live demo? We appreciate you sharing the project links 
so others can explore the workflow:

* **Code:** `https://github.com/BabaSanfour/Coord2Region <https://github.com/BabaSanfour/Coord2Region>`_
* **Web Builder:** `https://babasanfour.github.io/Coord2Region/builder/ <https://babasanfour.github.io/Coord2Region/builder/>`_