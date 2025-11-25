Coord2Region
============

.. |icon-tutorials| raw:: html

   <i class="fa-solid fa-graduation-cap" aria-hidden="true"></i>

.. |icon-examples| raw:: html

   <i class="fa-solid fa-flask" aria-hidden="true"></i>

.. |icon-pipeline| raw:: html

   <i class="fa-solid fa-cubes" aria-hidden="true"></i>

.. |icon-atlas| raw:: html

   <i class="fa-solid fa-map" aria-hidden="true"></i>

.. |icon-builder| raw:: html

   <i class="fa-solid fa-display" aria-hidden="true"></i>

.. |icon-brain| raw:: html

   <i class="fa-solid fa-brain" aria-hidden="true"></i>

.. |icon-cite| raw:: html

   <i class="fa-solid fa-bullhorn" aria-hidden="true"></i>

.. |icon-contrib| raw:: html

   <i class="fa-solid fa-hands-helping" aria-hidden="true"></i>

.. |icon-contributors| raw:: html

   <i class="fa-solid fa-star" aria-hidden="true"></i>

.. |icon-coc| raw:: html

   <i class="fa-solid fa-scroll" aria-hidden="true"></i>


.. container:: hero-panel

   .. container:: hero-surface

      .. container:: hero-logo

         .. container:: hero-copy__eyebrow

            **Current version** |release|

         .. image:: _static/images/logo.png
            :class: hero-logo__img hero-logo__img--light
            :alt: Coord2Region logo
            :width: 260
            :align: center

      .. container:: hero-copy

         .. container:: hero-copy__text

            *Coord2Region* maps 3D brain coordinates (e.g., MNI or Talairach)
            to anatomical regions across multiple atlases and connects them to
            meta-analytic resources via NiMARE, Neurosynth, and NeuroQuery.

            Optional large language model (LLM) utilities can summarize linked
            studies and generate illustrative views of queried regions. These
            AI-assisted features are designed to support interpretation and
            exploration, while remaining complementary to established
            neuroimaging workflows.

         .. container:: hero-actions

            - |icon-examples| :doc:`Examples <auto_examples/index>`
            - |icon-tutorials| :doc:`Tutorials <documentation/tutorials>`
            - |icon-coc| :doc:`Code of Conduct <documentation/coc>`
            - |icon-cite| :doc:`Cite <documentation/citation>`
            - |icon-contrib| :doc:`Contribute <developer_guide>`
            - |icon-contributors| :doc:`Contributors <documentation/contributors>`


   .. container:: hero-highlights

      .. container:: hero-highlight

         |icon-pipeline| :doc:`Reproducible pipeline <documentation/pipeline>`

            An easy-to-use, configurable pipeline that runs
            atlas lookups, studies retrieval, and AI tools.

      .. container:: hero-highlight

         |icon-atlas| :doc:`Atlases <documentation/atlases>`

            Translate coordinates into anatomical labels across +20
            atlases.

      .. container:: hero-highlight

         |icon-builder| :doc:`Interactive web interface <documentation/web_interface>`

            A Web-based configuration builder and a cloud platform to run Coord2Region pipelines.

.. toctree::
   :hidden:
   :maxdepth: 2

   install
   documentation
   api_reference
   get_help
   developer_guide
