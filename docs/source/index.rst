Coord2Region
============

.. list-table::
   :widths: 32 68
   :class: landing-hero

   * - **Region (Start here)**

         - :doc:`Install Coord2Region <install>`
         - :doc:`Documentation overview <documentation_overview>`
         - :doc:`API reference <autoapi/index>`
         - :doc:`Get help & development <support_development>`

       **Learn fast**

         - :doc:`Tutorials <tutorials>`
         - :doc:`Examples gallery <auto_examples/index>`
         - :doc:`Pipeline tour <pipeline>`
         - :doc:`Atlas guide <atlases>`
         - :doc:`Providers & integrations <providers>`

       **Community**

         - :ref:`How to cite <cite-coord2region>`
         - :ref:`Contribute <contribute>`
         - :ref:`Contributors <contributors>`
         - :doc:`Code of Conduct <CODE_OF_CONDUCT>`

     - .. figure:: ../static/images/logo.png
           :width: 240
           :alt: Coord2Region logo
           :align: center

       *Coord2Region* turns MNI coordinates into atlas-backed context, surfaces linked literature, and optionally adds AI-ready summaries. One workflow powers the CLI, Python API, and hosted builder/runner so you can stay reproducible across laptops and browsers.

       No manual atlas juggling: mix NiMARE, Nilearn, and MNE primitives; fetch atlases on demand; map points to names; crosswalk regions to studies; and archive structured YAML/JSON/CSV artefacts for every request.

       Head over to the :doc:`README` for the full release story, or keep scrolling for direct entry points.

What you will find
------------------

- Delivery-first install guide with environment, config, and validation steps (:doc:`install`).
- Documentation overview that mirrors the Coord2Region workflow and links every major guide (:doc:`documentation_overview`).
- A living API reference generated from the codebase (:doc:`autoapi/index`).
- Tutorials, gallery examples, and providers notes that show how to adapt Coord2Region to your atlases, studies, or hosted environments.

Coord2Region workflow
---------------------

.. figure:: ../static/images/workflow.jpg
   :alt: Coord2Region workflow overview
   :align: center
   :width: 90%

   High-level workflow from inputs to outputs.

Web interface previews
----------------------

.. |ui1| image:: ../static/images/web-interface-ui-builder1.png
   :alt: Config Builder – inputs and atlas
   :width: 31%

.. |ui2| image:: ../static/images/web-interface-ui-builder2.png
   :alt: Config Builder – outputs and providers
   :width: 31%

.. |ui3| image:: ../static/images/web-interface-ui-runner.png
   :alt: Runner preview
   :width: 31%

|ui1| |ui2| |ui3|

Paths to explore
----------------

- :doc:`tutorials` – start with runnable notebooks for atlas lookups, literature retrieval, and AI-ready summaries.
- :doc:`auto_examples/index` – browse short recipes you can adapt into pipelines or presentations.
- :doc:`pipeline` – understand how the CLI, Python API, and hosted runner cooperate.
- :doc:`atlases` – install, manage, and query supported atlases.
- :doc:`providers` – configure AI models, data providers, and optional integrations.
- :doc:`support_development` – discover how to cite Coord2Region, get help, or become a contributor.

.. toctree::
   :maxdepth: 1
   :caption: Install

   install

.. toctree::
   :maxdepth: 1
   :caption: Documentation

   documentation_overview
   tutorials
   pipeline
   atlases
   providers
   README

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   autoapi/index

.. toctree::
   :maxdepth: 1
   :caption: Get Help & Development

   support_development
   developer_guide
   roadmap
