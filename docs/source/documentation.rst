Documentation
=============

.. note::

   **New to Coord2Region?** Start with the :doc:`install` guide.
   
   If you are new to Python, check out the `Python tutorial
   <https://docs.python.org/3/tutorial/>`_ or the `Real Python getting started
   track <https://realpython.com/start-here/>`_ first.

**Coord2Region** bridges the gap between 3D brain coordinates and scientific 
insight. It unifies **anatomical mapping**, **meta-analytic literature search**, 
and **generative AI** into a streamlined workflow.

.. admonition:: Looking for technical details?

   This page focuses on usage concepts and tutorials.
   
   * **API Reference**: Visit the **API Reference** menu for precise class and function specifications.
   * **Development**: See the **Development** menu for contribution guidelines.
   * **Support**: Check **Get Help** if you are running into issues.

Documentation pillars
---------------------

The documentation is organized into three sections to guide your usage:

- **User Guide** – Step-by-step lessons on how to use Coord2Region.
  This includes the :doc:`documentation/tutorials`, :doc:`auto_examples/index`, and detailed
  breakdowns of supported :doc:`documentation/atlases` and :doc:`documentation/providers`.

- **Operational Interfaces** – Instructions for the specific tools you use to interact
  with the package. Covers the :doc:`documentation/cli_tools` and 
  :doc:`documentation/web_interface`, plus the core :doc:`documentation/workflow` logic.

- **Project Metadata** – Contextual information about the software itself.
  Read about our :doc:`documentation/design_philosophy`, how to provide :doc:`documentation/citation` 
  in your papers, and the people involved in :doc:`documentation/contributors`.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: User Guide

   documentation/workflow
   documentation/tutorials
   auto_examples/index
   documentation/atlases
   documentation/providers

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Operational Interfaces

   documentation/cli_tools
   documentation/web_interface
   documentation/pipeline
   documentation/glossary

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Project Metadata

   documentation/design_philosophy
   documentation/citation
   documentation/contributors
   documentation/coc