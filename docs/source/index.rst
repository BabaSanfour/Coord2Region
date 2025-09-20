Coord2Region
============

Map neuroimaging coordinates to atlas regions, query studies near a
coordinate, and optionally generate AI summaries and illustrative images.
Built on NiMARE, nilearn and MNE.

Requirements
------------

- ``nilearn>=0.11``

Quickstart
----------

Python
~~~~~~

.. code-block:: python

   from coord2region import AtlasMapper
   from coord2region.fetching import AtlasFetcher

   # Fetch a volumetric atlas and map a coordinate to a label
   atlas = AtlasFetcher().fetch_atlas("harvard-oxford")
   mapper = AtlasMapper("harvard-oxford", atlas["vol"], atlas["hdr"], atlas["labels"])
   print(mapper.mni_to_region_name([30, -22, 50]))

Command line
~~~~~~~~~~~~

.. note::
   The following commands are meant to be run in a shell (Terminal, bash, zsh, etc.),
   not inside a Python interpreter.

.. code-block:: bash

   # Map a coordinate to atlas labels
   coord2region coords-to-atlas 30 -22 50 --atlas harvard-oxford

   # Retrieve studies and summaries for the same coordinate
   coord2region coords-to-summary 30 -22 50 --atlas harvard-oxford

   # Produce a report with text and image outputs
   coord2region coords-to-insights 30 -22 50 --atlas harvard-oxford --image-backend nilearn

   # Start from a region name in a specific atlas
   coord2region region-to-insights "Left Amygdala" --atlas harvard-oxford

Explore More
------------

- Tutorials and runnable examples are in the gallery below.
- The user guide covers the end-to-end :mod:`coord2region.pipeline`.
- The API reference lists public classes and functions.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   README

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   pipeline
   atlases

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials

.. toctree::
   :maxdepth: 1
   :caption: Examples

   auto_examples/index

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   autoapi/index

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide

   developer_guide

.. toctree::
   :maxdepth: 1
   :caption: Roadmap

   roadmap
