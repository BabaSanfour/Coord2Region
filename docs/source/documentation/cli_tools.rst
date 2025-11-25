.. _documentation-cli-tools:

CLI Tools
=========

The :command:`coord2region` CLI is the quickest way to run repeatable recipes.
Use this page to pick the right command and learn where outputs land.



Quick Reference
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Command Group
     - Example
     - Purpose
   * - **Coordinate Mapping**
     - ``coords-to-atlas``
     - Convert MNI (x,y,z) to atlas labels.
   * - **Literature Search**
     - ``coords-to-study``
     - Find studies near a coordinate (Neurosynth).
   * - **AI Integration**
     - ``coords-to-summary`` / ``coords-to-image``
     - Generate text summaries or brain images.
   * - **Reverse Lookup**
     - ``region-to-coords``
     - Get centroids for named regions (e.g., "Left Amygdala").
   * - **Batch Processing**
     - ``run --config pipeline.yaml``
     - Execute complex workflows defined in a file.

Coordinate Commands
-------------------
These commands accept MNI coordinates as arguments (``x y z`` or ``x,y,z``) or via ``--coords-file``.

.. option:: coords-to-atlas

   Maps coordinates to anatomical regions.

   .. code-block:: bash

      coord2region coords-to-atlas 30 -22 50 --atlas harvard-oxford --atlas aal

.. option:: coords-to-study

   Retrieves related studies from Neurosynth/NeuroQuery.

   .. code-block:: bash

      coord2region coords-to-study 30 -22 50 --sources neurosynth

.. option:: coords-to-summary

   Generates a semantic summary of the location using an LLM.

   .. code-block:: bash

      coord2region coords-to-summary 30 -22 50 --gemini-api-key ...

.. option:: coords-to-image

   Creates a brain image (statistical map or AI-generated).

   .. code-block:: bash

      coord2region coords-to-image 30 -22 50 --image-backend nilearn

.. option:: coords-to-insights

   **The "All-in-One" command.** Runs the full pipeline: labeling + studies + summary + image.

   .. code-block:: bash

      coord2region coords-to-insights 30 -22 50 --atlas harvard-oxford --sources neurosynth

Region Commands
---------------
These commands take **region names** as input and resolve them to coordinates using a specific atlas.
*Note: These commands require exactly one atlas to be specified.*

.. option:: region-to-coords

   Finds the centroid coordinates for a given region name.

   .. code-block:: bash

      coord2region region-to-coords "Left Amygdala" --atlas harvard-oxford

.. option:: region-to-study

   Finds literature related to a specific anatomical region.

   .. code-block:: bash

      coord2region region-to-study "Precentral Gyrus" --atlas aal --sources neurosynth

.. option:: region-to-summary / region-to-image

   Generates AI summaries or images starting from a region name.

   .. code-block:: bash

      coord2region region-to-summary "Hippocampus" --atlas harvard-oxford

.. option:: region-to-insights

   **The "All-in-One" command for regions.** Runs the full pipeline starting from a name.

   .. code-block:: bash

      coord2region region-to-insights "Thalamus" --atlas harvard-oxford

Configuration & Batch Execution
-------------------------------

For reproducible research, define your pipeline in a YAML file instead of long CLI arguments.

.. option:: run

   Executes a pipeline defined in a configuration file.

   .. code-block:: bash

      coord2region run --config my_pipeline.yaml

   **Dry Run:** To see what commands would be executed without running them:
   
   .. code-block:: bash

      coord2region run --config my_pipeline.yaml --dry-run

Common Options
--------------

These flags apply to most commands.

* **--output-format**: ``json``, ``csv``, ``pickle``, ``pdf`` (default: prints to stdout).
* **--output-name**: specific filename for the output.
* **--working-directory**: Folder to store cache and results.
* **--batch-size**: Number of coordinates/regions to process at once.
* **--list-atlases**: Prints all available atlas identifiers.

Environment Variables
---------------------
To avoid passing API keys every time, set them in your environment:

* ``OPENAI_API_KEY``
* ``GEMINI_API_KEY``
* ``ANTHROPIC_API_KEY``
* ``HUGGINGFACE_API_KEY``

See :doc:`providers` for detailed setup instructions.