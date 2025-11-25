.. _documentation-web-interface:

Web Interface
=============

**Coord2Region** offers browser-based tools to help you design pipelines and 
test workflows without installing Python locally.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - 🛠️ Config Builder
     - 🚀 Web Runner
   * - `Launch Builder <https://babasanfour.github.io/Coord2Region/builder/>`_
     - `Launch Runner <https://huggingface.co/spaces/BabaSanfour/Coord2Region>`_
   * - **Hosted on GitHub Pages.**
       A visual editor for generating YAML configuration files. Best for designing reproducible recipes.
     - **Hosted on Hugging Face Spaces.**
       A live demonstration environment to test the pipeline on single coordinates.

Workflow: Design & Export
-------------------------

The primary goal of the web interface is to bridge the gap between visual 
exploration and reproducible command-line execution.

1. **Design Visually**: Open the `Builder <https://babasanfour.github.io/Coord2Region/builder/>`_ 
   and toggle the features you need (e.g., specific atlases, AI summaries, or 
   Neurosynth lookups).
2. **Preview Logic**: Watch the **Live Preview** panel. It updates the YAML 
   configuration and CLI command in real-time as you adjust settings.
3. **Export**: Click **Download YAML** to save your pipeline.
4. **Run Anywhere**: Use that YAML file with the Python package for high-performance execution:
   
   .. code-block:: bash

      coord2region run --config my_pipeline.yaml

Features
--------

**Configuration Builder**

- **Interactive Forms**: Supply coordinates or region names, select from available 
  atlases, and configure AI provider keys securely.
- **Privacy First**: The builder is a static site. Your API keys and data remain 
  in your browser until you choose to export them.
- **Import/Export**: Drag and drop an existing `coord2region.yaml` file to 
  edit it visually, keeping your experimental code and visual designs in sync.

**Web Runner (Hugging Face)**

- **Zero Installation**: Try out the pipeline immediately in the cloud.
- **Visual Results**: View generated AI images and summaries directly in the UI.

.. warning:: **Throughput Limitations**

   The Web Runner is hosted on a shared Hugging Face Space. It is designed for 
   **demonstration and single-coordinate lookups**.
   
   It cannot support high-throughput demands or large batch processing. For 
   analyzing full datasets (e.g., 100+ coordinates), please install the package 
   locally and use the :doc:`cli_tools`. High-throughput web support is planned 
   for future releases.

Running Locally
---------------

The Config Builder is a static React application that you can run offline if you 
prefer to keep your configuration environment air-gapped.

1. **Navigate** to the ``web-interface/`` directory in the repository.
2. **Install** dependencies (React + Vite ecosystem):
   
   .. code-block:: bash

      npm install

3. **Start** the development server:

   .. code-block:: bash

      npm run dev

4. **Test** using the included Playwright end-to-end suite:

   .. code-block:: bash

      npm run test:e2e