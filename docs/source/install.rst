Install Coord2Region
====================

Coord2Region ships on PyPI and targets Python 3.10+. Follow the steps below to create an isolated workspace, install the package, and validate that both the CLI and helper scripts can reach your atlases or API credentials.

1. Create and activate a virtual environment.

   .. code-block:: bash

      python -m venv .venv
      source .venv/bin/activate
      python -m pip install --upgrade pip

2. Install Coord2Region from PyPI.

   .. code-block:: bash

      pip install coord2region

3. Configure atlas locations and provider credentials once. The helper script writes ``config/coord2region-config.yaml`` so repeated CLI/Python runs stay in sync.

   .. code-block:: bash

      python -m scripts.configure_coord2region

   You can override any value later with environment variables such as ``OPENAI_API_KEY``, ``GEMINI_API_KEY``, or ``COORD2REGION_ATLAS_DIR``.

4. Verify the installation.

   .. code-block:: bash

      coord2region --help
      coord2region list-atlases

   The ``coord2region`` command creates a ``coord2region-output/`` folder the first time it runs. Each recipe stores YAML/JSON/CSV artefacts there for reproducibility.

Developer install
-----------------

If you plan to contribute, install Coord2Region from a clone of the repository so editable changes rebuild automatically.

.. code-block:: bash

   git clone https://github.com/babasanfour/Coord2Region.git
   cd Coord2Region
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .[dev]

Useful follow ups
-----------------

- Head to the :doc:`documentation_overview` for a map of every guide, tutorial, and reference.
- Skim the :doc:`README` if you want the full feature tour that appears on the project landing page.
- Need support or want to contribute? Jump to :doc:`support_development`.
