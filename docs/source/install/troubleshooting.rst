.. _install-troubleshooting:

Troubleshooting
===============

Install fails with dependency conflicts
---------------------------------------

If ``pip install coord2region`` reports version conflicts with other packages:

- Make sure you are using a **fresh** virtualenv or conda environment.
- Upgrade ``pip`` before install:

  .. code-block:: bash

     python -m pip install --upgrade pip

- If you are in a conda environment, avoid mixing ``conda install`` and
  ``pip install`` for overlapping packages.

Atlases or datasets keep re-downloading
---------------------------------------

Check that the atlas directory and working directory in
``coord2region-config.yaml`` point to stable locations that your user can write
to. If you override paths with environment variables such as
``COORD2REGION_ATLAS_DIR``, make sure you export them consistently in every
session.

AI calls fail even though keys are set
--------------------------------------

- Confirm that the relevant environment variables (e.g., ``OPENAI_API_KEY``)
  are set in the same shell where you run Coord2Region.
- If you edited ``coord2region-config.yaml`` by hand, check for indentation
  errors or mis-spelled provider names.
- Some providers require model names as well as keys; see the AI-related
  examples in the :doc:`Examples gallery <auto_examples/index>` for working
  configurations.
