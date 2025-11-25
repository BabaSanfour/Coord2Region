.. _install-conda:

Conda / mamba environment
=========================

If you use Anaconda or Miniconda, you can install Coord2Region into its own
conda environment and still use ``pip`` inside that environment (similar to the
MNE-Python install instructions).

1. Create a fresh environment
-----------------------------

.. code-block:: bash

   conda create -n coord2region python=3.11
   conda activate coord2region

If you prefer, replace ``3.11`` with any supported Python >= 3.10.

2. Install via pip inside the environment
-----------------------------------------

.. code-block:: bash

   pip install --upgrade pip
   pip install coord2region

3. Configure and verify
-----------------------

Run the configuration helper as above:

.. code-block:: bash

   python -m scripts.configure_coord2region

Then run a quick check:

.. code-block:: bash

   coord2region --help
