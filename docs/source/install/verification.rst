.. _install-verifying:

Verifying your installation
===========================

Once installed and configured, you can check both the **CLI** and the **Python
API**.

Command-line checks
-------------------

From an activated environment:

.. code-block:: bash

   coord2region --version
   coord2region --help

You should see the installed version and a summary of available commands.
If you already configured atlases, you can also list them:

.. code-block:: bash

   coord2region list-atlases

When you run Coord2Region for the first time, a ``coord2region-output/``
directory is created in the working directory.

Python API check
----------------

Start a Python session in the same environment:

.. code-block:: python

   >>> from coord2region.fetching import AtlasFetcher
   >>> af = AtlasFetcher()
   >>> atlas = af.fetch_atlas("harvard-oxford")
   >>> list(atlas.keys())
   ['vol', 'hdr', 'labels', ...]

If this runs without errors and downloads the requested atlas, the core
fetching and I/O stack is working. For a more complete tour of the API, see the
:doc:`Examples gallery <../auto_examples/index>` and :doc:`Tutorials <../documentation/tutorials>`.
