.. _install-config-layout:

Config file layout and working directories
==========================================

After running the configuration helper, you should have a YAML file similar to:

.. code-block:: text

   config/coord2region-config.yaml

Typical fields include:

- A path for atlas downloads and caching
- A default working directory for outputs
- A list of enabled AI providers and their API keys (if any)
- Options for retries, timeouts, and provider fallbacks

At runtime, Coord2Region also creates an output directory (by default):

.. code-block:: text

   coord2region-output/

Each pipeline run (CLI or Python) can emit YAML/JSON/CSV artefacts and any
generated images into subfolders of this directory, making it easy to re-run or
share specific configurations.

.. warning::

   The configuration file may contain private API keys. Keep
   ``coord2region-config.yaml`` out of version control and do not share it
   publicly.
