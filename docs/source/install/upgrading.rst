.. _install-upgrading:

Upgrading or uninstalling
=========================

To upgrade Coord2Region inside an existing environment:

.. code-block:: bash

   pip install --upgrade coord2region

If a new release adds configuration fields, re-run the configuration helper; it
updates or extends the YAML file rather than discarding it.

To uninstall:

.. code-block:: bash

   pip uninstall coord2region

This removes the package but leaves atlases, cached datasets, and
``coord2region-output/`` untouched so you can reinstall later without
re-downloading everything.
