.. _dev-python:

Python Core
===========

This guide covers the development workflow for the ``coord2region`` Python package, 
including environment setup, testing, and release engineering.

Environment Setup
-----------------

1. **Clone the repository:**

   .. code-block:: bash

      git clone https://github.com/BabaSanfour/Coord2Region.git
      cd Coord2Region

2. **Install in editable mode:**
   We recommend using a virtual environment. Install the package with its 
   ``dev`` and ``docs`` extras:

   .. code-block:: bash

      # On standard shells
      pip install -e .[dev,docs]

      # On zsh (requires quotes to avoid glob errors)
      pip install -e '.[dev,docs]'

Documentation Build
-------------------

With the development environment in place, build the docs using the themed
Sphinx site:

.. code-block:: bash

   make -C docs html

The official ``pydata-sphinx-theme`` assets are required by default. If you only
need a quick preview and do not want to install the docs extras, temporarily
allow the minimal fallback theme:

.. code-block:: bash

   COORD2REGION_DOCS_ALLOW_FALLBACK=1 make -C docs html

For live-reloading previews, install ``sphinx-autobuild`` (included in the
``docs`` extra) and run:

.. code-block:: bash

   make -C docs livehtml

Quality Assurance
-----------------

This project uses **Ruff** for linting/formatting and **Pre-commit** to enforce 
standards automatically.

* **Install Hook:** ``pre-commit install`` (run once)
* **Run Manually:** ``pre-commit run --all-files``

Testing & Coverage
------------------

We use ``pytest`` for unit testing. The project aims for at least **80% code coverage**.

.. code-block:: bash

   pytest --cov

Configuration Schema
--------------------

The CLI parameters are defined via the :class:`~coord2region.config.Coord2RegionConfig` 
Pydantic model. If you modify ``config.py``, you must regenerate the JSON schema:

.. code-block:: bash

   make schema

Release Workflow
----------------

This project follows **Semantic Versioning** via ``setuptools_scm``. To release a new version:

1. Verify all tests pass.
2. Create an annotated git tag (e.g., ``v0.2.0``).
3. Push the tag to GitHub. CI/CD will handle PyPI publishing.

.. code-block:: bash

   git tag -a v0.2.0 -m "Release v0.2.0"
   git push origin v0.2.0
