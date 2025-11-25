.. _install-venv-pip:

Virtual environment + pip
=========================

This is the simplest way to get Coord2Region running from a terminal. It works
the same on Linux, macOS, and Windows—only the activation command changes.

1. Create and activate a virtual environment
--------------------------------------------

From the directory where you want to keep your environment:

* Linux / macOS

  .. code-block:: bash

     python -m venv .venv
     source .venv/bin/activate
     python -m pip install --upgrade pip

* Windows (PowerShell)

  .. code-block:: powershell

     py -m venv .venv
     .\.venv\Scripts\Activate.ps1
     python -m pip install --upgrade pip

* Windows (cmd.exe)

  .. code-block:: bat

     py -m venv .venv
     .venv\Scripts\activate.bat
     python -m pip install --upgrade pip

After activation, the prompt should show ``(.venv)`` or a similar name.

2. Install Coord2Region from PyPI
---------------------------------

With the environment active:

.. code-block:: bash

   pip install coord2region

This installs the :mod:`coord2region` Python package, the
:command:`coord2region` command-line entry point, and dependencies such as
NiMARE, Nilearn, and MNE.

3. Run the guided configuration
-------------------------------

Run the helper script once to create a private configuration file. This records
where to store atlases, which meta-analytic datasets to enable, and any API
keys for AI providers (OpenAI, Gemini, Hugging Face, etc.).

.. code-block:: bash

   python -m scripts.configure_coord2region

The script writes a file such as ``config/coord2region-config.yaml`` with the
values you chose. You can edit this YAML file later if you want to fine-tune
atlas directories, provider timeouts, or default outputs.

.. note::

   All AI provider fields are **optional**. Coord2Region works for atlas lookup
   and study retrieval without any API keys. AI features are enabled only when
   the corresponding keys are present.

4. Override settings with environment variables (optional)
----------------------------------------------------------

Any config field can be overridden by an environment variable. Common examples:

- ``OPENAI_API_KEY`` – API key for OpenAI models
- ``GEMINI_API_KEY`` – API key for Google Gemini
- ``HUGGINGFACE_API_KEY`` – token for Hugging Face Inference
- ``COORD2REGION_ATLAS_DIR`` – base directory for storing atlas files

Set these in the shell before running Coord2Region, for example:

.. code-block:: bash

   export OPENAI_API_KEY="sk-..."
   export COORD2REGION_ATLAS_DIR="$HOME/coord2region_atlases"
