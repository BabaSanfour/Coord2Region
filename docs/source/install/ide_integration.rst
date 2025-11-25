.. _install-ide-integration:

IDE integration
===============

Once Coord2Region is installed, you can wire your preferred IDE to the same
environment so that linting, notebooks, and tests reuse the packages you just
installed.

Visual Studio Code
------------------

1. Open the Coord2Region repository folder in VS Code.
2. Select the interpreter that points to your virtualenv or conda env
   (``Ctrl+Shift+P`` → **Python: Select Interpreter** → choose ``.venv`` or
   ``coord2region``).
3. (Optional) Create a ``.vscode/settings.json`` with the following snippet so
   terminals pick up the environment for you:

   .. code-block:: json

      {
        "python.defaultInterpreterPath": ".venv/bin/python",
        "terminal.integrated.env.osx": {
          "PYTHONPATH": "${workspaceFolder}"
        }
      }

4. Use the built-in debugger to run ``coord2region`` CLI commands by adding a
   ``launch.json`` entry such as:

   .. code-block:: json

      {
        "name": "Coord2Region CLI",
        "type": "python",
        "request": "launch",
        "module": "coord2region.cli",
        "args": ["--help"],
        "cwd": "${workspaceFolder}"
      }

PyCharm / IntelliJ
------------------

1. Go to **Settings → Project → Python Interpreter** and add the interpreter
   from your virtualenv or conda environment.
2. Mark ``coord2region`` and ``tests`` as source roots so import resolution
   matches ``python -m`` execution.
3. Create a **Run/Debug Configuration** of type *Module name* with
   ``coord2region.cli`` or run the scripts in ``scripts/`` directly for guided
   configuration.
4. Enable **Emulate terminal in output console** so environment variables (such
   as ``OPENAI_API_KEY``) propagate to the run configuration.

Notebook-friendly workflow
--------------------------

If you author tutorials or examples in Jupyter or VS Code notebooks, install
``ipykernel`` inside the same environment and register it once:

.. code-block:: bash

   python -m pip install ipykernel
   python -m ipykernel install --user --name coord2region --display-name "Coord2Region"

The interpreter will now appear in IDE kernel selectors, making it easy to test
API calls and iterate on documentation examples without leaving the IDE.
