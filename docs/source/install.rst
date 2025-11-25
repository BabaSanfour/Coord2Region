Install
=======

Coord2Region ships on PyPI and targets Python 3.10 or newer. This landing page
helps you choose the best installation path, configure atlases and AI
providers, and verify that both the CLI and Python API work in your preferred
environment.

If you are new to Python, start with **Virtual environment + pip**. If you
already use Anaconda/conda, see **Conda / mamba environment**. Each topic below
dives into a single goal so you can jump straight to the step you need.


System requirements
~~~~~~~~~~~~~~~~~~~

Coord2Region is a pure Python package built on the scientific Python stack and
NiMARE/Nilearn/MNE. The installer pulls in required dependencies automatically,
but you should confirm the following prerequisites before installing:

- **Python**: 3.10 or newer (64-bit recommended).
- **Operating systems**: Linux, macOS, and Windows are supported.
- **Disk space**: set aside a few GB if you plan to cache multiple atlases,
  NiMARE datasets, and example data.
- **Optional AI providers**: API keys for services such as OpenAI, Gemini,
  Anthropic, or Hugging Face are required only if you intend to use the
  AI-powered summaries or image generation tools.

.. note::

   Coord2Region uses the same scientific stack as many neuroimaging tools.
   Installing into a fresh environment avoids version conflicts with existing
   projects and keeps dependencies reproducible.

.. _install-next-steps:

Next steps
~~~~~~~~~~

Once Coord2Region is installed and verified:

- See the :doc:`documentation` for a tour of core concepts and
  features.
- Explore :doc:`Tutorials <documentation/tutorials>` for end-to-end walkthroughs.
- Browse the :doc:`Examples gallery <auto_examples/index>` for concrete code
  snippets and multi-atlas workflows.
- Visit :doc:`Support & development <get_help>` if you want to report issues or
  contribute new features.

.. toctree::
   :maxdepth: 2
   :hidden:

   install/virtualenv_pip
   install/conda_mamba
   install/config_workdirs
   install/ide_integration
   install/verification
   install/upgrading
   install/troubleshooting
