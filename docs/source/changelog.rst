.. _changelog:

Changelog
=========

This page lists the major changes and contributions for each release of **Coord2Region**.
We follow a structure inspired by MNE-Python to provide clear and concise update notes.

.. contents:: Contents
   :local:
   :depth: 1

Current (Unreleased)
--------------------

Added
~~~~~
- Initial implementation of a release-based **Changelog** system (:pr:`#N/A`).

Changed
~~~~~~~
- Modernized repository configuration to satisfy sp-repo-review standards, including PEP 723/735, comprehensive Ruff linting, and Pytest enhancements (:pr:`34`).

Fixed
~~~~~
- Resolved **Arbitrary File Write and XML injection** in ``fonttools`` (CVE-2024-52233).
- Resolved **TOCTOU Symlink Vulnerability** in ``filelock`` (CVE-2024-56334).
- Resolved **Prototype Pollution** in ``js-yaml`` and ``lodash`` (NPM dependencies).
- Resolved security vulnerability in ``pyasn1``.
- Resolved **Out-of-bounds write** in ``Pillow``.
- Resolved **Decompression-bomb safeguards bypass** in ``urllib3``.

Security
~~~~~~~~
- Added dependency version pins in ``pyproject.toml`` to ensure secure environments.

Authors
~~~~~~~
The following people contributed to this release:

- **Hamza Abdelhedi** (Security remediations, changelog implementation)

v0.1.4
------
*Initial public release.*
- Support for +20 anatomical atlases.
- Coordinate-to-region mapping via NiMARE/nilearn.
- Integration with Neurosynth and NeuroQuery meta-analytic resources.
- Interactive web-based configuration builder.
- Reproducible pipeline for batch processing.
