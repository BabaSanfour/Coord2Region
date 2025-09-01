# Contributing

## Development environment

Install the package with its development extras (after the requirements files
are removed):

```bash
pip install .[dev]
```

## Pre-commit, linting, and formatting

This project uses [pre-commit](https://pre-commit.com) with
[Ruff](https://github.com/astral-sh/ruff) to enforce PEP8 and NumPy-style
docstrings. All commits must pass the pre-commit checks.

Set up the git hook once:

```bash
pre-commit install
```

Run the checks on all files before committing:

```bash
pre-commit run --all-files
```

## Testing and coverage

Run the unit test suite with coverage. The project aims for at least 80%
coverage, which is enforced by Codecov:

```bash
pytest --cov
```

## Documentation

Write docstrings in [numpydoc](https://numpydoc.readthedocs.io) style and build
the documentation with [Sphinx](https://www.sphinx-doc.org):

```bash
make -C docs html
```

## Versioning and releases

This project follows [Semantic Versioning](https://semver.org). Tag releases
using annotated Git tags:

```bash
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

## Code of Conduct and security

Please review our [Code of Conduct](CODE_OF_CONDUCT.md) and
[security policy](SECURITY.md) before contributing.
