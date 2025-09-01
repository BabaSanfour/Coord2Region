# Contributing

## Development environment

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

## Linting and formatting

This project uses [pre-commit](https://pre-commit.com) with
[Ruff](https://github.com/astral-sh/ruff) to enforce PEP8 and NumPy-style
docstrings.

Set up the git hook once:

```bash
pre-commit install
```

Run the checks on all files before committing:

```bash
pre-commit run --all-files
```

## Testing

Run the unit test suite:

```bash
pytest
```
