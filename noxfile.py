"""Nox sessions for the Coord2Region package."""

import nox

# Only test on versions we support
nox.options.sessions = ["lint", "tests"]


@nox.session(python=["3.10", "3.11", "3.12", "3.13", "3.14"])
def tests(session):
    """Run the test suite."""
    # Install the package and the 'dev' dependencies
    session.install(".[dev]")
    # Run pytest
    session.run("pytest", *session.posargs)


@nox.session
def lint(session):
    """Run linters."""
    session.install(".[dev]")
    session.run("ruff", "check", ".")
    session.run("ruff", "format", "--check", ".")


@nox.session
def docs(session):
    """Build the documentation."""
    session.install(".[docs]")
    with session.chdir("docs"):
        session.run("make", "html", external=True)
