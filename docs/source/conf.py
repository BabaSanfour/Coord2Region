"""Sphinx configuration for the coord2region documentation."""

import importlib
import os
import sys
import warnings
from datetime import date

from packaging.version import InvalidVersion, Version

# Add the project root to sys.path so sphinx can find the package
curdir = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(curdir, "..", "..")))

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def _has_module(module_name: str) -> bool:
    """Return True if ``module_name`` can be imported in this environment."""
    try:
        importlib.import_module(module_name)
    except ImportError:
        return False
    return True


def _env_flag(name: str) -> bool:
    """Interpret environment variables such as ``CI`` or feature flags."""
    value = os.environ.get(name, "")
    return value.lower() in {"1", "true", "yes", "on"}


# Documentation builds use the official theme by default. Developers may opt
# into the bare-bones fallback by setting COORD2REGION_DOCS_ALLOW_FALLBACK=1.
allow_theme_fallback = _env_flag("COORD2REGION_DOCS_ALLOW_FALLBACK")
strict_docs_mode = (
    _env_flag("COORD2REGION_DOCS_STRICT")
    or _env_flag("CI")
    or os.environ.get("READTHEDOCS") == "True"
    or not allow_theme_fallback
)


def autodoc_skip_member(app, what, name, obj, skip, options):  # noqa: ARG001
    """Skip private members and internal loggers (autodoc)."""
    if name.startswith("_") or name in {"logger", "get_logger"}:
        return True
    return skip


def autoapi_skip_member(app, what, name, obj, skip, options):  # noqa: ARG001
    """Skip irrelevant members in AutoAPI output.

    - Hide private members
    - Hide module/class attributes named ``logger`` or ``get_logger``
    - Optionally let other decisions stand (``skip``)
    """
    try:
        if name and name.startswith("_"):
            return True
    except Exception:
        pass
    if name in {"logger", "get_logger"}:
        return True
    return skip


# -----------------------------------------------------------------------------
# Project information
# -----------------------------------------------------------------------------

project = "coord2region"
author = "coord2region developers"
_today = date.today()
copyright = (
    f"2025-{_today.year}, coord2region developers. Last updated {_today.isoformat()}"
)

version = "0.1.2"
release = version
version_match = os.environ.get("READTHEDOCS_VERSION") or release

# -----------------------------------------------------------------------------
# General configuration
# -----------------------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",      # Required for MNE-style API tables
    "sphinx.ext.intersphinx",      # Links to external docs (numpy, pandas, mne)
    "sphinx.ext.viewcode",         # Add links to highlighted source code
    "sphinx.ext.napoleon",         # Parse Google/NumPy style docstrings
    "sphinx_gallery.gen_gallery",  # Generate the example gallery
    "sphinx_copybutton",           # Copy button for code blocks
    "sphinxcontrib.mermaid",       # Diagrams
    "myst_parser",                 # Markdown support
    "sphinx.ext.ifconfig",         # Conditional content
]

# Allow Markdown files to be used as documentation pages
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
python_use_unqualified_type_names = True
suppress_warnings = [
    "ref.python",  # Suppress the "more than one target found" warning
]
master_doc = "index"

# Autosummary configuration (MNE Style)
autosummary_generate = True

# Autodoc configuration
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "show-inheritance": True,
    "exclude-members": "logger",
    "noindex": True,  # Prevents duplication if you use autosummary + autodoc
}

# Intersphinx mapping (Links to other project's documentation)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "mne": ("https://mne.tools/stable/", None),
    "nilearn": ("https://nilearn.github.io/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# Linkcheck configuration to avoid flaky CI failures
linkcheck_timeout = 60
linkcheck_ignore = [
    r"https://www\.contributor-covenant\.org/.*",
]

# Gallery Configuration
examples_dir = os.path.abspath(os.path.join(curdir, "..", "..", "examples"))
sphinx_gallery_conf = {
    "doc_module": "coord2region",
    "reference_url": {"coord2region": None},
    "examples_dirs": examples_dir,
    "gallery_dirs": "auto_examples",
    "filename_pattern": "^((?!sgskip).)*$",
    "backreferences_dir": "generated",
}

# Should the examples be executed?
run_gallery = True
if os.environ.get("READTHEDOCS") == "True":
    run_gallery = False
elif os.environ.get("COORD2REGION_DOCS_RUN_GALLERY"):
    # Allow explicit enable/disable via env var (truthy runs, falsy skips)
    run_gallery = _env_flag("COORD2REGION_DOCS_RUN_GALLERY")

sphinx_gallery_conf["plot_gallery"] = run_gallery
sphinx_gallery_conf["run_stale_examples"] = run_gallery

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "_ideas"]
templates_path = ["_templates"]

# -----------------------------------------------------------------------------
# HTML theme and navbar
# -----------------------------------------------------------------------------

MIN_PYDATA_THEME = Version("0.16.0")
theme_error = None
html_theme = "alabaster"  # safe fallback

if _has_module("pydata_sphinx_theme"):
    import pydata_sphinx_theme  # type: ignore

    raw_version = getattr(pydata_sphinx_theme, "__version__", "0")
    try:
        installed_version = Version(raw_version)
    except InvalidVersion:
        installed_version = Version("0")

    if installed_version < MIN_PYDATA_THEME:
        theme_error = (
            f"pydata-sphinx-theme>={MIN_PYDATA_THEME} required, "
            f"found {raw_version}. Run `pip install -e .[docs]`."
        )
    else:
        html_theme = "pydata_sphinx_theme"
else:
    theme_error = (
        "pydata-sphinx-theme not installed; run `pip install -e .[docs]` "
        "to build with the official theme."
    )

if theme_error:
    if strict_docs_mode:
        raise RuntimeError(
            theme_error
            + " Set COORD2REGION_DOCS_ALLOW_FALLBACK=1 to preview with the "
              "minimal theme."
        )
    warnings.warn(theme_error + " Falling back to Sphinx's default theme locally.")

html_title = "Coord2Region"
html_logo = "_static/images/logo.png"

html_theme_options = {}
if html_theme == "pydata_sphinx_theme":
    html_theme_options = {
        "logo": {
            "text": "Coord2Region",
            # Paths are relative to _static
            "image_light": "_static/images/logo.png",
            "image_dark": "_static/images/logo_darkmode.png",
        },
        "navbar_align": "content",
        "show_toc_level": 1,
        "navbar_start": ["navbar-logo"],
        "navbar_center": ["navbar-nav"],
        "navbar_end": [
            "theme-switcher",
            "version-switcher",
            "navbar-icon-links",
        ],
        "show_nav_level": 4,          # Keep AutoAPI modules expanded in the sidebar
        "collapse_navigation": False, # Prevent navigation items from collapsing
        "navigation_depth": 4,        # Tells Sphinx to generate a deeper ToC
        "switcher": {
            "json_url": "_static/version_switcher.json",
            "version_match": version_match,
        },
        "icon_links": [
            {
                "name": "GitHub",
                "url": "https://github.com/BabaSanfour/Coord2Region",
                "icon": "fa-brands fa-github",
                "type": "fontawesome",
            },
        ],
        "navigation_with_keys": True,
        "secondary_sidebar_items": [], # Remove to allow sidebar on other pages
    }

html_css_files = [
    "css/custom.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css",
]

# -----------------------------------------------------------------------------
# Static paths
# -----------------------------------------------------------------------------

html_static_path = ["_static"]
html_extra_path = []

# -----------------------------------------------------------------------------
# AutoAPI (Optional - kept for backward compatibility but guarded)
# -----------------------------------------------------------------------------

# We default this to FALSE to favor the MNE-style autosummary tables.
# To enable AutoAPI (full index generation), set COORD2REGION_DOCS_ENABLE_AUTOAPI=1
autoapi_env_override = 1 # _env_flag("COORD2REGION_DOCS_ENABLE_AUTOAPI")
autoapi_available = _has_module("autoapi") and (
    autoapi_env_override or os.environ.get("READTHEDOCS") == "True"
)

def setup(app):
    """Register build events for documentation setup."""
    app.add_config_value("autoapi_available", autoapi_available, "env")
    app.connect("autodoc-skip-member", autodoc_skip_member)
    if autoapi_available:
        app.connect("autoapi-skip-member", autoapi_skip_member)
    app.add_css_file("css/custom.css")
    app.add_js_file("js/page-flags.js")

if autoapi_available:
    extensions.append("autoapi.extension")
    
    # 1. Where to look for code (relative to conf.py)
    autoapi_dirs = ["../../coord2region"]
    
    # 2. How to generate files
    autoapi_type = "python"
    autoapi_root = "autoapi"  # The folder where rst files will be generated
    
    # 3. What to include
    autoapi_options = [
        "members",
        "undoc-members",      # Include members without docstrings
        "show-inheritance",
        "show-module-summary",
        "imported-members",   # Show members imported from other libraries (optional)
    ]
    
    # 4. Clean up the sidebar
    # We set this to False so we can manually place it in your api.rst
    autoapi_add_toctree_entry = False
    
    # 5. Ignore specific files (optional)
    autoapi_ignore = ["*tests*", "*migrations*"]
    
else:
    print("WARNING: sphinx-autoapi not installed. Run `pip install sphinx-autoapi`.")
