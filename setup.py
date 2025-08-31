"""Setup script for the coord2region package."""

import setuptools
import subprocess
from pathlib import Path

# Try retrieving the version dynamically
try:
    version = (
        subprocess.check_output(["git", "describe", "--abbrev=0", "--tags"], stderr=subprocess.DEVNULL)
        .strip()
        .decode("utf-8")
    )
except subprocess.CalledProcessError:
    try:
        from importlib.metadata import version as get_version
        version = get_version("coord2region")
    except Exception:
        version = "0.0.1"  # Default fallback version

__version__ = version

# Read long description from README.md
long_description = Path("README.md").read_text(encoding="utf-8")

setuptools.setup(
    name="coord2region",
    version=__version__,
    author="Hamza Abdelhedi",
    author_email="hamza.abdelhedii@gmail.com",
    description="Find region name for a given MNI coordinate in a selected atlas",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        'nilearn @ git+https://github.com/BabaSanfour/nilearn.git@main#egg=nilearn',
        'pyyaml',
        'pandas',
        'openpyxl'
    ],
    include_package_data=True,
    entry_points={
        'console_scripts': ['coord2region=coord2region.cli:main']
    },
)
