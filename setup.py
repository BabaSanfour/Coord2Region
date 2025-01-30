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
        version = get_version("sourcelocalizer")
    except Exception:
        version = "0.0.1"  # Default fallback version

__version__ = version

# Read long description from README.md
long_description = Path("README.md").read_text(encoding="utf-8")

setuptools.setup(
    name="sourcelocalizer",
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
    python_requires=">=3.8",
    install_requires=[
        "nibabel",  
    ],
    include_package_data=True,
)