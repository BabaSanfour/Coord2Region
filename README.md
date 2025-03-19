# Coord2Region

![Codecov](https://img.shields.io/codecov/c/github/BabaSanfour/Coord2Region)
[![Test Status](https://img.shields.io/github/actions/workflow/status/BabaSanfour/Coord2Region/python-tests.yml?branch=main&label=tests)](https://github.com/BabaSanfour/Coord2Region/actions?query=workflow%3Apython-tests)
[![Documentation Status](https://readthedocs.org/projects/coord2region/badge/?version=latest)](https://coord2region.readthedocs.io/en/latest/?badge=latest)
[![GitHub Repository](https://img.shields.io/badge/Source%20Code-BabaSanfour%2FCoord2Region-blue)](https://github.com/BabaSanfour/Coord2Region)
[![Preprint](https://img.shields.io/badge/Preprint-Zenodo-orange)](https://zenodo.org/records/15048848)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD3Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

**coord2region** is an open‐source Python package designed to simplify neuroimaging workflows by automatically mapping Montreal Neurological Institute (MNI) coordinates to anatomical brain regions. It supports both single-coordinate lookups and high-throughput batch processing across multiple brain atlases. The package also integrates with meta-analytic resources like Neurosynth and NeuroQuery to link anatomical regions with published neuroimaging literature.

## Features

- **Automated Anatomical Mapping:** Quickly retrieve region names for any given MNI coordinate.
- **Batch Processing:** Efficiently map large sets of coordinates using vectorized operations.
- **Multiple Atlas Support:** Compare and cross-validate results using different atlases (e.g., Harvard-Oxford, Juelich, Schaefer, Yeo).
- **Meta-analytic Integration:** Interface with Neurosynth and NeuroQuery for enriched functional interpretations.
- **Extensible and Modular:** Easily add new atlases or extend functionality.

## Installation

### Prerequisites

- Python 3.8 or higher

### Installing for End Users

You can install coord2region in your own virtual environment using either conda or Python's venv:

#### Using Conda
```bash
conda create -n coord2region python=3.8
conda activate coord2region
pip install -r requirements.txt
```

#### Using venv
```bash
python -m venv env
# On Linux/Mac:
source env/bin/activate
# On Windows:
env\Scripts\activate
pip install -r requirements.txt
```

Alternatively, clone the repository and install using pip:
```bash
git clone https://github.com/BabaSanfour/Coord2Region.git
cd coord2region
pip install .
```

### Installing for Development

If you wish to contribute or modify the package, set up a development environment:
1. Create and activate your virtual environment (see above).
2. Install the runtime requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Install additional development requirements:
   ```bash
   pip install -r requirements-dev.txt
   ```
4. (Optional) Run tests with:
   ```bash
   pytest
   ```

## Usage

Below is a simple example of how to use coord2region in your Python code:

```python
from coord2region.coord2region import AtlasMapper, BatchAtlasMapper, MultiAtlasMapper
from coord2region.fetching import AtlasFetcher

# Fetch an atlas (e.g., Harvard-Oxford)
af = AtlasFetcher()
atlas_data = af.fetch_atlas('harvard-oxford')

# Create an AtlasMapper
mapper = AtlasMapper(
    name='harvard-oxford',
    vol=atlas_data['vol'],
    hdr=atlas_data['hdr'],
    labels=atlas_data['labels']
)

# Map a single MNI coordinate to its region name
mni_coord = [30, -22, 50]
region_name = mapper.mni_to_region_name(mni_coord)
print(f"The region for coordinate {mni_coord} is {region_name}")

# For batch processing:
batch_mapper = BatchAtlasMapper(mapper)
region_names = batch_mapper.batch_mni_to_region_name([[30, -22, 50], [10, 20, 30]])
print("Batch mapped regions:", region_names)
```

For more examples and detailed API usage, please refer to the code in `combined_python_code.txt` and the examples in the repository.

## Documentation, Examples and Paper

For a comprehensive overview and detailed description of the package design, please see the 
**coord2region** [documentation](https://coord2region.readthedocs.io/en/latest/) and [Preprint](https://zenodo.org/records/15048848)

## Contributing

Contributions are welcome! If you would like to contribute:

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add YourFeature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

Please ensure that you have installed the development dependencies and that all tests pass before submitting a pull request.

## License

This project is open-source. Please refer to the LICENSE file for details.

## Citation

If you use coord2region in your research, please cite the associated paper.