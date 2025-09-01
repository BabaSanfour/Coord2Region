"""Demonstrate dataset caching with :func:`prepare_datasets`.

Running this script twice shows that the second invocation loads the
previously merged dataset from disk instead of downloading it again.
"""

import errno
import logging
import os
import sys

import requests

from coord2region.coord2study import prepare_datasets
from coord2region.paths import get_data_directory

# Use a custom data directory; the deduplicated dataset will be stored in
# ``<data_dir>/cached_data`` alongside any downloaded atlases.
data_dir = get_data_directory("coord2region_example")

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Using data directory: %s", data_dir)

# First call will download/merge if the cache file is missing
try:
    merged = prepare_datasets(data_dir, sources=["nidm_pain"])  # limit to small dataset for speed
except requests.exceptions.RequestException as exc:
    logger.error("Failed to download datasets: %s", exc)
    sys.exit(1)
except PermissionError as exc:
    logger.error("Permission denied while preparing datasets: %s", exc)
    sys.exit(1)
except OSError as exc:
    if exc.errno == errno.ENOSPC:
        logger.error("Insufficient disk space while preparing datasets: %s", exc)
    else:
        logger.error("OS error while preparing datasets: %s", exc)
    sys.exit(1)
except Exception as exc:
    logger.error("An unexpected error occurred while preparing datasets: %s", exc)
    sys.exit(1)
else:
    logger.info("Merged dataset contains %d studies", len(merged.ids))

# Second call immediately reuses the cache
try:
    merged_again = prepare_datasets(data_dir, sources=["nidm_pain"])
except requests.exceptions.RequestException as exc:
    logger.error("Failed to download datasets: %s", exc)
    sys.exit(1)
except PermissionError as exc:
    logger.error("Permission denied while loading cached dataset: %s", exc)
    sys.exit(1)
except OSError as exc:
    if exc.errno == errno.ENOSPC:
        logger.error("Insufficient disk space while loading cached dataset: %s", exc)
    else:
        logger.error("OS error while loading cached dataset: %s", exc)
    sys.exit(1)
except Exception as exc:
    logger.error("An unexpected error occurred while loading cached dataset: %s", exc)
    sys.exit(1)
else:
    logger.info("Loaded cached dataset with %d studies", len(merged_again.ids))
