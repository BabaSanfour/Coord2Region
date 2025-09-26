"""Demonstrate dataset caching with :func:`prepare_datasets`
==========================================================

Running this script twice shows that the second invocation loads the
previously merged dataset from disk instead of downloading it again.
"""

import errno
import logging
import sys

import requests

from coord2region.coord2study import prepare_datasets
from coord2region.paths import get_working_directory


# Use a custom working directory; the deduplicated dataset will be stored in
# ``<working_directory>/cached_data`` alongside any downloaded atlases.
working_dir = get_working_directory("coord2region_example")

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Using working directory: %s", working_dir)


def load_dataset(path, sources, *, is_cached):
    """Wrap :func:`prepare_datasets` with unified error handling and logging."""

    action = "loading cached dataset" if is_cached else "preparing datasets"

    try:
        dataset = prepare_datasets(path, sources=sources)
    except requests.exceptions.RequestException as exc:
        logger.error("Failed to download datasets: %s", exc)
        sys.exit(1)
    except PermissionError as exc:
        logger.error("Permission denied while %s: %s", action, exc)
        sys.exit(1)
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            logger.error("Insufficient disk space while %s: %s", action, exc)
        else:
            logger.error("OS error while %s: %s", action, exc)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        logger.error("An unexpected error occurred while %s: %s", action, exc)
        sys.exit(1)

    if dataset is None:
        logger.error("No dataset returned while %s", action)
        sys.exit(1)

    if is_cached:
        logger.info("Loaded cached dataset with %d studies", len(dataset.ids))
    else:
        logger.info("Merged dataset contains %d studies", len(dataset.ids))
    return dataset


merged = load_dataset(working_dir, ["nidm_pain"], is_cached=False)
assert merged is not None

merged_again = load_dataset(working_dir, ["nidm_pain"], is_cached=True)
assert merged_again is not None
