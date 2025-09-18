import os
from pathlib import Path

import pytest

HCP_LICENSE_ENV = "COORD2REGION_ACCEPT_HCPMMP"
HCP_LICENSE_PATH = Path.home() / ".mne" / "hcpmmp-license.txt"


@pytest.fixture(scope="session", autouse=True)
def _accept_hcp_license() -> None:
    """Ensure tests can access HCP-MMP without interactive prompts."""
    license_path = HCP_LICENSE_PATH.expanduser()
    license_path.parent.mkdir(parents=True, exist_ok=True)
    if not license_path.exists():
        license_path.write_text("accepted", encoding="utf8")
    os.environ.setdefault(HCP_LICENSE_ENV, "1")
