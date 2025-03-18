# File: tests/test_fetching.py

import pytest
import os
from unittest.mock import patch, MagicMock
from coord2region.fetching import AtlasFileHandler, AtlasFetcher

@pytest.fixture
def mock_data_dir(tmp_path):
    """
    A pytest fixture that creates a temporary data directory
    so tests don’t write to your real filesystem.
    """
    return str(tmp_path / "test_fetch_data")

def test_atlasfilehandler_init(mock_data_dir):
    """
    Basic test: AtlasFileHandler initializes with a valid data directory.
    """
    handler = AtlasFileHandler(data_dir=mock_data_dir)
    assert os.path.isdir(handler.data_dir)
    assert os.access(handler.data_dir, os.W_OK)

def test_atlasfilehandler_init_failure(tmp_path):
    """
    Test: AtlasFileHandler raises an error if data_dir is not writable.
    """
    # Create a read-only directory:
    ro_dir = tmp_path / "readonly"
    ro_dir.mkdir()
    ro_dir.chmod(0o444)  # Make it read-only

    with pytest.raises(ValueError) as excinfo:
        _ = AtlasFileHandler(data_dir=str(ro_dir))
    assert "is not writable" in str(excinfo.value)

@pytest.mark.unit
def test_fetch_from_local(mock_data_dir):
    """
    Test: fetch_from_local using a mocked pack_vol_output and fetch_labels.
    """
    handler = AtlasFileHandler(data_dir=mock_data_dir)

    # Mock the methods that actually load files
    with patch.object(handler, 'pack_vol_output', return_value={"vol": "MOCK_VOL", "hdr": "MOCK_HDR"}) as mock_pack:
        with patch.object(handler, 'fetch_labels', return_value=["RegionA", "RegionB"]) as mock_lbl:
            atlas_dict = handler.fetch_from_local("/fake/path/to_atlas.nii.gz", labels=["dummy"])
            assert atlas_dict["vol"] == "MOCK_VOL"
            assert atlas_dict["hdr"] == "MOCK_HDR"
            assert atlas_dict["labels"] == ["RegionA", "RegionB"]

            mock_pack.assert_called_once_with("/fake/path/to_atlas.nii.gz")
            mock_lbl.assert_called_once()

@pytest.mark.unit
def test_fetch_from_url_success(mock_data_dir):
    """
    Test: fetch_from_url with a mock request that simulates a successful download.
    """
    handler = AtlasFileHandler(data_dir=mock_data_dir)
    fake_url = "https://example.com/dummy_atlas.nii.gz"
    local_path = os.path.join(handler.data_dir, "dummy_atlas.nii.gz")

    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"FAKE_CONTENT"]
    mock_response.__enter__.return_value = mock_response
    mock_response.raise_for_status.return_value = None

    with patch("requests.get", return_value=mock_response) as mock_get:
        downloaded_path = handler.fetch_from_url(fake_url)
        assert downloaded_path == local_path
        assert os.path.exists(local_path)
        mock_get.assert_called_once_with(fake_url, stream=True, timeout=30, verify=False)

@pytest.mark.unit
def test_fetch_from_url_failure(mock_data_dir):
    """
    Test: fetch_from_url simulates a download error.
    """
    handler = AtlasFileHandler(data_dir=mock_data_dir)
    fake_url = "https://example.com/bogus_atlas.nii.gz"

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = Exception("Download failed!")
    mock_response.__enter__.return_value = mock_response

    with patch("requests.get", return_value=mock_response):
        with pytest.raises(RuntimeError) as excinfo:
            handler.fetch_from_url(fake_url)
        assert "Failed to download from" in str(excinfo.value)

@pytest.mark.unit
def test_fetch_atlas_nilearn(mock_data_dir):
    """
    Test: AtlasFetcher fetches a known Nilearn atlas (mocked).
    """
    fetcher = AtlasFetcher(data_dir=mock_data_dir)

    # Patch the internal call to nilearn's fetch function
    with patch.object(fetcher, '_fetch_atlas', return_value={
        'maps': '/fake/path.nii.gz',
        'labels': ['Region1', 'Region2'],
        'description': 'Mocked Description'
    }) as mock_internal:
        atlas_data = fetcher.fetch_atlas("harvard-oxford")
        assert "vol" in atlas_data
        assert "hdr" in atlas_data
        assert len(atlas_data['labels']) == 2
        mock_internal.assert_called_once()

@pytest.mark.unit
def test_fetch_atlas_url(mock_data_dir):
    """
    Test: fetch_atlas with a direct URL provided.
    We mock fetch_from_url -> fetch_from_local flow.
    """
    fetcher = AtlasFetcher(data_dir=mock_data_dir)
    fake_url = "https://example.com/fake_atlas.nii.gz"

    # First, mock file_handler.fetch_from_url to return a "downloaded" path
    with patch.object(fetcher.file_handler, 'fetch_from_url', return_value="/some/local/path.nii.gz") as mock_dl:
        # Then, mock file_handler.fetch_from_local to simulate loaded data
        with patch.object(fetcher.file_handler, 'fetch_from_local', return_value={"vol":"FAKEVOL","hdr":"FAKEHDR","labels":[]}) as mock_local:
            atlas_data = fetcher.fetch_atlas(atlas_name="someName", atlas_url=fake_url)
            mock_dl.assert_called_once_with(fake_url)
            mock_local.assert_called_once_with("/some/local/path.nii.gz")

    assert atlas_data["vol"] == "FAKEVOL"

# pytest -v tests/test_fetching.py
# pytest -v --maxfail=1 tests/test_fetching.py
# pytest -v --pdb tests/test_fetching.py

# {
#   "name": "Pytest Debugger: test_fetching",
#   "type": "python",   // or "debugpy" depending on your setup
#   "request": "launch",
#   "module": "pytest",
#   "args": [
#     "tests/test_fetching.py",
#     "-v", 
#     "--pdb"
#   ],
#   "console": "integratedTerminal",
#   "justMyCode": false
# }
