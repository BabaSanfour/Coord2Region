# File: tests/test_coord2study.py

import pytest
import os
import logging
from unittest.mock import patch, MagicMock

from coord2region.coord2study import (
    fetch_datasets,
    get_studies_for_coordinate,
    _extract_study_metadata
)

@pytest.mark.integration
def test_fetch_datasets_integration(tmp_path):
    """
    Integration test: fetch real NiMARE datasets (Neurosynth & NeuroQuery),
    assuming internet access and a working NiMARE environment.
    """
    data_dir = tmp_path / "nimare_data"  # Use a temporary dir
    data_dir.mkdir(exist_ok=True)

    # This will actually download (unless cached) the datasets.
    # If you want to skip real data fetch, remove or mark with @pytest.mark.skip
    dsets = fetch_datasets(str(data_dir))
    assert isinstance(dsets, dict), "fetch_datasets must return a dict"
    # Expect at least Neurosynth or NeuroQuery in the dictionary
    assert len(dsets) > 0, "Expected at least one dataset"

    for name, dset in dsets.items():
        assert hasattr(dset, "coordinates"), f"NiMARE Dataset missing 'coordinates' for {name}"

@pytest.mark.integration
def test_get_studies_for_coordinate_integration(tmp_path):
    """
    Integration test: downloads real NiMARE data and queries a coordinate.
    The result can be non-empty or empty, depending on the coordinate.
    """
    data_dir = tmp_path / "nimare_data"
    data_dir.mkdir(exist_ok=True)
    dsets = fetch_datasets(str(data_dir))

    # Known coordinate for testing (this might or might not yield hits)
    coordinate = (-30, -22, 50)
    results = get_studies_for_coordinate(dsets, coordinate)
    assert isinstance(results, list), "get_studies_for_coordinate should return a list"
    # We can't guarantee a non-empty result, but let's just confirm it doesn't crash:
    for entry in results:
        assert "id" in entry
        assert "source" in entry

@pytest.mark.unit
def test_get_studies_for_coordinate_empty_dict():
    """
    Unit test: If datasets dict is empty, we expect an empty list (no hits).
    """
    results = get_studies_for_coordinate({}, coord=[-30, -22, 50])
    assert isinstance(results, list)
    assert len(results) == 0

@pytest.mark.unit
def test_get_studies_for_coordinate_mock():
    """
    Unit test with mocking the NiMARE Dataset call.
    Ensures that get_studies_for_coordinate() behaves as expected 
    without needing real data.
    """
    # Create a fake NiMARE Dataset that returns a known study ID
    mock_dataset = MagicMock()
    mock_dataset.get_studies_by_coordinate.return_value = ["123456"]
    # Minimal mocking for _extract_study_metadata
    mock_dataset.get_metadata.side_effect = lambda ids, field: (
        ["Mock Title"] if field == "title" else ["Mock Author"]
    )

    dsets = {"MockSource": mock_dataset}
    results = get_studies_for_coordinate(dsets, coord=[-30, -22, 50], email="test@example.com")
    
    assert len(results) == 1
    first = results[0]
    assert first["id"] == "123456"
    assert first["source"] == "MockSource"
    assert first["title"] == "Mock Title"
    assert first["email"] == "test@example.com"

@pytest.mark.unit
def test_extract_study_metadata_mock():
    """
    Tests the _extract_study_metadata function directly.
    This does not require NiMARE to fetch any data from the internet.
    """
    # Create a mock NiMARE Dataset
    mock_dataset = MagicMock()

    # Return some dummy metadata
    mock_dataset.get_metadata.side_effect = lambda ids, field: {
        "title":    ["Example Title"],
        "authors":  ["John Doe"],
        "year":     [2021],
    }.get(field, [None])

    # Call the function
    entry = _extract_study_metadata(mock_dataset, sid="12345")
    assert entry["id"] == "12345"
    assert entry["title"] == "Example Title"

    # If you want to test the PubMed retrieval portion, you could also patch Bio.Entrez calls.
