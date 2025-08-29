# File: tests/test_coord2study.py

import pytest
import os
import logging
from unittest.mock import patch, MagicMock

from coord2region.coord2study import (
    fetch_datasets,
    get_studies_for_coordinate,
    _extract_study_metadata,
    remove_duplicate_studies,
    prepare_datasets,
)

@pytest.mark.integration
@pytest.mark.requires_network

def test_fetch_datasets_integration(tmp_path):
    """
    Integration test: fetch real NiMARE datasets (Neurosynth & NeuroQuery),
    assuming internet access and a working NiMARE environment.
    """
    data_dir = tmp_path / "nimare_data"  # Use a temporary dir
    data_dir.mkdir(exist_ok=True)

    # This will actually download (unless cached) the datasets.
    # If you want to skip real data fetch, remove or mark with @pytest.mark.skip
    dsets = fetch_datasets(str(data_dir), neurosynth=False, neuroquery=False)
    assert isinstance(dsets, dict), "fetch_datasets must return a dict"
    # Expect at least Neurosynth or NeuroQuery in the dictionary
    assert len(dsets) > 0, "Expected at least one dataset"

    for name, dset in dsets.items():
        assert hasattr(dset, "coordinates"), f"NiMARE Dataset missing 'coordinates' for {name}"

@pytest.mark.integration
@pytest.mark.requires_network
def test_get_studies_for_coordinate_integration(tmp_path):
    """
    Integration test: downloads real NiMARE data and queries a coordinate.
    The result can be non-empty or empty, depending on the coordinate.
    """
    data_dir = tmp_path / "nimare_data"
    data_dir.mkdir(exist_ok=True)
    dsets = fetch_datasets(str(data_dir),neurosynth=False, neuroquery=False)

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
    results = get_studies_for_coordinate(
        dsets, coord=[-30, -22, 50], email="test@example.com"
    )
    
    assert len(results) == 1
    first = results[0]
    assert first["id"] == "123456"
    assert first["source"] == "MockSource"
    assert first["title"] == "Mock Title"
    assert first["email"] == "test@example.com"


@pytest.mark.unit
def test_get_studies_for_coordinate_radius():
    """Non-zero radius should change query results and be forwarded to NiMARE."""

    mock_dataset = MagicMock()

    def side_effect(coords, r):
        # Return a study only when radius is non-zero
        return ["123456"] if r > 0 else []

    mock_dataset.get_studies_by_coordinate.side_effect = side_effect
    mock_dataset.get_metadata.side_effect = lambda ids, field: {
        "title": ["Mock Title"],
        "authors": ["Mock Author"],
    }.get(field, [None])

    dsets = {"MockSource": mock_dataset}

    coord = [-30, -22, 50]
    no_hits = get_studies_for_coordinate(dsets, coord=coord, radius=0)
    hits = get_studies_for_coordinate(dsets, coord=coord, radius=5)

    assert len(no_hits) == 0
    assert len(hits) == 1
    # Ensure the radius argument was passed through correctly
    from unittest.mock import call

    expected_coord_list = [coord]
    assert mock_dataset.get_studies_by_coordinate.call_args_list == [
        call(expected_coord_list, r=0),
        call(expected_coord_list, r=5),
    ]

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
@pytest.mark.unit
def test_remove_duplicate_studies():
    """
    Tests that remove_duplicate_studies properly removes duplicates 
    by extracting the PMID from the 'id' field (e.g. '123456-1' -> '123456').
    """
    # Suppose we have multiple entries with the same PMID but different sources or suffixes
    studies = [
        {"id": "123456-1", "title": "Study A", "source": "Neurosynth"},
        {"id": "123456-2", "title": "Study A (dup)", "source": "NeuroQuery"},
        {"id": "19224116-1", "title": "Study B", "source": "Neurosynth"},
        {"id": "19224116-1", "title": "Study B (exact dup)", "source": "Neurosynth"},
        {"id": "999999-9", "title": "Study C", "source": "NeuroQuery"},
    ]
    # After removing duplicates by PMID, we should only have one of each unique PMID: 123456, 19224116, 999999
    # The logic in remove_duplicate_studies() picks the first occurrence for each PMID key
    cleaned = remove_duplicate_studies(studies)

    # We expect 3 unique PMIDs in the result
    assert len(cleaned) == 3, f"Expected 3 unique entries but got {len(cleaned)}"

    # We expect only the first occurrence of '123456' to appear
    # which is the '123456-1' from Neurosynth
    pmids = [entry["id"].split("-")[0] for entry in cleaned]
    assert "123456" in pmids
    assert "19224116" in pmids
    assert "999999" in pmids

    # Make sure we don't see the second or third duplicates in the result.
    # Specifically, '123456-2' should be replaced by '123456-1'
    # and '19224116-1' repeated entry is also consolidated.
    # The test doesn't necessarily require us to check the "source"
    # but let's confirm the code used the FIRST entry for '123456':
    # i.e., the "title" is "Study A" from "Neurosynth"
    for entry in cleaned:
        if entry["id"].startswith("123456"):
            assert entry["title"] == "Study A"
            assert entry["source"] == "Neurosynth"
        if entry["id"].startswith("19224116"):
            assert entry["title"] == "Study B"
            assert entry["source"] == "Neurosynth"


@pytest.mark.unit
@patch("coord2region.coord2study.fetch_datasets")
@patch("coord2region.coord2study.deduplicate_datasets")
@patch("coord2region.coord2study.load_deduplicated_dataset")
@patch("coord2region.coord2study.os.path.exists")
def test_prepare_datasets_uses_cache(mock_exists, mock_load, mock_dedup, mock_fetch, tmp_path):
    """If a cached dataset exists it should be loaded without fetching."""
    mock_exists.return_value = True
    mock_dataset = MagicMock()
    mock_load.return_value = mock_dataset

    result = prepare_datasets(str(tmp_path), neurosynth=False, neuroquery=False)
    assert result is mock_dataset
    mock_fetch.assert_not_called()
    mock_dedup.assert_not_called()


@pytest.mark.unit
@patch("coord2region.coord2study.fetch_datasets")
@patch("coord2region.coord2study.deduplicate_datasets")
@patch("coord2region.coord2study.load_deduplicated_dataset")
@patch("coord2region.coord2study.os.path.exists")
def test_prepare_datasets_fetches_when_missing(mock_exists, mock_load, mock_dedup, mock_fetch, tmp_path):
    """When no cache exists datasets should be fetched and deduplicated."""
    mock_exists.return_value = False
    mock_dataset = MagicMock()
    mock_fetch.return_value = {"A": MagicMock()}
    mock_dedup.return_value = mock_dataset

    result = prepare_datasets(str(tmp_path))
    assert result is mock_dataset
    mock_fetch.assert_called_once()
    mock_dedup.assert_called_once()
