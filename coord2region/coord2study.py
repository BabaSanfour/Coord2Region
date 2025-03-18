#!/usr/bin/env python3
import os
import sys
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from nimare.extract import fetch_neurosynth, fetch_neuroquery
from nimare.io import convert_neurosynth_to_dataset
from nimare.dataset import Dataset

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter("%(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

try:
    from Bio import Entrez, Medline
    BIO_AVAILABLE = True
except ImportError:
    BIO_AVAILABLE = False
    logger.warning("Biopython not found. Abstract fetching will be disabled.")


def fetch_datasets(data_dir: str) -> Dict[str, Dataset]:
    """
    Fetch and convert Neurosynth and NeuroQuery datasets into NiMARE Dataset objects.
    
    :param data_dir: Directory to store downloaded data.
    :return: Dictionary of NiMARE Dataset objects indexed by dataset name.        
    """
    datasets: Dict[str, Dataset] = {}
    os.makedirs(data_dir, exist_ok=True)
    
    # Fetch Neurosynth data
    try:
        ns_files = fetch_neurosynth(
            data_dir=data_dir,
            version="7",
            source="abstract",
            vocab="terms",
            overwrite=False
        )
        ns_data = ns_files[0]  # fetch_neurosynth returns a list of dicts
        neurosynth_dset = convert_neurosynth_to_dataset(
            coordinates_file=ns_data["coordinates"],
            metadata_file=ns_data["metadata"],
            annotations_files=ns_data.get("features")
        )
        datasets["Neurosynth"] = neurosynth_dset
        logger.info("Neurosynth dataset loaded successfully.")
    except Exception as e:
        logger.warning(f"Failed to fetch/convert Neurosynth dataset: {e}")

    # Fetch NeuroQuery data
    try:
        nq_files = fetch_neuroquery(
            data_dir=data_dir,
            version="1",
            source="combined",
            vocab="neuroquery6308",
            type="tfidf",
            overwrite=False
        )
        nq_data = nq_files[0]
        neuroquery_dset = convert_neurosynth_to_dataset(
            coordinates_file=nq_data["coordinates"],
            metadata_file=nq_data["metadata"],
            annotations_files=nq_data.get("features")
        )
        datasets["NeuroQuery"] = neuroquery_dset
        logger.info("NeuroQuery dataset loaded successfully.")
    except Exception as e:
        logger.warning(f"Failed to fetch/convert NeuroQuery dataset: {e}")

    if not datasets:
        sys.exit("Error: No datasets could be loaded. Ensure you have internet access and NiMARE supports the datasets.")
    #TODO: Add more datasets as needed.
    return datasets


def _extract_study_metadata(dset: Dataset, sid: Any) -> Dict[str, Any]:
    """
    Given a study ID from a NiMARE Dataset, extract study metadata (title and abstract if available).
    
    :param dset: A NiMARE Dataset.
    :param sid: Study ID.
    :return: A dictionary with keys "id", "source", "title", and optionally "abstract".
    """
    study_entry: Dict[str, Any] = {"id": str(sid)}
    
    title: Optional[str] = None
    try:
        titles = dset.get_metadata(ids=[sid], field="title")
        if titles and titles[0] not in (None, "", "NaN"):
            title = titles[0]
    except Exception:
        try:
            authors = dset.get_metadata(ids=[sid], field="authors")
            year = dset.get_metadata(ids=[sid], field="year")
            if authors and year:
                title = f"{authors[0]} ({year[0]})"
        except Exception:
            title = None
    if title:
        study_entry["title"] = title

    # Optionally, retrieve abstract using PubMed via Entrez if email provided and Bio is available.
    if BIO_AVAILABLE and study_entry.get("id") and "email" in study_entry:
        pmid = str(sid).split("-")[0]  
        try:
            handle = Entrez.efetch(db="pubmed", id=pmid, rettype="medline", retmode="text")
            records = list(Medline.parse(handle))
            if records:
                rec = records[0]
                abstract_text = rec.get("AB")
                if abstract_text:
                    study_entry["abstract"] = abstract_text.strip()
                if "title" not in study_entry:
                    pub_title = rec.get("TI")
                    if pub_title:
                        study_entry["title"] = pub_title.strip()
        except Exception as e:
            logger.warning(f"Failed to fetch abstract for PMID {pmid}: {e}")
    return study_entry


def get_studies_for_coordinate(
    datasets: Dict[str, Dataset],
    coord: Union[List[float], Tuple[float, float, float]],
    email: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Given an MNI coordinate and a dict of NiMARE datasets, return study metadata for studies
    that report that coordinate.
    
    :param datasets: Dictionary of NiMARE Dataset objects keyed by source name.
    :param coord: MNI coordinate [x, y, z].
    :param email: Email address to use with Entrez for abstract fetching (if available).
    :return: List of study metadata dictionaries.
    """
    # NiMARE expects a list of coordinates.
    coord_list = [list(coord)]
    studies_info: List[Dict[str, Any]] = []

    for source, dset in datasets.items():
        try:
            study_ids = dset.get_studies_by_coordinate(coord_list, r=0)
        except Exception as e:
            logger.warning(f"Failed to search coordinate {coord} in {source} dataset: {e}")
            continue
        if not study_ids:
            continue

        for sid in study_ids:
            study_entry = {"id": str(sid), "source": source}
            if email:
                Entrez.email = email
                study_entry["email"] = email  

            study_metadata = _extract_study_metadata(dset, sid)
            study_entry.update(study_metadata)
            studies_info.append(study_entry)

    return studies_info


# Example usage:
if __name__ == '__main__':
    DATA_DIR = "nimare_data"
    # Fetch datasets (Neurosynth and NeuroQuery)
    nimare_datasets = fetch_datasets(DATA_DIR)
    # Example coordinate (MNI)
    coordinate = [-30, -22, 50]
    # Optionally, provide an email for fetching PubMed abstracts.
    email_address = "babasanfour1503@gmail.com"  
    studies = get_studies_for_coordinate(nimare_datasets, coordinate, email=email_address)
    for study in studies:
        print(study)
