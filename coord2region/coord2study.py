import os
import sys
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from nimare.extract import fetch_neurosynth, fetch_neuroquery, download_nidm_pain
from nimare.utils import get_resource_path
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

# TODO: Add more nimare datasets ! and optinally private datasets

def fetch_datasets(data_dir: str, neurosynth: bool = True, neuroquery: bool = True) -> Dict[str, Dataset]:
    """
    Fetch and convert Neurosynth and NeuroQuery datasets into NiMARE Dataset objects.
    
    :param data_dir: Directory to store downloaded data.
    :return: Dictionary of NiMARE Dataset objects indexed by dataset name.        
    """
    datasets: Dict[str, Dataset] = {}
    os.makedirs(data_dir, exist_ok=True)
    
    if neurosynth:
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

    if neuroquery:
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

    # Fetch NIDM-Pain data
    try:
        nidm_pain_file = download_nidm_pain(data_dir=data_dir, overwrite=False)
        dset_file = os.path.join(get_resource_path(), "nidm_pain_dset.json")
        nidm_pain_dset = Dataset(dset_file,target="mni152_2mm", mask=None)
        datasets["NIDM-Pain"] = nidm_pain_dset
        logger.info("NIDM-Pain dataset loaded successfully.")
    except Exception as e:
        logger.warning(f"Failed to fetch/convert NIDM-Pain dataset: {e}")

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

def remove_duplicate_studies(studies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique = {}
    for st in studies:
        # If IDs are like '24984958-1', split at the dash
        full_id = st.get("id", "")
        # Keep only the left part of the dash (the actual PMID)
        pmid = full_id.split("-")[0]
        # Use that as your key to unify across sources
        key = pmid

        if key not in unique:
            unique[key] = st
        else:
            # Optionally merge or just skip
            pass
    return list(unique.values())


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
    # Remove duplicates before returning
    studies_info = remove_duplicate_studies(studies_info)
    return studies_info