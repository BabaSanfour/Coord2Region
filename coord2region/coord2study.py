"""
Coordinate to Study Mapping Module

This module provides functionality to map MNI coordinates to relevant neuroimaging studies
by leveraging NiMARE datasets (Neurosynth, NeuroQuery). It handles dataset fetching,
deduplication of studies across datasets, and extraction of study metadata.
"""

"""Fetch and handle neuroimaging datasets using NiMARE.

This module provides functions to download, convert, and query neuroimaging
datasets such as Neurosynth, NeuroQuery, and NIDM-Pain using the NiMARE
library.
"""

import os
import sys
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import nimare
from nimare.extract import fetch_neurosynth, fetch_neuroquery
from nimare.io import convert_neurosynth_to_dataset
from nimare.dataset import Dataset

logger = logging.getLogger(__name__)

# Check for Biopython availability (needed for abstract fetching)
try:
    from Bio import Entrez, Medline

    BIO_AVAILABLE = True
except ImportError:
    BIO_AVAILABLE = False
    logger.warning("Biopython not found. Abstract fetching will be disabled.")


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


def load_deduplicated_dataset(filepath: str) -> Optional[Dataset]:
    """
    Load a previously saved deduplicated dataset.
    
    Parameters
    ----------
    filepath : str
        Path to the saved dataset file (.pkl.gz).
        
    Returns
    -------
    Optional[Dataset]
        The loaded Dataset object or None if loading fails.
    """
    if not os.path.exists(filepath):
        logger.warning(f"Dataset file not found: {filepath}")
        return None
    
    try:
        dataset = Dataset.load(filepath, compressed=True)
        logger.info(
            f"Loaded deduplicated dataset with {len(dataset.ids)} studies from {filepath}"
        )
        return dataset
    except Exception as e:
        logger.error(f"Failed to load dataset from {filepath}: {e}")
        return None


def deduplicate_datasets(
    datasets: Dict[str, Dataset], 
    save_dir: Optional[str] = None
) -> Dataset:
    """
    Create a deduplicated dataset by merging all datasets and removing duplicate studies 
    based on PMID. This identifies duplicates by extracting the PMIDs from study IDs 
    (assuming format 'PMID-X') and keeping only one instance of each.
    
    Parameters
    ----------
    datasets : Dict[str, Dataset]
        Dictionary of NiMARE Dataset objects to deduplicate.
    save_dir : Optional[str], default=None
        Directory to save the deduplicated dataset (if provided).
        
    Returns
    -------
    Dataset
        A deduplicated NiMARE Dataset combining all input datasets.
    """
    if not datasets:
        logger.warning("No datasets provided for deduplication.")
        return None
    
    # If there's only one dataset, no need to deduplicate
    if len(datasets) == 1:
        return list(datasets.values())[0]
    
    dataset_list = list(datasets.values())
    merged_dataset = dataset_list[0].copy()
    
    tracked_pmids = set()
    
    # Add initial PMIDs from the first dataset
    for sid in merged_dataset.ids:
        pmid = str(sid).split("-")[0]
        tracked_pmids.add(pmid)
    
    # Merge with remaining datasets, ignoring duplicates
    for i in range(1, len(dataset_list)):
        current_dataset = dataset_list[i]
        ids_to_include = []
        
        for sid in current_dataset.ids:
            pmid = str(sid).split("-")[0]
            if pmid not in tracked_pmids:
                ids_to_include.append(sid)
                tracked_pmids.add(pmid)
        
        if ids_to_include:
            subset_dataset = current_dataset.slice(ids_to_include)
            merged_dataset = merged_dataset.merge(subset_dataset)
            logger.info(
                f"Added {len(ids_to_include)} non-duplicate studies from dataset index {i}."
            )
    
    logger.info(f"Created deduplicated dataset with {len(merged_dataset.ids)} studies.")
    
    # Save the deduplicated dataset if requested
    if save_dir and isinstance(save_dir, str):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "deduplicated_dataset.pkl.gz")
        merged_dataset.save(save_path, compress=True)
        logger.info(f"Saved deduplicated dataset to {save_path}")
    
    return merged_dataset


def _extract_study_metadata(dset: Dataset, sid: Any) -> Dict[str, Any]:
    """
    Extract study metadata (title, abstract if available) from a NiMARE Dataset.
    
    Parameters
    ----------
    dset : Dataset
        A NiMARE Dataset object.
    sid : Any
        Study ID from the dataset.
        
    Returns
    -------
    Dict[str, Any]
        A dictionary with keys "id", "title", and optionally "abstract".
    """Extract study metadata from a NiMARE dataset.

    Given a study ID, retrieve the study title and optionally the abstract
    if available.

    Parameters
    ----------
    dset : Dataset
        A NiMARE ``Dataset``.
    sid : Any
        Study identifier.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys ``"id"``, ``"source"``, ``"title"`` and
        optionally ``"abstract"``.
    """
    study_entry: Dict[str, Any] = {"id": str(sid)}
    
    title: Optional[str] = None
    try:
        titles = dset.get_metadata(ids=[sid], field="title")
        if titles and titles[0] not in (None, "", "NaN"):
            title = titles[0]
    except Exception:
        pass
    
    # Fallback: try constructing a "title" from authors/year if possible
    if not title:
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
            handle = Entrez.efetch(
                db="pubmed", id=pmid, rettype="medline", retmode="text"
            )
            records = list(Medline.parse(handle))
            if records:
                rec = records[0]
                abstract_text = rec.get("AB")
                if abstract_text:
                    study_entry["abstract"] = abstract_text.strip()
                # Use PubMed title if we don't already have one
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
        full_id = st.get("id", "")
        pmid = full_id.split("-")[0]
        
        if pmid not in unique:
            unique[pmid] = st
            
    return list(unique.values())


def get_studies_for_coordinate(
    datasets: Dict[str, Dataset],
    coord: Union[List[float], Tuple[float, float, float]],
    email: Optional[str] = None,
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
            study_ids = dset.get_studies_by_coordinate(coord_list, r=1) #TODO: make this a parameter
        except Exception as e:
            logger.warning(
                f"Failed to search coordinate {coord} in {source} dataset: {e}"
            )
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


def get_studies_for_coordinate_dedup(
    dataset: Dataset,
    coord: Union[List[float], Tuple[float, float, float]],
    source_name: str = "Combined",
    email: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Find studies that report a specific MNI coordinate in a deduplicated dataset.
    
    Parameters
    ----------
    dataset : Dataset
        A deduplicated NiMARE Dataset object.
    coord : Union[List[float], Tuple[float, float, float]]
        MNI coordinate [x, y, z].
    source_name : str, default="Combined"
        A label to indicate that the studies come from the combined dataset.
    email : Optional[str], default=None
        Email address to use with Entrez for abstract fetching (if available).
        
    Returns
    -------
    List[Dict[str, Any]]
        List of study metadata dictionaries for studies reporting this coordinate.
    """
    if not dataset:
        logger.warning("No dataset provided to search coordinates.")
        return []
    
    coord_list = [list(coord)]
    studies_info: List[Dict[str, Any]] = []
    
    try:
        study_ids = dataset.get_studies_by_coordinate(coord_list, r=0)
    except Exception as e:
        logger.warning(f"Failed to search coordinate {coord} in dataset: {e}")
        return []
    
    if not study_ids:
        return []
    
    for sid in study_ids:
        study_entry = {"id": str(sid), "source": source_name}
        if email:
            Entrez.email = email
        
        study_metadata = _extract_study_metadata(dataset, sid)
        study_entry.update(study_metadata)
        studies_info.append(study_entry)
    
    return studies_info


def generate_llm_prompt(
    studies: List[Dict[str, Any]], 
    coordinate: Union[List[float], Tuple[float, float, float]],
    prompt_type: str = "summary"
) -> str:
    """
    Generate a detailed prompt for language models based on studies found for a coordinate.
    
    This function creates a structured prompt that includes study IDs, titles, and abstracts
    formatted for optimal LLM analysis and summarization.
    
    Parameters
    ----------
    studies : List[Dict[str, Any]]
        List of study metadata dictionaries.
    coordinate : Union[List[float], Tuple[float, float, float]]
        The MNI coordinate [x, y, z] that was searched.
    prompt_type : str, default="summary"
        Type of prompt to generate ("summary", "region_name", "function", etc.).
        
    Returns
    -------
    str
        A detailed prompt for language models, incorporating all relevant study info.
    """
    if not studies:
        return (
            f"No neuroimaging studies were found reporting activation at MNI coordinate "
            f"{coordinate}."
        )
    
    coord_str = f"[{coordinate[0]}, {coordinate[1]}, {coordinate[2]}]"
    
    # Build the prompt header with clear instructions
    if prompt_type == "summary":
        prompt_intro = (
            f"You are an advanced AI with expertise in neuroanatomy and cognitive neuroscience. "
            f"The user is interested in understanding the significance of MNI coordinate {coord_str}.\n\n"
            "Below is a list of neuroimaging studies that report activation at this coordinate. "
            "Your task is to integrate and synthesize the knowledge from these studies, focusing on:\n"
            "1) The anatomical structure(s) most commonly associated with this coordinate\n"
            "2) The typical functional roles or processes linked to activation in this region\n"
            "3) The main tasks or experimental conditions in which it was reported\n"
            "4) Patterns, contradictions, or debates in the findings\n\n"
            "Do NOT simply list each study separately. Provide an integrated, cohesive summary.\n"
        )
    elif prompt_type == "region_name":
        prompt_intro = (
            f"You are a neuroanatomy expert. The user wants to identify the probable anatomical "
            f"labels for MNI coordinate {coord_str}. The following studies reported activation "
            "around this location. Incorporate anatomical knowledge and any direct references "
            "to brain regions from these studies. If multiple labels are possible, mention all "
            "and provide rationale and confidence levels.\n\n"
        )
    elif prompt_type == "function":
        prompt_intro = (
            f"You are a cognitive neuroscience expert. The user wants a deep functional profile "
            f"of the brain region(s) around MNI coordinate {coord_str}. The studies below report "
            "activation at or near this coordinate. Synthesize a clear description of:\n"
            "1) Core functions or cognitive processes\n"
            "2) Typical experimental paradigms or tasks\n"
            "3) Known functional networks or connectivity\n"
            "4) Divergent or debated viewpoints in the literature\n\n"
        )
    else:
        # Default to a basic integrated summary
        prompt_intro = (
            f"Please analyze the following neuroimaging studies reporting activation at MNI "
            f"coordinate {coord_str} and provide a concise yet thorough discussion of its "
            "anatomical location and functional significance.\n\n"
        )
    
    # Add study details
    prompt_body = "STUDIES REPORTING ACTIVATION AT MNI COORDINATE " + coord_str + ":\n"
    for i, study in enumerate(studies, 1):
        prompt_body += f"\n--- STUDY {i} ---\n"
        prompt_body += f"ID: {study.get('id', 'Unknown ID')}\n"
        prompt_body += f"Title: {study.get('title', 'No title available')}\n"
        abstract_text = study.get("abstract", "No abstract available")
        prompt_body += f"Abstract: {abstract_text}\n"
    
    # Final instructions
    prompt_outro = (
        "\nUsing ALL of the information above, produce a single cohesive synthesis. "
        "Avoid bullet-by-bullet summaries of each study. Instead, integrate the findings "
        "across them to describe the region's location, function, and context."
    )
    
    return prompt_intro + prompt_body + prompt_outro


# Example usage (as script)
if __name__ == '__main__':
    DATA_DIR = "nimare_data"
    coordinate = [30, 22, -8]
    email_address = "example@email.com"
    
    # Attempt to load deduplicated dataset
    dedup_dataset_path = os.path.join(DATA_DIR, "deduplicated_dataset.pkl.gz")
    deduplicated_dataset = None
    
    if os.path.exists(dedup_dataset_path):
        deduplicated_dataset = load_deduplicated_dataset(dedup_dataset_path)
    
    # Otherwise fetch and create one
    if not deduplicated_dataset:
        print("No deduplicated dataset found. Fetching datasets...")
        nimare_datasets = fetch_datasets(DATA_DIR)
        deduplicated_dataset = deduplicate_datasets(nimare_datasets, save_dir=DATA_DIR)
    
    # Search for studies
    if deduplicated_dataset:
        studies = get_studies_for_coordinate_dedup(
            deduplicated_dataset, 
            coordinate, 
            email=email_address
        )
        print(f"Found {len(studies)} studies for coordinate {coordinate}")
        
        # Generate a prompt
        prompt = generate_llm_prompt(studies, coordinate, prompt_type="summary")
        print("\n===================== GENERATED PROMPT =====================")
        print(prompt[:600] + "..." if len(prompt) > 600 else prompt)
        
        # Example: Using AIModelInterface to get an LLM summary (requires valid keys)
        # from .ai_model_interface import AIModelInterface  # Adjust import to your environment
        # ai = AIModelInterface(
        #     gemini_api_key="YOUR_GEMINI_KEY",
        #     openrouter_api_key="YOUR_OPENROUTER_KEY"
        # )
        # summary = ai.generate_text(model="gemini-2.0-flash", prompt=prompt)
        # print("\n===================== LLM SUMMARY =====================")
        # print(summary)

