"""
Coordinate to Study Mapping Module

This module provides functionality to map MNI coordinates to relevant neuroimaging studies
by leveraging NiMARE datasets (Neurosynth, NeuroQuery). It handles dataset fetching,
deduplication of studies across datasets, and extraction of study metadata.
"""

import os
import sys
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import nimare
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

# Check for Biopython availability (needed for abstract fetching)
try:
    from Bio import Entrez, Medline
    BIO_AVAILABLE = True
except ImportError:
    BIO_AVAILABLE = False
    logger.warning("Biopython not found. Abstract fetching will be disabled.")


# Dataset Handling Functions
def fetch_datasets(data_dir: str) -> Dict[str, Dataset]:
    """
    Fetch and convert Neurosynth and NeuroQuery datasets into NiMARE Dataset objects.
    
    Parameters
    ----------
    data_dir : str
        Directory to store downloaded data.
        
    Returns
    -------
    Dict[str, Dataset]
        Dictionary of NiMARE Dataset objects indexed by dataset name.        
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
        logger.info(f"Loaded deduplicated dataset with {len(dataset.ids)} studies from {filepath}")
        return dataset
    except Exception as e:
        logger.error(f"Failed to load dataset from {filepath}: {e}")
        return None


def deduplicate_datasets(datasets: Dict[str, Dataset], save_dir: Optional[str] = None) -> Dataset:
    """
    Create a deduplicated dataset by merging all datasets and removing duplicate studies based on PMID.
    
    This function identifies duplicates by extracting the PMIDs from study IDs (assuming format 'PMID-X')
    and keeping only one instance of each study.
    
    Parameters
    ----------
    datasets : Dict[str, Dataset]
        Dictionary of NiMARE Dataset objects to deduplicate.
    save_dir : Optional[str], default=None
        Directory to save the deduplicated dataset. If None, the dataset is not saved.
        
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
    
    # Get all dataset values as a list
    dataset_list = list(datasets.values())
    
    # Start with the first dataset
    merged_dataset = dataset_list[0].copy()
    
    # Track PMIDs to avoid duplicates
    tracked_pmids = set()
    
    # Add initial PMIDs from the first dataset
    for sid in merged_dataset.ids:
        pmid = str(sid).split("-")[0]
        tracked_pmids.add(pmid)
    
    # Merge with remaining datasets, avoiding duplicates
    for i in range(1, len(dataset_list)):
        current_dataset = dataset_list[i]
        # Get IDs to include (non-duplicates)
        ids_to_include = []
        
        for sid in current_dataset.ids:
            pmid = str(sid).split("-")[0]
            if pmid not in tracked_pmids:
                ids_to_include.append(sid)
                tracked_pmids.add(pmid)
        
        # If there are new studies to add, slice the dataset and merge
        if ids_to_include:
            subset_dataset = current_dataset.slice(ids_to_include)
            merged_dataset = merged_dataset.merge(subset_dataset)
            logger.info(f"Added {len(ids_to_include)} non-duplicate studies from dataset {i+1}.")
    
    logger.info(f"Created deduplicated dataset with {len(merged_dataset.ids)} studies.")
    
    # Save the deduplicated dataset if requested
    if save_dir and isinstance(save_dir, str):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "deduplicated_dataset.pkl.gz")
        merged_dataset.save(save_path, compress=True)
        logger.info(f"Saved deduplicated dataset to {save_path}")
    
    return merged_dataset


# Study Metadata Functions
def _extract_study_metadata(dset: Dataset, sid: Any) -> Dict[str, Any]:
    """
    Extract study metadata (title and abstract if available) from a NiMARE Dataset.
    
    Parameters
    ----------
    dset : Dataset
        A NiMARE Dataset.
    sid : Any
        Study ID.
        
    Returns
    -------
    Dict[str, Any]
        A dictionary with keys "id", "source", "title", and optionally "abstract".
    """
    study_entry: Dict[str, Any] = {"id": str(sid)}
    
    # Try to extract title from dataset
    title: Optional[str] = None
    try:
        titles = dset.get_metadata(ids=[sid], field="title")
        if titles and titles[0] not in (None, "", "NaN"):
            title = titles[0]
    except Exception:
        # Fallback: try to construct a title from authors and year
        try:
            authors = dset.get_metadata(ids=[sid], field="authors")
            year = dset.get_metadata(ids=[sid], field="year")
            if authors and year:
                title = f"{authors[0]} ({year[0]})"
        except Exception:
            title = None
            
    if title:
        study_entry["title"] = title

    # Optional: retrieve abstract using PubMed via Entrez if email provided and Bio is available
    if BIO_AVAILABLE and study_entry.get("id"):
        pmid = str(sid).split("-")[0]  
        try:
            handle = Entrez.efetch(db="pubmed", id=pmid, rettype="medline", retmode="text")
            records = list(Medline.parse(handle))
            if records:
                rec = records[0]
                # Get abstract if available
                abstract_text = rec.get("AB")
                if abstract_text:
                    study_entry["abstract"] = abstract_text.strip()
                # Use PubMed title as fallback if we don't have one
                if "title" not in study_entry:
                    pub_title = rec.get("TI")
                    if pub_title:
                        study_entry["title"] = pub_title.strip()
        except Exception as e:
            logger.warning(f"Failed to fetch abstract for PMID {pmid}: {e}")
            
    return study_entry


def remove_duplicate_studies(studies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate studies from a list based on PMID.
    
    Parameters
    ----------
    studies : List[Dict[str, Any]]
        List of study metadata dictionaries.
        
    Returns
    -------
    List[Dict[str, Any]]
        Deduplicated list of study metadata.
    """
    unique = {}
    for st in studies:
        # If IDs are like '24984958-1', split at the dash to get PMID
        full_id = st.get("id", "")
        pmid = full_id.split("-")[0]
        
        # Use PMID as key to identify duplicates across sources
        if pmid not in unique:
            unique[pmid] = st
            
    return list(unique.values())


# Coordinate Search Functions
def get_studies_for_coordinate(
    datasets: Dict[str, Dataset],
    coord: Union[List[float], Tuple[float, float, float]],
    email: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Find studies that report a specific MNI coordinate across multiple datasets.
    
    Parameters
    ----------
    datasets : Dict[str, Dataset]
        Dictionary of NiMARE Dataset objects keyed by source name.
    coord : Union[List[float], Tuple[float, float, float]]
        MNI coordinate [x, y, z].
    email : Optional[str], default=None
        Email address to use with Entrez for abstract fetching (if available).
        
    Returns
    -------
    List[Dict[str, Any]]
        List of study metadata dictionaries for studies reporting this coordinate.
    """
    # NiMARE expects a list of coordinates
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


def get_studies_for_coordinate_dedup(
    dataset: Dataset,
    coord: Union[List[float], Tuple[float, float, float]],
    source_name: str = "Combined",
    email: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Find studies that report a specific MNI coordinate in a deduplicated dataset.
    
    This function is optimized to work with a single deduplicated dataset, avoiding 
    the need to check for duplicates across multiple datasets.
    
    Parameters
    ----------
    dataset : Dataset
        A deduplicated NiMARE Dataset object.
    coord : Union[List[float], Tuple[float, float, float]]
        MNI coordinate [x, y, z].
    source_name : str, default="Combined"
        Name to use as the source for the studies.
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
    
    # NiMARE expects a list of coordinates
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
            study_entry["email"] = email  
        
        study_metadata = _extract_study_metadata(dataset, sid)
        study_entry.update(study_metadata)
        studies_info.append(study_entry)
    
    # No need to remove duplicates as the dataset is already deduplicated
    return studies_info


# Prompt Generation Functions
def generate_llm_prompt(
    studies: List[Dict[str, Any]], 
    coordinate: Union[List[float], Tuple[float, float, float]],
    prompt_type: str = "summary"
) -> str:
    """
    Generate a detailed prompt for language models based on studies found for a coordinate.
    
    This function creates a structured prompt that includes all study data (IDs, titles, abstracts)
    formatted for optimal language model analysis and summarization.
    
    Parameters
    ----------
    studies : List[Dict[str, Any]]
        List of study metadata dictionaries from get_studies_for_coordinate functions
    coordinate : Union[List[float], Tuple[float, float, float]]
        The MNI coordinate [x, y, z] that was searched
    prompt_type : str, default="summary"
        Type of prompt to generate ("summary", "region_name", "function", etc.)
        
    Returns
    -------
    str
        A detailed prompt for language models including all study information
    """
    if not studies:
        return f"No neuroimaging studies were found reporting activation at MNI coordinate {coordinate}."
    
    # Format the coordinate as a string
    coord_str = f"[{coordinate[0]}, {coordinate[1]}, {coordinate[2]}]"
    
    # Build the prompt header with clear instructions
    if prompt_type == "summary":
        prompt = f"""
You are a neuroscience expert assistant. I need a comprehensive yet concise summary of the brain region at MNI coordinate {coord_str}.

Below are {len(studies)} neuroimaging studies that report activation at this coordinate.
Analyze these studies and provide:
1. The most likely brain region name(s) at this coordinate
2. The functional role of this region based on the studies
3. A synthesis of the key findings across studies
4. Any patterns or common themes in the experimental tasks/conditions

Do not summarize each study separately. Instead, integrate information across studies to provide a cohesive picture of this brain region.

STUDIES REPORTING ACTIVATION AT MNI COORDINATE {coord_str}:
"""
    elif prompt_type == "region_name":
        prompt = f"""
You are a neuroanatomy expert. Based on the following neuroimaging studies reporting activation at MNI coordinate {coord_str}, 
determine the most likely anatomical name(s) of this brain region. Consider all possible anatomical labels at different scales 
(gyrus, sulcus, specific nucleus, etc.) and provide confidence levels for each possible label.

STUDIES REPORTING ACTIVATION AT MNI COORDINATE {coord_str}:
"""
    elif prompt_type == "function":
        prompt = f"""
You are a cognitive neuroscience expert. Based on the following neuroimaging studies reporting activation at MNI coordinate {coord_str},
provide a detailed functional profile of this brain region. Describe:
1. Primary cognitive functions associated with this region
2. Secondary or context-dependent functions
3. Functional connectivity patterns if mentioned
4. How this region fits into larger brain networks
5. Any contradictions or debates about this region's function

STUDIES REPORTING ACTIVATION AT MNI COORDINATE {coord_str}:
"""
    else:
        # Default to basic summary prompt
        prompt = f"""
Please analyze the following neuroimaging studies reporting activation at MNI coordinate {coord_str} and 
provide a comprehensive summary of this brain region's anatomical location and functional role.

STUDIES REPORTING ACTIVATION AT MNI COORDINATE {coord_str}:
"""
    
    # Add each study's information to the prompt
    for i, study in enumerate(studies, 1):
        prompt += f"\n\n--- STUDY {i} ---"
        prompt += f"\nID: {study.get('id', 'Unknown ID')}"
        prompt += f"\nTitle: {study.get('title', 'No title available')}"
        
        if 'abstract' in study and study['abstract']:
            prompt += f"\nAbstract: {study['abstract']}"
        else:
            prompt += "\nAbstract: Not available"
    
    # Add final instructions
    prompt += f"""

Based on ALL the studies above, provide a comprehensive analysis of the brain region at MNI coordinate {coord_str}.
Be specific about anatomical labels and functional roles. Use evidence from the studies to support your conclusions.
Do not list or summarize individual studies; instead, synthesize across them to provide an integrated view of this brain region.
"""
    
    return prompt


# Example usage
if __name__ == '__main__':
    DATA_DIR = "nimare_data"
    
    # Example coordinate in MNI space [x, y, z]
    coordinate = [30, 22, -8]
    email_address = "snesmaeil@gmail.com"  # Replace with your email
    
    # Option 1: Try to load existing deduplicated dataset
    dedup_dataset_path = os.path.join(DATA_DIR, "deduplicated_dataset.pkl.gz")
    deduplicated_dataset = None
    
    if os.path.exists(dedup_dataset_path):
        deduplicated_dataset = load_deduplicated_dataset(dedup_dataset_path)
    
    # Option 2: If no deduplicated dataset exists, fetch and create one
    if not deduplicated_dataset:
        print("No deduplicated dataset found. Fetching datasets...")
        nimare_datasets = fetch_datasets(DATA_DIR)
        deduplicated_dataset = deduplicate_datasets(nimare_datasets, save_dir=DATA_DIR)
    
    # Search for studies using the deduplicated dataset
    if deduplicated_dataset:
        studies = get_studies_for_coordinate_dedup(
            deduplicated_dataset, 
            coordinate, 
            source_name="Combined", 
            email=email_address
        )
        
        print(f"Found {len(studies)} studies for coordinate {coordinate}")
        
        # # Print study information
        # for i, study in enumerate(studies, 1):
        #     print(f"\nStudy {i}:")
        #     print(f"  ID: {study.get('id')}")
        #     print(f"  Title: {study.get('title', 'N/A')}")
        #     if 'abstract' in study:
        #         abstract = study['abstract']
        #         # Truncate long abstracts for display
        #         if len(abstract) > 200:
        #             abstract = abstract[:200] + "..."
        #         print(f"  Abstract: {abstract}")
        
        # Generate a prompt for LLM about this coordinate
        prompt = generate_llm_prompt(studies, coordinate, prompt_type="summary")
        print("\n" + "="*80)
        print("GENERATED PROMPT FOR LANGUAGE MODEL:")
        print("="*80)
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        print("="*80)
        
        # Example of using the prompt with AIModelInterface
        try:
            # Use conditional import approach to handle both direct script execution and package import
            try:
                # Try relative import first (when used as part of package)
                from .ai_model_interface import AIModelInterface
            except ImportError:
                # Fall back to direct import (when script is run directly)
                try:
                    from ai_model_interface import AIModelInterface
                except ImportError:
                    # If in the same directory but not installed as package
                    import sys
                    import os
                    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    from coord2region.ai_model_interface import AIModelInterface
            
            print("\nAttempting to generate summary using AIModelInterface...")
            # Initialize the AI interface with your API keys
            ai = AIModelInterface(
                gemini_api_key="AIzaSyBXQFQ4PbB29BteSFs1zDq5dD8o8YkbKxg",
                openrouter_api_key="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"
            )
            
            # Generate a summary using one of the models (adjust as needed)
            summary = ai.generate_text(
                model="gemini-2.0-flash",  # Using a more capable model for complex reasoning
                prompt=prompt
            )
            
            print("\n" + "="*80)
            print("AI-GENERATED REGION SUMMARY:")
            print("="*80)
            print(summary)
            
        except (ImportError, Exception) as e:
            print(f"\nCouldn't generate AI summary: {e}")
            print("You can copy the prompt above and use it with any AI model of your choice.")
            
