"""Utilities for mapping coordinates to neuroimaging studies.

This module fetches, converts, and queries NiMARE-compatible datasets (e.g.,
Neurosynth, NeuroQuery, and NIDM-Pain) and assembles study metadata for
coordinates of interest.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import re
import requests
from Bio import Entrez, Medline

from nimare.extract import fetch_neurosynth, fetch_neuroquery, download_nidm_pain
from nimare.io import convert_neurosynth_to_dataset
from nimare.dataset import Dataset
from nimare.utils import get_resource_path

from .utils import resolve_working_directory

logger = logging.getLogger(__name__)


def _fetch_crossref_metadata(pmid: str) -> Dict[str, Optional[str]]:
    """Fetch title and abstract from CrossRef using a PMID.

    Parameters
    ----------
    pmid : str
        PubMed identifier for the study.

    Returns
    -------
    dict[str, Optional[str]]
        Dictionary containing ``title`` and ``abstract`` keys when available.
    """
    try:
        # Crossref does not support a 'pmid' filter; many records expose PubMed
        # IDs via 'alternative-id'. Use that when possible.
        headers = {"User-Agent": "coord2region (mailto:example@example.com)"}
        url = f"https://api.crossref.org/works?filter=alternative-id:{pmid}"
        resp = requests.get(url, headers=headers, timeout=5)
        resp.raise_for_status()
        data = resp.json().get("message", {}).get("items", [])
        if not data:
            return {}
        item = data[0]
        title_list = item.get("title", [])
        title = title_list[0] if title_list else None
        abstract = item.get("abstract")
        if abstract:
            # Remove simple XML/HTML tags
            abstract = re.sub("<[^>]+>", "", abstract).strip()
        return {"title": title, "abstract": abstract}
    except requests.HTTPError as exc:  # pragma: no cover - network errors
        # Downgrade common "bad request/not found" cases to info to reduce noise
        status = getattr(exc.response, "status_code", None)
        level = logger.info if status in (400, 404) else logger.warning
        level(f"Failed to fetch CrossRef metadata for PMID {pmid}: {exc}")
        return {}
    except Exception as exc:  # pragma: no cover - network errors
        logger.warning(f"Failed to fetch CrossRef metadata for PMID {pmid}: {exc}")
        return {}


def fetch_datasets(
    data_dir: str, sources: Optional[List[str]] = None
) -> Dict[str, Dataset]:
    """Fetch and convert NiMARE datasets into ``Dataset`` objects.

    Parameters
    ----------
    data_dir : str
        Directory to store downloaded data.
    sources : Optional[List[str]], optional
        List of dataset names to download. Valid entries include ``"neurosynth"``,
        ``"neuroquery"`` and ``"nidm_pain"``. If ``None`` (default), all
        available datasets are fetched.

    Returns
    -------
    Dict[str, Dataset]
        Dictionary of NiMARE ``Dataset`` objects indexed by dataset name.
    """
    datasets: Dict[str, Dataset] = {}
    os.makedirs(data_dir, exist_ok=True)

    # Determine which sources to fetch
    if sources is None:
        requested = {"neurosynth", "neuroquery", "nidm_pain"}
    else:
        requested = {src.lower() for src in sources}

    if "neurosynth" in requested:
        try:
            ns_files = fetch_neurosynth(
                data_dir=data_dir,
                version="7",
                source="abstract",
                vocab="terms",
                overwrite=False,
            )
            ns_data = ns_files[0]
            neurosynth_dset = convert_neurosynth_to_dataset(
                coordinates_file=ns_data["coordinates"],
                metadata_file=ns_data["metadata"],
                annotations_files=ns_data.get("features"),
            )
            datasets["Neurosynth"] = neurosynth_dset
            logger.info("Neurosynth dataset loaded successfully.")
        except Exception as e:  # pragma: no cover - network errors
            logger.warning(f"Failed to fetch/convert Neurosynth dataset: {e}")

    if "neuroquery" in requested:
        try:
            nq_files = fetch_neuroquery(
                data_dir=data_dir,
                version="1",
                source="combined",
                vocab="neuroquery6308",
                type="tfidf",
                overwrite=False,
            )
            nq_data = nq_files[0]
            neuroquery_dset = convert_neurosynth_to_dataset(
                coordinates_file=nq_data["coordinates"],
                metadata_file=nq_data["metadata"],
                annotations_files=nq_data.get("features"),
            )
            datasets["NeuroQuery"] = neuroquery_dset
            logger.info("NeuroQuery dataset loaded successfully.")
        except Exception as e:  # pragma: no cover - network errors
            logger.warning(f"Failed to fetch/convert NeuroQuery dataset: {e}")

    if "nidm_pain" in requested:
        if download_nidm_pain is not None and get_resource_path is not None:
            try:
                _ = download_nidm_pain(data_dir=data_dir, overwrite=False)
                dset_file = os.path.join(get_resource_path(), "nidm_pain_dset.json")
                nidm_pain_dset = Dataset(dset_file, target="mni152_2mm", mask=None)
                datasets["NIDM-Pain"] = nidm_pain_dset
                logger.info("NIDM-Pain dataset loaded successfully.")
            except Exception as e:  # pragma: no cover - network errors
                logger.warning(f"Failed to fetch/convert NIDM-Pain dataset: {e}")
        else:  # pragma: no cover - optional dependency
            logger.warning(
                "Skipping NIDM-Pain: required NiMARE helpers are not available."
            )

    if sources is not None:
        unknown = requested - {"neurosynth", "neuroquery", "nidm_pain"}
        for src in unknown:
            logger.warning(f"Unknown dataset source '{src}' requested; skipping.")

    if not datasets:
        logger.warning("No datasets were loaded.")
    return datasets


def load_deduplicated_dataset(filepath: str) -> Optional[Dataset]:
    """Load a previously saved deduplicated dataset.

    Parameters
    ----------
    filepath : str
        Path to the saved dataset file (``.pkl.gz``).

    Returns
    -------
    Optional[Dataset]
        The loaded ``Dataset`` object, or ``None`` if loading fails.
    """
    if not os.path.exists(filepath):
        logger.warning(f"Dataset file not found: {filepath}")
        return None

    try:
        dataset = Dataset.load(filepath, compressed=True)
        logger.info(
            "Loaded deduplicated dataset with %d studies from %s",
            len(dataset.ids),
            filepath,
        )
        return dataset
    except Exception as e:
        logger.error(f"Failed to load dataset from {filepath}: {e}")
        return None


def deduplicate_datasets(
    datasets: Dict[str, Dataset], save_dir: Optional[str] = None
) -> Optional[Dataset]:
    """Create a deduplicated dataset across sources using PMIDs.

    Duplicates are identified via PubMed IDs extracted from study identifiers.
    Datasets are merged sequentially using :meth:`~nimare.dataset.Dataset.merge`,
    introduced in NiMARE 0.0.9. Older NiMARE versions without ``merge`` will
    return the first dataset unchanged.

    Parameters
    ----------
    datasets : Dict[str, Dataset]
        Dictionary of NiMARE ``Dataset`` objects to deduplicate.
    save_dir : Optional[str], default=None
        Directory to save the deduplicated dataset (if provided).

    Returns
    -------
    Optional[Dataset]
        A deduplicated NiMARE ``Dataset`` combining all inputs, or ``None`` if
        no datasets were provided.
    """
    if not datasets:
        logger.warning("No datasets provided for deduplication.")
        return None

    dataset_list = list(datasets.values())
    tracked_pmids: set[str] = set()
    merged_dataset: Optional[Dataset] = None

    for dset in dataset_list:
        ids_to_include: List[str] = []
        for sid in dset.ids:
            pmid = str(sid).split("-")[0]
            if pmid not in tracked_pmids:
                tracked_pmids.add(pmid)
                ids_to_include.append(sid)
        if not ids_to_include:
            continue
        subset = dset.slice(ids_to_include) if hasattr(dset, "slice") else dset
        if merged_dataset is None:
            merged_dataset = subset
        elif hasattr(merged_dataset, "merge"):
            merged_dataset = merged_dataset.merge(subset)
        else:  # pragma: no cover - merge method unavailable
            logger.warning("Dataset.merge not available; returning first dataset only.")
            break

    if merged_dataset is None:
        logger.warning("No studies remained after deduplication.")
        return None

    logger.info(f"Created deduplicated dataset with {len(merged_dataset.ids)} studies.")

    # Save the deduplicated dataset if requested
    if save_dir and isinstance(save_dir, str):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "deduplicated_dataset.pkl.gz")
        merged_dataset.save(save_path, compress=True)
        logger.info(f"Saved deduplicated dataset to {save_path}")

    return merged_dataset


def prepare_datasets(
    data_dir: Optional[str] = None,
    sources: Optional[List[str]] = None,
) -> Optional[Dataset]:
    """Load or create a deduplicated NiMARE dataset.

    Parameters
    ----------
    data_dir : str, optional
        Base directory for downloaded datasets and the merged cache. If ``None``
        (default) the path ``~/coord2region`` is used. Relative paths are
        interpreted relative to the user's home directory so that passing
        ``"my_cache"`` stores data in ``~/my_cache``.
    sources : Optional[List[str]], optional
        Dataset names to fetch if a cache needs to be built. See
        :func:`fetch_datasets` for valid entries. ``None`` (default) fetches all
        available datasets.

    Returns
    -------
    Optional[Dataset]
        The deduplicated NiMARE ``Dataset`` object, or ``None`` if preparation
        fails.
    """
    base_dir = resolve_working_directory(data_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = base_dir / "cached_data"
    cache_dir.mkdir(parents=True, exist_ok=True)

    dedup_path = cache_dir / "deduplicated_dataset.pkl.gz"

    if os.path.exists(dedup_path):
        dataset = load_deduplicated_dataset(str(dedup_path))
        if dataset is not None:
            return dataset

    datasets = fetch_datasets(str(base_dir), sources=sources)
    return deduplicate_datasets(datasets, save_dir=str(cache_dir))


def _extract_study_metadata(dset: Dataset, sid: Any) -> Dict[str, Any]:
    """Extract title and abstract for a study.

    Parameters
    ----------
    dset : Dataset
        NiMARE ``Dataset`` containing the study.
    sid : Any
        Study identifier.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys ``"id"``, ``"title"`` and, when available,
        ``"abstract"``. Metadata are drawn from NiMARE and supplemented with
        PubMed (via Biopython) or CrossRef when necessary.
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
            if (
                authors
                and authors[0] not in (None, "", "NaN")
                and year
                and year[0] not in (None, "", "NaN")
            ):
                title = f"{authors[0]} ({year[0]})"
        except Exception:
            title = None

    if title:
        study_entry["title"] = title

    pmid = str(sid).split("-")[0]

    # Optionally retrieve abstract via Entrez if email provided and Bio available.
    if study_entry.get("id") and "email" in study_entry:
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

    # Fallback to CrossRef to fill only missing fields
    needs_title = "title" not in study_entry or not study_entry.get("title")
    needs_abstract = "abstract" not in study_entry
    if needs_title or needs_abstract:
        crossref_data = _fetch_crossref_metadata(pmid)
        if needs_title and crossref_data.get("title"):
            study_entry["title"] = crossref_data["title"]
        if needs_abstract and crossref_data.get("abstract"):
            study_entry["abstract"] = crossref_data["abstract"]

    return study_entry


def remove_duplicate_studies(studies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate studies based on PMID.

    Parameters
    ----------
    studies : List[Dict[str, Any]]
        Study entries containing an ``"id"`` field with a ``PMID-`` prefix.

    Returns
    -------
    List[Dict[str, Any]]
        Unique study entries keyed by PMID.
    """
    unique: Dict[str, Dict[str, Any]] = {}
    for st in studies:
        full_id = st.get("id", "")
        pmid = full_id.split("-")[0]

        if pmid not in unique:
            unique[pmid] = st

    return list(unique.values())


def search_studies(
    datasets: Dict[str, Dataset],
    coord: Union[List[float], Tuple[float, float, float]],
    radius: float = 0,
    sources: Optional[List[str]] = None,
    email: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search selected datasets for studies reporting an MNI coordinate.

    Parameters
    ----------
    datasets : Dict[str, Dataset]
        NiMARE ``Dataset`` objects keyed by source name.
    coord : Union[List[float], Tuple[float, float, float]]
        MNI coordinate ``[x, y, z]``.
    radius : float, default=0
        Search radius in millimeters around the coordinate. ``0`` indicates an
        exact match.
    sources : Optional[List[str]], optional
        Specific dataset names to query. ``None`` searches across all provided
        datasets.
    email : Optional[str], optional
        Email address for Entrez (if abstract fetching is enabled).

    Returns
    -------
    List[Dict[str, Any]]
        Deduplicated list of study metadata dictionaries.
    """
    coord_list = [list(coord)]
    studies_info: List[Dict[str, Any]] = []

    if sources is None:
        selected = datasets.items()
    else:
        selected = [(src, datasets[src]) for src in sources if src in datasets]

    for source, dset in selected:
        try:
            study_ids = dset.get_studies_by_coordinate(coord_list, r=radius)
        except Exception as e:  # pragma: no cover - underlying library errors
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

    return remove_duplicate_studies(studies_info)


def get_studies_for_coordinate(
    datasets: Union[Dict[str, Dataset], Dataset],
    coord: Union[List[float], Tuple[float, float, float]],
    radius: float = 0,
    email: Optional[str] = None,
    sources: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Find studies reporting an MNI coordinate across all datasets.

    This is a thin wrapper around :func:`search_studies` that searches every
    dataset in ``datasets``. When a single deduplicated dataset is supplied, it
    is treated as a combined source.

    Parameters
    ----------
    datasets : Union[Dict[str, Dataset], Dataset]
        NiMARE ``Dataset`` objects keyed by source name, or a single
        deduplicated ``Dataset`` instance.
    coord : Union[List[float], Tuple[float, float, float]]
        MNI coordinate ``[x, y, z]``.
    radius : float, default=0
        Search radius in millimeters around the coordinate. ``0`` indicates an
        exact match.
    email : Optional[str], optional
        Email address for Entrez (if abstract fetching is enabled).
    sources : Optional[List[str]], optional
        Restrict the search to the specified dataset names when ``datasets`` is
        a mapping.
    """
    dataset_map: Dict[str, Dataset]
    if isinstance(datasets, dict):
        dataset_map = datasets
    else:
        dataset_map = {"Combined": datasets}

    return search_studies(
        dataset_map, coord, radius=radius, email=email, sources=sources
    )
