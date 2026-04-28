---
title: 'Coord2Region: A Python Package for Mapping 3D Brain Coordinates to Atlas Labels, Literature, and AI Summaries'
tags:
  - Python
  - neuroimaging
  - brain atlas
  - meta-analysis
  - large language models
authors:
  - name: Hamza Abdelhedi
    orcid: "0000-0002-4638-640X"
    equal-contrib: true
    affiliation: "1, 2"
  - name: Yorguin-Jose Mantilla-Ramos
    orcid: "0000-0003-4473-0876"
    equal-contrib: true
    affiliation: "1, 2"
  - name: Sina Esmaeili
    orcid: "0000-0002-4740-1915"
    equal-contrib: true
    affiliation: "1"
  - name: Annalisa Pascarella
    orcid: "0000-0001-8795-0815"
    affiliation: "3"
  - name: Vanessa Hadid
    orcid: "0000-0001-6597-3805"
    affiliation: "4, 5"
  - name: Karim Jerbi
    orcid: "0000-0002-3790-9651"
    affiliation: "1, 2, 5, 6"
affiliations:
  - name: Cognitive and Computational Neuroscience Laboratory (CoCo Lab), University of Montreal, Montreal, QC, Canada
    index: 1
  - name: Mila – Quebec Artificial Intelligence Institute, Montreal, QC, Canada
    index: 2
  - name: Institute for Applied Mathematics Mauro Picone, National Research Council, 00185 Rome, Italy
    index: 3
  - name: McGill University Health Centre (MUHC), Montreal, Quebec, Canada
    index: 4
  - name: Psychology Department, University of Montreal, Montreal, QC, Canada
    index: 5
  - name: UNIQUE Center (Quebec Neuro-AI Research Center), Montreal, QC, Canada
    index: 6
date: 4 March 2026
bibliography: paper.bib
---

# Summary

Coord2Region is an open-source Python package for coordinate-based neuroimaging workflows. It maps 3D brain coordinates (e.g., MNI [@evans19933d] and Talairach [@talairach1980application]) to anatomical labels across multiple brain atlases. The package supports single-coordinate queries, high-throughput batch processing, and multi-atlas comparison. When a coordinate falls outside a labeled parcel or lies near atlas boundaries, Coord2Region uses a KD-tree–accelerated nearest-region fallback and reports distances to help users interpret uncertainty.

Coord2Region also links mapped coordinates and regions to the neuroimaging literature. Through NiMARE [@salo2022nimare], it integrates with Neurosynth [@yarkoni2011large] and NeuroQuery [@dockes2020neuroquery], enabling users to move from coordinates to atlas labels and then to relevant studies. Optional large language model (LLM) utilities can generate short summaries of linked studies and illustrative images of queried regions. These outputs are intended as exploratory aids and do not replace peer-reviewed literature or systematic review.

The package provides a Python API, command-line interface (CLI), reproducible YAML configuration files, and a web interface for configuration building and small cloud-based runs. Together, these tools reduce manual lookups, improve consistency, and make coordinate-to-region interpretation more reproducible and accessible.

# Statement of need

Neuroimaging studies commonly report results as standardized 3D coordinates. Translating these coordinates into meaningful anatomical or functional labels is a routine step in functional MRI, lesion mapping, EEG/MEG source localization, intracranial recording, and meta-analysis. However, this step is often performed manually using visualization tools or web resources, which can be slow, inconsistent, and difficult to reproduce across studies.

Interactive viewers such as MRIcron [@rorden2000stereotaxic; @MRIcron_NITRC], MRIcroGL [@rorden2025mricrogl; @MRIcroGL_NITRC], and NiiVue [@Drake2021NiiVue; @NiiVue_Docs] are valuable for visual inspection and rendering, but they are not designed primarily for automated, scriptable, high-throughput coordinate-to-label mapping. Other coordinate-labeling resources, including Talairach Software [@Lancaster2000TalairachDaemon] and label4MRI [@Chuang2022label4MRI], provide useful functionality but are limited by atlas coverage, interface constraints, or long-term maintainability. Meta-analytic resources such as Neurosynth and NeuroQuery connect text and brain maps, while NiMARE provides a modern Python framework for neuroimaging meta-analysis. Yet researchers still lack a simple “glue layer” that starts from individual coordinates, maps them across atlases, and connects the resulting labels to supporting literature in one reproducible workflow.

Coord2Region addresses this gap. It is intended for researchers, students, clinicians, and developers who need to interpret reported coordinates, annotate large coordinate tables, compare labels across atlases, or rapidly identify literature linked to a region. The package lowers the barrier for non-programmers through a CLI and web interface, while remaining flexible enough for developers to integrate into Python pipelines.

# State of the field

Coord2Region complements rather than replaces existing neuroimaging software. Visualization tools such as MRIcron, MRIcroGL, and NiiVue are strong choices for inspecting images and producing figures. Coord2Region focuses instead on automation: batch coordinate mapping, multi-atlas outputs, reproducible exports, and integration with literature search. Similarly, NiMARE, Neurosynth, and NeuroQuery provide essential meta-analytic infrastructure, but they do not provide a complete user-facing workflow for mapping arbitrary coordinate tables to atlas labels and then producing region-level study summaries.

The decision to build Coord2Region rather than only contribute to an existing package was driven by this combined workflow need. The package brings together atlas management, coordinate mapping, nearest-region fallback, study retrieval, optional AI-assisted summaries, and web/CLI execution behind a single interface. Its scholarly contribution is not a new atlas or a new meta-analytic algorithm, but a reproducible coordination layer between common neuroimaging outputs and existing interpretive resources. This is useful in research contexts where coordinates are the main reported result and where manual atlas lookup introduces avoidable inconsistency.

Another important distinction is that Coord2Region treats coordinate annotation as a reproducible data-processing step rather than as an interactive visualization task. In many studies, anatomical labels are reported in tables, supplementary materials, or downstream analyses, yet the exact lookup procedure is rarely documented in enough detail to be repeated. Coord2Region makes these choices explicit by recording the atlas, coordinate space, search radius, fallback behavior, and exported outputs. This makes it easier to compare results across atlases, re-run analyses when atlas versions change, and share the full coordinate-annotation workflow with collaborators or reviewers.

# Software design

Coord2Region uses a modular architecture designed to separate atlas handling, coordinate mapping, literature retrieval, optional AI-assisted interpretation, and export. This separation was chosen because coordinate-based neuroimaging workflows often combine heterogeneous tasks: loading atlases, mapping points to labels, resolving ambiguous coordinates, querying meta-analytic datasets, generating summaries, and saving outputs for downstream analysis. Keeping these steps modular makes the package easier to test, maintain, and extend while still allowing users to run the full workflow through a single pipeline.

The first layer handles atlas retrieval, file management, and initialization. Atlas management utilities load common volumetric and surface atlas formats, including NIfTI and NumPy-based files, resolve data directories, cache downloaded resources, and precompute information needed for fast lookup. This design allows repeated queries to reuse previously loaded atlas metadata rather than reinitializing atlas files for every coordinate. It also makes it easier to add new atlases without changing the higher-level pipeline.

The core mapping layer is organized around complementary mapper classes. `AtlasMapper` handles coordinate-to-label mapping for a single atlas. `MultiAtlasMapper` applies the same coordinate query across multiple atlases and returns per-atlas labels, supporting cross-atlas comparison. `BatchAtlasMapper` is designed for high-throughput use from many coordinates, participants or studies. This division keeps simple use cases lightweight while supporting larger analyses without requiring users to write custom batching logic.

A central design choice is the nearest-region fallback. Coordinates can fall outside labeled voxels because of atlas resolution, thresholding, subject-template mismatch, or boundary effects. Rather than silently returning no label, Coord2Region precomputes labeled voxels or vertices, builds KD-trees, assigns the nearest region when needed, and reports the corresponding distance. This design preserves usability while making uncertainty explicit. It also avoids hiding ambiguous cases: users can inspect fallback distances, set thresholds, or compare labels across atlases before deciding how to report a coordinate.

The literature layer connects anatomical mapping to functional interpretation. Through NiMARE-backed datasets, Coord2Region retrieves studies from Neurosynth and NeuroQuery using either coordinates or region labels. Retrieved records can be deduplicated and exported with available metadata such as titles, abstracts, identifiers, and sources. This layer was designed as an interpretive bridge rather than a replacement for formal meta-analysis. It lets users move quickly from an isolated coordinate to a list of potentially relevant studies while keeping the study records available for manual inspection or downstream analyses.

The optional AI layer is isolated behind a provider-agnostic `AIModelInterface`. This interface supports different LLM providers and backends while keeping core atlas mapping usable without API keys or model dependencies. Optional LLM outputs include summaries, captions, and illustrative images, with provenance information such as model name and prompt settings recorded when available. This separation is especially important because the package combines deterministic and non-deterministic components. Atlas lookup, nearest-region search, study retrieval, caching, and export are treated as inspectable computational steps, whereas LLM-based summaries and generated images are optional post-processing layers that can be disabled entirely.

A second design trade-off concerns accessibility versus extensibility. A purely graphical tool would be easier for occasional users but harder to integrate into automated analyses, while a code-only library would exclude many clinicians, students, and domain experts who need coordinate annotation without building custom scripts. Coord2Region therefore exposes the same core workflow through several interfaces: low-level Python functions, a high-level pipeline API, CLI commands, YAML configuration files, and a browser-based configuration builder. This design avoids maintaining separate logic for different user groups: each interface calls the same underlying pipeline, so outputs remain consistent across notebooks, terminal workflows, and web-based demonstrations.

The high-level pipeline chains coordinate mapping, dataset fetching, study retrieval, optional image generation, optional summarization, and export into a coherent workflow. The CLI exposes this functionality through commands such as `coords-to-study`, `coords-to-summary`, `coords-to-insights`, and `region-to-insights`. YAML configuration files make analyses easier to reproduce and share, while the JSON Schema–driven web interface helps users build valid configurations and corresponding CLI commands without manually editing configuration files. A Streamlit/Hugging Face cloud runner supports small browser-based examples, while larger analyses remain best suited to the local Python or CLI interfaces.

![Coord2Region workflow and integrations. Users provide MNI/Talairach coordinates or region names. Coordinates are mapped to standardized labels across multiple atlases; when a point falls outside a parcel or near boundaries, a KD-tree nearest-region fallback assigns the closest region and reports distances. Mapped regions are then linked to Neurosynth and NeuroQuery through NiMARE. Optional LLM utilities generate study summaries and illustrative region images. Results are exportable and runnable through a Python API, CLI, YAML configuration, or web interface.](workflow.pdf)

# Research impact statement

Coord2Region supports reproducible research workflows in coordinate-based neuroimaging by replacing manual atlas lookup with scriptable, multi-atlas, exportable analyses. Its immediate impact is practical: a researcher can process a table of activation peaks, lesion centers, source-localized EEG/MEG coordinates, or intracranial electrode locations and obtain consistent labels and linked studies with a single command or Python call. This reduces time spent on manual interpretation and makes reported coordinate-to-label decisions easier to audit.

Coord2Region is also useful as a teaching and methods-development tool. Because it exposes the same workflow through code, CLI commands, YAML files, and a web interface, users can learn how coordinate interpretation changes with atlas choice, search radius, and fallback distance. For example, a course exercise or methods tutorial can ask students to annotate the same coordinate set across several atlases and inspect where labels agree or diverge. This supports more critical use of atlas labels and helps prevent over-interpretation of a single anatomical lookup.

The package is openly developed on GitHub with documentation, examples, tests, and continuous-integration workflows. It provides reusable command-line and YAML-based execution, making it suitable for teaching, collaborative projects, and reproducible analysis pipelines. The web interface broadens access for users who need coordinate annotation but do not want to install or script a Python environment. The package is also positioned for near-term use in neuro-AI and cognitive neuroscience projects where coordinates from published studies, source localization pipelines, or model-brain comparisons need to be connected to anatomical labels and literature context.

By integrating established packages such as nibabel [@nibabel], nilearn [@Nilearn; @abraham2014machine], MNE-Python [@MNE_Python; @Gramfort2013], and NiMARE, Coord2Region builds on existing community infrastructure rather than duplicating it. Its main contribution is to make these resources easier to combine in a transparent coordinate-centric workflow.

# AI usage disclosure

Generative AI tools were used during the development of some optional Coord2Region features and during drafting/editing of parts of the paper and documentation. In the software, LLMs are not required for core coordinate-to-atlas mapping, atlas management, or study retrieval. They are used only in optional modules that generate summaries, captions, or illustrative images from user-provided regions and retrieved study metadata.

AI-assisted outputs were reviewed by the authors, and the package design keeps these outputs separate from deterministic atlas labels and bibliographic retrieval results. The authors checked generated documentation and paper text for factual accuracy, consistency with the implemented software, and appropriate framing of LLM outputs as exploratory aids rather than definitive scientific evidence. Users are encouraged to verify AI-generated summaries against the primary literature and to report model/provider provenance when using these features.

# Limitations

Coord2Region inherits limitations from the atlases and datasets it uses. Atlas choice, spatial resolution, coordinate space, and surface/volume mismatch can affect labels. Nearest-region fallback improves coverage but introduces distance-dependent uncertainty, so reported distances should be inspected and thresholded when needed. Literature retrieval depends on the coverage and metadata quality of Neurosynth, NeuroQuery, PubMed, and CrossRef. LLM-based summaries may reflect incomplete evidence, provider-specific behavior, or model drift, and should be treated as heuristic aids rather than definitive interpretations. For this reason, Coord2Region should be used to support transparent interpretation rather than to assign definitive functional meaning to a coordinate.

# Acknowledgments

We gratefully acknowledge the developers and maintainers of nilearn, nibabel, MNE-Python, NiMARE, Neurosynth, and NeuroQuery, whose open-source tools and datasets made Coord2Region possible. The authors also thank AntiCafe Loft for their support.