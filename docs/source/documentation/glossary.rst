.. _documentation-glossary:

Glossary
========

Keep this page handy when reading the tutorials or API docs. It collects the 
recurring concepts in **Coord2Region** and points to the sections that explain 
them in depth.

.. glossary::
   :sorted:

   Atlas Fetcher
      The utility class (:class:`coord2region.fetching.AtlasFetcher`) responsible 
      for downloading, caching, and managing atlas files from remote repositories. 
      It handles the logic for storing templates in your home directory (default: ``~/nilearn_data``).

   Atlas Mapper
      The core engine (:class:`coord2region.coord2region.AtlasMapper`) that 
      performs the mathematical translation between 3D coordinates and anatomical 
      labels. See :doc:`documentation/atlases` for the list of supported parcellations.

   Builder
      The visual configuration editor hosted on GitHub Pages. It allows you to 
      design pipelines interactively and export the YAML configuration without 
      writing code. See :doc:`documentation/web_interface`.

   Config File
      A YAML file (typically `coord2region.yaml`) that defines a reproducible 
      run. It captures inputs, enabled atlases, API keys, and output settings 
      in a single document. See :doc:`documentation/cli_tools`.

   MNI152
      The standard brain template (Montreal Neurological Institute) used by 
      Coord2Region. All input coordinates (x, y, z) are assumed to be in **MNI 
      space** unless otherwise specified.

   NiMARE
      The underlying library used to interface with neuroimaging meta-analysis 
      databases like Neurosynth and NeuroQuery.

   Pipeline
      The sequential workflow that chains distinct modules together: 
      *Coordinate* $\rightarrow$ *Atlas Lookup* $\rightarrow$ *Literature Search* $\rightarrow$ *AI Summary*.
      See :doc:`workflow` for the logic breakdown.

   Provider
      Any external service or API that enriches local data. This includes 
      **Literature Providers** (Neurosynth, NeuroQuery) and **AI Providers** (OpenAI, Gemini, Hugging Face). Configuration details live in :doc:`documentation/providers`.

   Recipe
      A specific combination of CLI flags or YAML settings designed to solve a 
      particular problem (e.g., "batch processing fMRI peaks" or "generating 
      an anatomical report").

   Result Directory
      The folder (default: ``coord2region-output/``) where the pipeline saves 
      artefacts. This includes the `results.json`, generated images, and 
      provenance logs.

   Study Radius
      A spherical distance parameter (default: 10mm) used when querying 
      literature databases. It determines how close a study's reported peak 
      must be to your query coordinate to be included in the analysis.

   Web Runner
      The client-side execution environment (hosted on Hugging Face Spaces) that 
      allows users to run the Python pipeline directly in the browser for demonstration purposes.
      distinct from the :term:`Builder`.