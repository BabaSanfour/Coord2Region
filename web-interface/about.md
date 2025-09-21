---
layout: default
hero_title: "Coord2Region: Coordinates <em class=\"into\">into</em> Insights"
hero_tagline: "Transform coordinates or region names into related studies, AI summaries, and AI‑generated images"
title: "About Coord2Region"
description: "About the Coord2Region project and Phase 2 goals."
---

## About

Coord2Region is a Python package and web interface that turn brain coordinates or region names into actionable insights. Under the hood, it combines atlas lookups, public neuroimaging datasets, and optional AI providers to help you go from “where is this peak?” to “what does it likely mean?” in a few clicks.

With coordinates (MNI) as input, Coord2Region maps peaks to atlas labels, finds related studies (e.g., Neurosynth, NeuroQuery), summarizes evidence with LLMs, and can generate illustrative images. With region names as input, it can resolve names to coordinates and perform the same downstream analyses. Typical use cases include annotating peaks in fMRI results, localizing iEEG/MEG sources, cross‑referencing coordinates with literature, and packaging results for sharing and reproducibility.

Documentation: [Read the Docs](https://coord2region.readthedocs.io/en/latest/)

## Why the web interface?

The web interface is a no‑install way to compose and preview a complete Coord2Region run. It mirrors the Python CLI and exposes all key options as a guided builder, so you can:

- Prototype analyses quickly without writing code.
- Export the exact YAML and CLI commands you can run locally or in pipelines.
- Share frozen configurations with collaborators for reproducibility.
- Use presets to learn the tool (single peak lookup, multiple peaks with summaries, region → coords, full insights).

If you already use the Python package, the web interface doubles as a “config authoring” surface — you can create a YAML here, check that it’s valid, and run it from your terminal with the generated commands.

## Config Builder walkthrough

The builder walks you through a few focused sections. Each choice live‑updates the YAML preview, the standard CLI command (YAML‑driven), and the direct “no YAML” sub‑commands.

1. Inputs — Choose between coordinates or region names.
   - Coordinates: paste triples (x, y, z) or point to a CSV file.
   - Region names: provide one name per line (single atlas enforced for clarity).
2. Atlas — Select one or more atlases. Groups are collapsible; search helps filter names. Per‑atlas parameters are supported when needed.
3. Studies — Toggle study retrieval from open sources and configure the search radius. Great for gathering related literature around your peaks.
4. Summaries — Use LLMs to synthesize quick, human‑readable summaries. Pick models, tokens, and optional custom prompts. Summaries imply studies because they use study context.
5. Images — Optionally produce images per input using AI (prompt templates or custom) and/or nilearn for anatomical context. Choose backend and model as needed.
6. Outputs — Configure export format/name and working directory; these settings control file outputs when you run the CLI.

On the right, you’ll find:

- YAML preview — a live config you can copy or download.
- CLI command — the standard coord2region command that consumes a YAML file.
- Direct CLI command — a convenience sub‑command based on your selections; no YAML required.
- Templates & Import — load teaching presets or import a YAML you saved earlier to continue where you left off.

## Phase Two: Cloud Runner (coming soon)

Today, the builder is focused on generating valid configurations and commands you run locally. Phase Two will add an optional cloud runner: authenticate, submit your config, and let a managed job produce outputs (YAML/JSON/CSV/images) you can browse or download — no local setup required.

- Queue a run from your browser; check job status and logs.
- Persist results for sharing with co‑authors or students.
- Bring your own API keys for AI providers when summaries/images are enabled.

Until then, the web app remains the fastest way to craft correct configs and preview commands with guardrails.
