---
layout: default
hero_title: "Coord2Region"
hero_tagline: "Turn MNI coordinates and atlas regions into actionable insights—labels, related studies, summaries, and optional images."
title: "Home"
description: "Overview of Coord2Region with links to the Builder, About, Cloud, and Docs."
---

<section class="logo-text-block" data-logo-light="{{ '/assets/img/logo.png' | relative_url }}" data-logo-dark="{{ '/assets/img/logo_darkmode.png' | relative_url }}">
  <img src="{{ '/assets/img/logo.png' | relative_url }}" alt="Coord2Region logo" class="inline-logo small theme-logo">
  <div>
    <h2>Meet Coord2Region</h2>
    <p>
      From raw neuroimaging coordinates to atlas‑level labels, study retrieval, AI summaries, and reproducible outputs. Build a configuration in the browser, then run it locally with the CLI — or start in the CLI and import the YAML here to keep everything in sync.
    </p>
  </div>
</section>

➡️ Open the Config Builder: [{{ '/builder/' | relative_url }}]({{ '/builder/' | relative_url }})

### What you can do

- Map coordinates to atlas regions and summaries
- Query related studies and generate YAML configs
- Export CLI commands for quick runs
- Optionally include AI‑generated images and nilearn anatomical slices

### Getting started

1. Install the Python package (`pip install coord2region`) and run `python scripts/configure_coord2region.py` once to capture API keys and atlas cache paths.
2. Launch the builder from this site, choose coordinates or region names, and fill out the guided form.
3. Copy the YAML or CLI command into your terminal – or download the YAML and import it later to continue editing.

### Quick links

[About]({{ '/about/' | relative_url }}) · [Config Builder]({{ '/builder/' | relative_url }}) · [Cloud Runner]({{ '/cloud/' | relative_url }}) · [Docs](https://coord2region.readthedocs.io/en/latest/) · [GitHub](https://github.com/BabaSanfour/Coord2Region)

## How it works

1. <span class="inline-logo-text">Load coordinates or select atlas regions in the Builder.</span>
2. <span class="inline-logo-text">Choose providers and output options.</span>
3. <span class="inline-logo-text">Copy the generated YAML or CLI, then run locally.</span>

Want details? See [About]({{ '/about/' | relative_url }}) or the [Docs](https://coord2region.readthedocs.io/en/latest/).

<figure class="workflow-figure">
  <img src="{{ '/assets/img/workflow.jpg' | relative_url }}" alt="Coord2Region workflow overview" class="workflow-image">
  <figcaption>From coordinates to atlas labels, studies, summaries, and optional images.</figcaption>
</figure>
