---
layout: default
hero_title: "Coord2Region: Coordinates <em class=\"into\">into</em> Insights"
hero_tagline: "Transform coordinates or region names into related studies, AI summaries, and AI‑generated images"
title: "About Coord2Region"
description: "About the Coord2Region project and Phase 2 goals."
---

<section class="logo-text-block" data-logo-light="{{ '/assets/img/logo.png' | relative_url }}" data-logo-dark="{{ '/assets/img/logo_darkmode.png' | relative_url }}">
  <img src="{{ '/assets/img/logo.png' | relative_url }}" alt="Coord2Region logo" class="inline-logo small theme-logo">
  <div>
    <p>Coord2Region is both a Python package and a web experience. Feed it MNI coordinates or region names and it lines up atlas labels, nearby studies, optional LLM summaries, and illustrative images. Underneath the hood you get NiMARE, Nilearn, and MNE working together without wiring every dependency yourself.</p>
  </div>
</section>

<div class="about-grid">
  <section>
    <h2>Build <span>insights</span> from coordinates</h2>
    <p>
      Coord2Region is both a Python package and a web experience. Feed it MNI coordinates or region names and it lines up atlas labels, nearby studies, optional LLM summaries, and illustrative images. Underneath the hood you get NiMARE, Nilearn, and MNE working together without wiring every dependency yourself.
    </p>
    <ul class="feature-list">
      <li><strong>Atlas lookups</strong> across Harvard–Oxford, Schaefer, Destrieux, Jülich, and more.</li>
      <li><strong>Study retrieval</strong> via Neurosynth, NeuroQuery, and NiMARE datasets.</li>
      <li><strong>AI extras</strong> for summaries and images (bring your API keys, or run purely offline).</li>
      <li><strong>Export-ready outputs</strong>: YAML + CLI commands + JSON/CSV deliverables.</li>
    </ul>
    <p>Want the full story? Browse the <a href="https://coord2region.readthedocs.io/en/latest/">Read the Docs guide</a>.</p>
  </section>

  <section>
    <h2>Why use the builder?</h2>
    <div class="card-grid">
      <article>
        <h3>No-install prototyping</h3>
        <p>Craft a complete Coord2Region run from the browser, inspect the YAML live, and copy the CLI command with zero setup.</p>
      </article>
      <article>
        <h3>Config authoring surface</h3>
        <p>Already scripted? Import a YAML, tweak parameters safely, download the updated config, or hand collaborators a clickable preset.</p>
      </article>
      <article>
        <h3>Teaching-friendly presets</h3>
        <p>Jump-start analyses with curated templates (single peak lookup, region → coords, full insights bundle) and see what changes.</p>
      </article>
      <article>
        <h3>Guardrails built in</h3>
        <p>Each toggle always updates the YAML, the CLI command, and a direct “no YAML” command so you know exactly what will execute.</p>
      </article>
    </div>
  </section>
</div>

<section class="stepper">
  <h2>Builder walkthrough</h2>
  <ol>
    <li><strong>Inputs.</strong> Paste coordinate triples or list region names. CSV upload is supported for large batches.</li>
    <li><strong>Atlas selection.</strong> Search, filter, or add custom URLs/paths. Mix volumetric and surface atlases as needed.</li>
    <li><strong>Studies & summaries.</strong> Toggle nearby study retrieval, then layer in LLM summaries with your preferred provider.</li>
    <li><strong>Images.</strong> Combine nilearn anatomical figures with AI prompts or swap in Stable Diffusion/Claude/GPT image backends.</li>
    <li><strong>Outputs.</strong> Choose the working directory, export format, and whether to emit batch-friendly CSV/JSON bundles.</li>
  </ol>
  <p>
    The right panel mirrors every change: YAML preview (copy/download), the standard YAML-driven CLI command, a direct “no YAML” command,
    and a template/import drawer for round-tripping configs.
  </p>
</section>

<section class="cloud-runner">
  <h2>Cloud Runner</h2>
  <p>
    Prefer not to run anything locally? Authenticate, submit a config, and let the hosted Cloud Runner execute the full pipeline (labels, studies, summaries, images). You can monitor jobs, stream logs, and download results (YAML/JSON/CSV/images) right from the browser.
  </p>
  <div class="card-grid">
    <article>
      <h3>Queue & monitor</h3>
      <p>Drop a YAML, pick your providers, and watch progress in real time. Cancel or rerun jobs with a click.</p>
    </article>
    <article>
      <h3>Shareable outputs</h3>
      <p>Results stick around so co-authors and students can inspect the same artefacts without juggling local environments.</p>
    </article>
    <article>
      <h3>BYO credentials</h3>
      <p>Use your own AI keys securely. Summaries and images are powered by the exact providers you choose.</p>
    </article>
  </div>
  <p class="cta">Ready to try it? Open the <a href="{{ '/cloud/' | relative_url }}">Cloud Runner</a> and submit a config.</p>
</section>
