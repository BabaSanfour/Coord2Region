---
layout: default
hero_title: "Coord2Region: Coordinates <em class=\"into\">into</em> Insights"
hero_tagline: "Transform brain coordinates into region names, related studies, AI summaries, and AI‑generated images — with optional region‑based workflows too."
title: "Coord2Region Config Builder"
description: "Landing page and interactive configuration builder for the Coord2Region project."
---

<section class="section-card">
  <div class="status-pill">Phase 2 · Under construction</div>
  <h2>Meet Coord2Region</h2>
  <p>Coord2Region streamlines the path from raw neuroimaging coordinates to actionable atlas-level insights. Phase 2 is focused on building a collaborative research companion that makes configuration, summarization, and data review effortless.</p>
</section>

<section class="section-card section-grid">
  <div>
    <h3>Phase 2 focus</h3>
    <p>We are adding tooling for richer atlas selection, customizable summarization flows, and better dataset provenance tracking. The new web interface will guide you through the available options and generate a shareable configuration for the CLI.</p>
  </div>
  <div>
    <h3>Try the prototype</h3>
  <p>Open the <a href="{{ '/builder/' | relative_url }}">Config Builder</a> to generate a YAML config you can run with <code>coord2region</code>.</p>
  </div>
</section>

<section class="section-card">
  <h2>About</h2>
  <p>
    Coord2Region turns MNI coordinates and atlas region names into actionable insights. It maps coordinates to atlas labels,
    finds related studies from open neuro datasets, generates concise AI‑powered summaries, and optionally produces
    reproducible images.
  </p>
  <p>
  Documentation: <a href="https://coord2region.readthedocs.io/en/latest/" target="_blank" rel="noopener">Read the Docs</a>.
  </p>
  <p>
  Ready to experiment? Visit the <a href="{{ '/builder/' | relative_url }}">Config Builder</a> or learn about our plans for the <a href="{{ '/cloud/' | relative_url }}">Cloud Runner</a>.
  </p>
</section>
