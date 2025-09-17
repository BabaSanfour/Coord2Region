---
layout: default
hero_title: "Coord2Region Coordinates → Insights"
hero_tagline: "Translate brain coordinate datasets into reproducible atlases, summary statistics, and CLI-ready configs with a guided assistant."
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
    <p>The interactive builder below is wired to the live JSON schema that powers the CLI. Fill in coordinates manually or upload a file, choose atlases to query, and instantly preview the YAML config you can run with <code>coord2region</code>.</p>
  </div>
</section>

<section id="config-builder" class="section-card">
  <div class="card-title">
    <h2>Interactive Config Builder</h2>
    <span>Powered by JSON Schema-driven forms</span>
  </div>
  <div class="config-builder-wrapper responsive-frame">
    <div id="coord2region-root"></div>
    <aside>
      <h3>How it works</h3>
      <ul>
        <li>Toggle between manual coordinate entry and file uploads.</li>
        <li>Select one or more atlases for reverse inference lookups.</li>
        <li>Enable study review or summarization flows in one click.</li>
        <li>Copy the generated YAML or CLI command for automation.</li>
      </ul>
      <p>Feedback welcome while we build out the full Phase 2 experience.</p>
    </aside>
  </div>
</section>
