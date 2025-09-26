---
layout: default
hero_title: "Coord2Region: Coordinates <em class=\"into\">into</em> Insights"
hero_tagline: "Build a YAML config to run coord2region with minimal effort — map coordinates to region names, studies, summaries, and images."
title: "Config Builder"
description: "Interactive configuration builder for Coord2Region."
body_class: builder-page
---

<!-- markdownlint-disable MD033 -->
<section id="config-builder" class="section-card">
  <div class="card-title">
    <h2>Interactive Config Builder</h2>
    <span>Powered by JSON Schema-driven forms</span>
  </div>
  <div class="responsive-frame">
    <details class="howto howto--inframe">
      <summary>How it works</summary>
      <div class="howto__content">
        <ol>
          <li><strong>Choose input mode</strong>:
            Coordinates (paste directly or upload a CSV/TSV with <code>x,y,z</code> columns in MNI space) or Region names. If you upload a file, provide a valid path; for Region names, enter names directly.
          </li>
          <li><strong>Select atlas(es)</strong>:
            For Coordinates, you can pick multiple atlases (grouped by type and searchable). For Region names, select <em>exactly one</em> atlas.
          </li>
          <li><strong>Add sources</strong>:
            Choose study/provider sources under the <code>sources</code> field. If you plan to keep <em>Studies</em> enabled, select at least one source.
          </li>
          <li><strong>Toggle outputs</strong>:
            Studies are <em>enabled by default</em> and can be turned off. Turn on Summaries and/or Images as needed (Summaries depend on Studies).
          </li>
          <li><strong>Configure Summaries</strong> (when enabled):
            Pick a prompt type (or write a custom prompt), select one or more summary models, and optionally set a token limit. Add provider API keys if required by your chosen models.
          </li>
          <li><strong>Configure Images</strong> (when enabled):
            Choose <code>image_backend</code> (<code>ai</code>, <code>nilearn</code>, or <code>both</code>), select an <code>image_model</code>, and set a prompt type or a custom template (supports placeholders like <code>{coordinate}</code>, <code>{first_paragraph}</code>, and <code>{atlas_context}</code>).
          </li>
          <li><strong>Save & export options</strong>:
            In the Outputs mini‑section, set the working directory (used for caches and downloads), choose an optional <code>output_format</code>, and provide an <code>output_name</code> when exporting.
          </li>
          <li><strong>Review & copy</strong>:
            The YAML preview, a CLI command (that uses a saved YAML), and Direct CLI commands (no YAML needed) update live. Use the copy buttons or download the YAML.
          </li>
          <li><strong>Run</strong>:
            Either save the YAML and run the provided CLI command (e.g., <code>coord2region --config …</code>) or copy a Direct CLI command matching your current selections.
          </li>
          <li><strong>Tips</strong>:
            Reuse the same working directory to cache datasets and atlases across runs. Add provider API keys for AI features when needed.
          </li>
        </ol>
      </div>
    </details>
    <div id="coord2region-root"></div>
  </div>
</section>
<!-- markdownlint-enable MD033 -->
