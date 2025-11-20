# Coord2Region Web Interface

The `web-interface/` package hosts the Jekyll landing page and React based
configuration builder for the Coord2Region project. It provides a schema-driven
form that maps directly onto `Coord2RegionConfig` and produces shareable YAML
configs plus ready-to-run CLI commands.

## Prerequisites

- Node.js 18 or newer (the Vite toolchain targets modern ESM features)
- `npm` for dependency management (Yarn/PNPM will work, but the scripts assume npm)
- Ruby 3.1+ with Bundler (required for Jekyll previews)

Install all front-end dependencies from the `web-interface/` directory:

```bash
cd web-interface
npm install
bundle install
```

## Local development

- Start the live development server (uses Vite + React fast refresh):

  ```bash
  npm run dev
  ```

  The dev server prints the local URL (default `http://localhost:5173`). Hot
  module reloading is enabled for both the React bundle and CSS changes.

- The Jekyll layout consumes the compiled bundle at `assets/js/bundle.js`. When
  you need the static assets (for example, before running Jekyll or Playwright
  in CI) run:

  ```bash
  npm run build
  ```

  This writes deterministic assets into `web-interface/assets/js/` without
  clobbering authored CSS.

- To preview the Jekyll shell without Ruby, generate a static HTML page that
  stitches together the layout, head include, and landing content:

  ```bash
  npm run generate:test-page
  npm run preview:test   # serves http://127.0.0.1:4173/test-preview/
  ```

  The preview server is also what the Playwright tests use.

## Automated UI testing

The Playwright suite exercises the key interactions (coordinate/file toggles,
YAML preview, clipboard helpers) against the generated preview:

```bash
npm run test:ui
```

Playwright downloads the required browser binaries on first run. If you need to
refresh them manually, run `npx playwright install` inside `web-interface/`.

## Deployment

The site is published automatically to https://babasanfour.github.io/Coord2Region/
whenever the `Website` GitHub Actions workflow runs on `main`. The job generates
`docs/static/schema.json`, installs the web stack, runs the Playwright UI suite,
builds the Jekyll site, and uploads the resulting `_site/` directory as the Pages
artifact. Pull requests get the same build and test checks without deploying.

## Full-site preview

To preview the production site locally, keep both the Vite dev server and the
Jekyll renderer running from `web-interface/`:

1. Run `npm run build -- --watch` once to seed the compiled assets (or rerun
   `npm run build` after each change if you skip watch mode).
2. In one terminal, start the React builder dev server with `npm run dev`
   (served at http://localhost:5173).
3. In another terminal, serve the landing page with `bundle exec jekyll serve
   --livereload` and open http://127.0.0.1:4000 to view the combined site.

Jekyll consumes the files in `web-interface/assets`, so be sure to rerun the
build step whenever you change the builder if watch mode is not running.

### Custom atlas sources

The atlas picker accepts canonical atlas names as well as direct URLs or local
file paths (e.g. `/data/custom_atlas.nii.gz`). Entering a URL/path automatically
adds the fetch information to the generated YAML via the `atlas_configs` block.

When using the CLI directly you can supply the same values with repeated
`--atlas` flags. For advanced cases where you want to keep a friendly alias,
the CLI also provides helper flags:

```bash
coord2region coords-to-atlas 30 -22 50 \
  --atlas-url custom=https://example.org/custom_atlas.nii.gz \
  --atlas custom

coord2region coords-to-atlas 30 -22 50 \
  --atlas-file local=/path/to/custom_atlas.nii.gz \
  --atlas local
```

## Project layout

- `_config.yml`, `_layouts/`, `_includes/` – Jekyll scaffold and landing page
- `assets/css/` – authored global styles for the landing page
- `website/src/` – Vite + React application (entry point: `ConfigBuilder.tsx`)
- `scripts/` – helper scripts for preview generation and static serving
- `tests/ui/` – Playwright coverage for the config builder UX

For more information on project-wide contribution practices, see
[`CONTRIBUTING.md`](../CONTRIBUTING.md).
