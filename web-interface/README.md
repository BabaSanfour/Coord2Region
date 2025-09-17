# Coord2Region Web Interface

The `web-interface/` package hosts the Jekyll landing page and React based
configuration builder for the Coord2Region project. It provides a schema-driven
form that maps directly onto `Coord2RegionConfig` and produces shareable YAML
configs plus ready-to-run CLI commands.

## Prerequisites

- Node.js 18 or newer (the Vite toolchain targets modern ESM features)
- `npm` for dependency management (Yarn/PNPM will work, but the scripts assume npm)

Install all front-end dependencies from the `web-interface/` directory:

```bash
cd web-interface
npm install
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

## Project layout

- `_config.yml`, `_layouts/`, `_includes/` – Jekyll scaffold and landing page
- `assets/css/` – authored global styles for the landing page
- `website/src/` – Vite + React application (entry point: `ConfigBuilder.tsx`)
- `scripts/` – helper scripts for preview generation and static serving
- `tests/ui/` – Playwright coverage for the config builder UX

For more information on project-wide contribution practices, see
[`CONTRIBUTING.md`](../CONTRIBUTING.md).
