import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import matter from 'gray-matter';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const siteRoot = path.resolve(__dirname, '..');

const read = (relativePath) => fs.readFileSync(path.join(siteRoot, relativePath), 'utf8');

const escapeHtml = (value) =>
  String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');

const indexRaw = read('index.md');
const parsed = matter(indexRaw);
const { content, data } = parsed;

const pageTitle = data.title ?? data.hero_title ?? 'Coord2Region';
const description = data.description ?? 'Interactive configuration builder for Coord2Region.';
const heroTitle = data.hero_title ?? pageTitle;
const heroTagline = data.hero_tagline ?? description;

const html = `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>${escapeHtml(pageTitle)}</title>
    <meta name="description" content="${escapeHtml(description)}">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="/assets/css/styles.css">
  </head>
  <body class="theme">
    <header class="hero">
      <div class="hero__content container">
        <h1>${escapeHtml(heroTitle)}</h1>
        <p class="hero__tagline">${escapeHtml(heroTagline)}</p>
        <div class="hero__cta">
          <a class="btn btn-primary" href="#config-builder">Open Config Builder</a>
          <a class="btn btn-secondary" href="https://github.com/neurostuff/coord2region" target="_blank" rel="noopener">View on GitHub</a>
        </div>
      </div>
    </header>
    <main class="content container">
${content.trim()}
      <section id="config-builder" class="section-card">
        <div class="card-title">
          <h2>Interactive Config Builder (Test Preview)</h2>
          <span>Injected for automated UI tests</span>
        </div>
        <div class="responsive-frame">
          <div id="coord2region-root"></div>
        </div>
      </section>
    </main>
    <footer class="site-footer">
      <div class="container">
        <p>Coord2Region Phase 2 is currently under construction. Follow updates on the repository.</p>
        <p class="footnote">&copy; ${new Date().getFullYear()} Coord2Region contributors.</p>
      </div>
    </footer>
    <script type="module" src="/assets/js/bundle.js"></script>
  </body>
</html>`;

const previewDir = path.join(siteRoot, 'test-preview');
fs.mkdirSync(previewDir, { recursive: true });
fs.writeFileSync(path.join(previewDir, 'index.html'), html, 'utf8');
