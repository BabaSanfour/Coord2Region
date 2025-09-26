import { fileURLToPath } from 'node:url';
import fs from 'node:fs';
import path from 'node:path';
import { test, expect } from '@playwright/test';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '../..');

const ensureExists = (relativePath: string) => {
  const absolute = path.join(projectRoot, relativePath);
  if (!fs.existsSync(absolute)) {
    throw new Error(`Expected ${relativePath} to exist. Run npm run build first.`);
  }
  return absolute;
};

const assetText = (relativePath: string) => fs.readFileSync(ensureExists(relativePath), 'utf8');

const sanitizeHtmlShell = (html: string) =>
  html
    .replace(/<link rel="stylesheet" href="\/assets\/css\/styles.css">/, '')
    .replace(/<script type="module" src="\/assets\/js\/bundle.js"><\/script>/, '');

const loadPreview = async (page: import('@playwright/test').Page) => {
  // Ensure built assets and preview shell exist (repository includes test-preview/index.html)
  ensureExists('assets/js/bundle.js');
  ensureExists('assets/css/styles.css');
  ensureExists('test-preview/index.html');

  const htmlShell = sanitizeHtmlShell(assetText('test-preview/index.html'));
  const cssContent = assetText('assets/css/styles.css');
  const jsBundle = assetText('assets/js/bundle.js');

  await page.setContent(htmlShell, { waitUntil: 'domcontentloaded' });
  await page.addStyleTag({ content: cssContent });
  await page.addScriptTag({ content: jsBundle, type: 'module' });
  await page.waitForSelector('#coord2region-root');
};

test.describe('Coord2Region Builder Smoke', () => {
  test('renders builder and key UI elements', async ({ page }) => {
    await loadPreview(page);

    // Config root mounts
    await expect(page.locator('#coord2region-root')).toBeVisible();

  // Input mode toggles visible (use role for disambiguation)
  await expect(page.getByRole('radio', { name: 'Coordinates', exact: true })).toBeVisible();
  await expect(page.getByRole('radio', { name: 'Region names', exact: true })).toBeVisible();

  // YAML preview / CLI blocks are present (use structural selectors)
  await expect(page.locator('.yaml-output')).toBeVisible();
  await expect(page.locator('.cli-command')).toBeVisible();
  await expect(page.locator('.direct-cli')).toBeVisible();
  });
});
