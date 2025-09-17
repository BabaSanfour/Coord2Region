import { fileURLToPath } from 'node:url';
import fs from 'node:fs';
import path from 'node:path';
import { test, expect } from '@playwright/test';
import type { Page } from '@playwright/test';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '../..');

const ensureExists = (relativePath: string) => {
  const absolute = path.join(projectRoot, relativePath);
  if (!fs.existsSync(absolute)) {
    throw new Error(`Expected ${relativePath} to exist. Run npm run build && npm run generate:test-page first.`);
  }
  return absolute;
};

const assetText = (relativePath: string) => fs.readFileSync(ensureExists(relativePath), 'utf8');

const sanitizeHtmlShell = (html: string) =>
  html
    .replace(/<link rel="stylesheet" href="\/assets\/css\/styles.css">/, '')
    .replace(/<script type="module" src="\/assets\/js\/bundle.js"><\/script>/, '');

const waitForBuildArtifacts = () => {
  ensureExists('assets/js/bundle.js');
  ensureExists('assets/css/styles.css');
  ensureExists('test-preview/index.html');
};

const loadPreview = async (page: Page) => {
  const htmlShell = sanitizeHtmlShell(assetText('test-preview/index.html'));
  const cssContent = assetText('assets/css/styles.css');
  const jsBundle = assetText('assets/js/bundle.js');

  await page.setContent(htmlShell, { waitUntil: 'domcontentloaded' });
  await page.addStyleTag({ content: cssContent });
  await page.addScriptTag({ content: jsBundle, type: 'module' });
  await page.waitForSelector('#coord2region-root');
};

test.describe('Coord2Region Config Builder', () => {
  test.beforeAll(() => {
    waitForBuildArtifacts();
  });

  test('supports coordinate and file flows with YAML + CLI helpers', async ({ page }) => {
    await loadPreview(page);

    const coordinateTextarea = page.locator('#coord-textarea');
    await coordinateTextarea.fill('30, -22, 50\n12 34 56');

    const yamlOutput = page.locator('.yaml-output code');
    await expect(yamlOutput).toContainText('coordinates:');
    await expect(yamlOutput).toContainText('- 30');

    const atlasSelect = page.locator('#root_atlas_names');
    await atlasSelect.selectOption(['harvard-oxford', 'aal']);
    await expect(yamlOutput).toContainText('harvard-oxford');
    await expect(yamlOutput).toContainText('aal');
    const studyCardButton = page.locator('.card', { hasText: 'Study review' }).locator('button');
    await studyCardButton.click();
    await expect(studyCardButton).toHaveClass(/toggle--active/);

    const summaryCardButton = page.locator('.card', { hasText: 'Summaries' }).locator('button');
    await summaryCardButton.click();
    await expect(summaryCardButton).not.toHaveClass(/toggle--active/);
    await expect(yamlOutput).not.toContainText('summary_model');

    await summaryCardButton.click();
    await expect(summaryCardButton).toHaveClass(/toggle--active/);
    await expect(yamlOutput).not.toContainText('summary_model');

    const fileRadio = page.getByRole('radio', { name: 'Use coordinate file' });
    await fileRadio.click();
    const fileInput = page.locator('#coord-file');
    await fileInput.fill('/tmp/coords.tsv');
    await expect(yamlOutput).toContainText('coords_file: /tmp/coords.tsv');
    await expect(yamlOutput).not.toContainText('coordinates:');

    const copyYaml = page.getByRole('button', { name: 'Copy YAML' });
    await copyYaml.click();
    await expect(page.locator('.status--success', { hasText: 'YAML copied' })).toBeVisible();

    const copyCommand = page.getByRole('button', { name: 'Copy command' });
    await copyCommand.click();
    await expect(page.locator('.status--success', { hasText: 'Command copied' })).toBeVisible();

    const commandCode = page.locator('.cli-command');
    await expect(commandCode).toHaveText(/coord2region --config coord2region-config.yaml/);
  });
});
