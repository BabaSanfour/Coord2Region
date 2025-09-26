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

    // No residual RJSF root title/description or duplicated blocks
    await expect(page.getByText('Coord2RegionConfig').first()).toHaveCount(0);
    await expect(page.getByText('Pydantic model capturing all CLI-facing configuration options.').first()).toHaveCount(0);
  // The app has a custom Outputs mini-section; ensure the default RJSF Outputs field is not rendered.
  // We check there is no default RJSF form group labeled 'Outputs Type: array'.
  const defaultOutputsGroup = page.locator('text=Outputs Type: array');
  await expect(defaultOutputsGroup).toHaveCount(0);
  // No default RJSF number field for study_search_radius (we render a custom one elsewhere)
  await expect(page.locator('text=Study Search Radius Type: number')).toHaveCount(0);

    const coordinateTextarea = page.locator('#coord-textarea');
    await coordinateTextarea.fill('30, -22, 50\n12 34 56');

    const yamlOutput = page.locator('.yaml-output code');
    await expect(yamlOutput).toContainText('coordinates:');
    await expect(yamlOutput).toContainText('- 30');

    // New atlas selection UI lives under .atlas-section with friendly labels
    const atlasSection = page.locator('.atlas-section');

    const checkboxForLabel = async (labelText: string) => {
      const item = atlasSection.locator('.atlas-select__item', { hasText: labelText });
      await item.waitFor();
      return item.locator('input[type="checkbox"]');
    };

    // Toggle Jülich off if selected, ensure Harvard–Oxford on, then add AAL
    const juelichCheckbox = await checkboxForLabel('Jülich');
    if (await juelichCheckbox.isChecked()) {
      await juelichCheckbox.uncheck();
    }

    const harvardCheckbox = await checkboxForLabel('Harvard–Oxford');
    if (!(await harvardCheckbox.isChecked())) {
      await harvardCheckbox.check();
    }

    const aalCheckbox = await checkboxForLabel('AAL');
    await aalCheckbox.check();
    await expect(yamlOutput).toContainText('harvard-oxford');
    await expect(yamlOutput).toContainText('aal');

    // Select all in Volumetric group and verify the group meta shows 10/10
    const volumetricGroup = atlasSection.locator('.atlas-select__group', { hasText: 'Volumetric (nilearn)' });
    const selectAllVolumetric = volumetricGroup.locator('.atlas-select__toggle');
    await selectAllVolumetric.click();
    await expect(selectAllVolumetric).toHaveText(/Clear all/);
    await expect(volumetricGroup.locator('.atlas-select__meta')).toHaveText('10/10');

    // Switch to Studies tab and enable Studies via the new switch UI
    const studiesPill = page.getByRole('button', { name: 'Studies' });
    await studiesPill.click();
    const studySwitch = page.locator('.card.card--inline', { has: page.getByRole('heading', { name: 'Studies' }) }).locator('.switch');
    await studySwitch.waitFor();
    let studiesPressed = await studySwitch.getAttribute('aria-pressed');
    if (studiesPressed !== 'true') {
      await studySwitch.click();
    }
    await expect(studySwitch).toHaveAttribute('aria-pressed', 'true');

    // Show Summaries tab to reveal its switch
    const summariesPill = page.getByRole('button', { name: 'Summaries' });
    await summariesPill.click();
    const summarySwitch = page
        .locator('.card.card--inline', {
          has: page.getByRole('heading', { name: 'Summaries' })
        })
        .locator('.switch');
    await summarySwitch.waitFor();
    // Toggle summaries off then back on, asserting aria-pressed
    let summariesPressed = await summarySwitch.getAttribute('aria-pressed');
    if (summariesPressed === 'true') {
      await summarySwitch.click();
    }
    await expect(summarySwitch).toHaveAttribute('aria-pressed', 'false');
    await expect(yamlOutput).not.toContainText('summary_model');

    await summarySwitch.click();
    await expect(summarySwitch).toHaveAttribute('aria-pressed', 'true');
    await expect(yamlOutput).not.toContainText('summary_model');
    await expect(yamlOutput).toContainText('prompt_type: summary');

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

  test('supports multiple summary model selection', async ({ page }) => {
    await loadPreview(page);

    // Enable studies first (summaries depend on studies)
    const studiesPill = page.getByRole('button', { name: 'Studies' });
    await studiesPill.click();
    const studySwitch = page.locator('.card.card--inline', { has: page.getByRole('heading', { name: 'Studies' }) }).locator('.switch');
    await studySwitch.waitFor();
    let studiesPressed = await studySwitch.getAttribute('aria-pressed');
    if (studiesPressed !== 'true') {
      await studySwitch.click();
    }
    await expect(studySwitch).toHaveAttribute('aria-pressed', 'true');

    // Enable summaries
    // Reveal Summaries tab
    const summariesPill2 = page.getByRole('button', { name: 'Summaries' });
    await summariesPill2.click();
    const summarySwitch = page
        .locator('.card.card--inline', {
          has: page.getByRole('heading', { name: 'Summaries' })
        })
        .locator('.switch');
    await summarySwitch.waitFor();
    // Ensure it's enabled
    let summariesPressed2 = await summarySwitch.getAttribute('aria-pressed');
    if (summariesPressed2 !== 'true') {
      await summarySwitch.click();
    } else {
      // toggle off and back on to ensure UI updates
      await summarySwitch.click();
      await summarySwitch.click();
    }
    await expect(summarySwitch).toHaveAttribute('aria-pressed', 'true');

    // Wait for the form to update and check if summary models field appears
    await page.waitForTimeout(3000);

    // Check if the summary models field exists (it might be hidden initially)
    // Add a summary model via the custom UI in Summaries section
    const summaryModelsContainer = page.locator('.summaries-section .form-field', { hasText: 'Summary Models' });
    const modelInput = summaryModelsContainer.locator('input[type="text"]');
    await modelInput.fill('gemini-2.0-flash');
    await modelInput.press('Enter');
  
    // Verify model chip was added
    await expect(summaryModelsContainer.locator('.selected-item')).toContainText('gemini-2.0-flash');

      // Check YAML output contains the model
      const yamlOutput = page.locator('.yaml-output code');
      await expect(yamlOutput).toContainText('summary_models:');
      await expect(yamlOutput).toContainText('- gemini-2.0-flash');

    // Test API key field appears in custom inputs
    const geminiApiKeyField = page.locator('#gemini-key');
    await expect(geminiApiKeyField).toBeVisible();
  });
});
