import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests/ui',
  retries: process.env.CI ? 1 : 0,
});