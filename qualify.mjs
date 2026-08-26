import { mkdir, writeFile } from 'node:fs/promises';
import { chromium } from 'playwright-core';

const previewUrl =
  process.env.TYPEGPU_PREVIEW_URL ??
  'https://127.0.0.1:9443/?autorun=1&benchmark=1';
const evidenceDir = new URL('./evidence/', import.meta.url);
await mkdir(evidenceDir, { recursive: true });

const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH ?? '/snap/bin/chromium',
  headless: true,
  args: [
    '--allow-chrome-scheme-url',
    '--disable-vulkan-surface',
    '--enable-features=Vulkan',
    '--enable-gpu',
    '--enable-unsafe-webgpu',
    '--use-angle=vulkan',
  ],
});

const context = await browser.newContext({
  ignoreHTTPSErrors: true,
  viewport: { width: 1440, height: 1100 },
});
const page = await context.newPage();
const consoleMessages = [];
const pageErrors = [];
page.on('console', (message) => consoleMessages.push(`${message.type()}: ${message.text()}`));
page.on('pageerror', (error) => pageErrors.push(error.stack ?? error.message));

let qualification;
let gpuReport = '';
let outcome = 'failed';
try {
  await page.goto(previewUrl, { waitUntil: 'domcontentloaded', timeout: 60_000 });
  await page.waitForFunction(
    () => ['complete', 'failed'].includes(document.documentElement.dataset.qualification ?? ''),
    undefined,
    { timeout: 420_000 },
  );
  await page.waitForTimeout(2_000);
  qualification = await page.evaluate(() => globalThis.__TYPEGPU_QUALIFICATION__ ?? null);
  outcome = qualification?.status === 'complete' ? 'passed' : 'failed';
  await page.screenshot({ path: new URL('static-demo.png', evidenceDir).pathname, fullPage: true });
  for (const [value, label] of [
    ['0', 'relit'],
    ['1', 'camera'],
    ['2', 'relative-disparity'],
    ['3', 'normals'],
  ]) {
    await page.locator('#view-select').selectOption(value);
    await page.waitForTimeout(500);
    await page.screenshot({
      path: new URL(`static-demo-${label}.png`, evidenceDir).pathname,
      fullPage: true,
    });
  }

  const gpuPage = await context.newPage();
  await gpuPage.goto('chrome://gpu', { waitUntil: 'domcontentloaded', timeout: 30_000 });
  gpuReport = await gpuPage.locator('body').innerText({ timeout: 30_000 });
  await gpuPage.close();
} finally {
  const report = {
    schemaVersion: 1,
    capturedAt: new Date().toISOString(),
    outcome,
    previewUrl,
    qualification: qualification ?? null,
    browserVersion: browser.version(),
    consoleMessages,
    pageErrors,
    chromeGpuReport: gpuReport,
  };
  await writeFile(new URL('qualification.json', evidenceDir), `${JSON.stringify(report, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
  await browser.close();
}

if (outcome !== 'passed') {
  process.exitCode = 1;
}
