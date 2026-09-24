// Headless screenshot capture for the stock-trend-lab dashboard.
// Prerequisites: `npm run dev` (frontend) and `uvicorn src.api.main:app` (backend) running.
// Usage: node scripts/capture-screenshots.mjs
import { mkdirSync } from 'node:fs';
import { chromium } from 'playwright';

const BASE = 'http://localhost:5173';
const OUT_DIR = 'C:/PythonProject/stock_trent_lab/docs/screenshots';
// Cached range, matching the numbers documented in the README.
const START = '2022-01-01';
const END = '2024-12-31';

mkdirSync(OUT_DIR, { recursive: true });

const browser = await chromium.launch();
const page = await browser.newPage({
  viewport: { width: 1440, height: 900 },
  deviceScaleFactor: 2,
});

async function setDate(index, ymd) {
  const input = page.locator('input.date-input').nth(index);
  await input.click();
  await input.fill(ymd);
  await input.press('Enter');
  await page.keyboard.press('Escape');
}

await page.goto(BASE, { waitUntil: 'networkidle' });

await setDate(0, START);
await setDate(1, END);
console.log('start =', await page.locator('input.date-input').nth(0).inputValue());
console.log('end   =', await page.locator('input.date-input').nth(1).inputValue());

await page.locator('.controls button').click();
await page.waitForSelector('table.metrics', { timeout: 90000 });
await page.waitForTimeout(1500);

await page.screenshot({ path: `${OUT_DIR}/dashboard.png`, fullPage: true });

await page.locator('.glossary summary').click();
await page.waitForTimeout(300);
await page.locator('.glossary').screenshot({ path: `${OUT_DIR}/glossary.png` });

const reportSection = page.locator('section').filter({ hasText: 'RAG report' });
await reportSection.scrollIntoViewIfNeeded();
await page.locator('.report-actions button').click();
await page.waitForSelector('.report-markdown', { timeout: 120000 });
await page.waitForTimeout(1500);
await reportSection.scrollIntoViewIfNeeded();
await reportSection.screenshot({ path: `${OUT_DIR}/rag-report.png` });

await browser.close();
console.log('Captured: dashboard.png, glossary.png, rag-report.png');
