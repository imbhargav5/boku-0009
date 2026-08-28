import { chromium } from "playwright";
import { mkdir, readFile } from "node:fs/promises";
import { dirname, extname } from "node:path";
import { validateSvg } from "./svg.js";

export async function renderSvg(svg: string, outputPath: string): Promise<void> {
  const safeSvg = validateSvg(svg);
  await mkdir(dirname(outputPath), { recursive: true });

  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 768, height: 768 }, deviceScaleFactor: 1 });
    await page.setContent(
      `<!doctype html><html><head><meta charset="utf-8"><style>html,body{margin:0;width:100%;height:100%;background:#fff}svg{display:block;width:768px;height:768px}</style></head><body>${safeSvg}</body></html>`,
      { waitUntil: "load" }
    );
    await page.locator("svg").screenshot({ path: outputPath });
  } finally {
    await browser.close();
  }
}

function rasterMimeType(inputPath: string): string {
  switch (extname(inputPath).toLowerCase()) {
    case ".png":
      return "image/png";
    case ".jpg":
    case ".jpeg":
      return "image/jpeg";
    case ".webp":
      return "image/webp";
    case ".gif":
      return "image/gif";
    default:
      throw new Error("Reference images must be SVG, PNG, JPG, JPEG, WEBP, or GIF.");
  }
}

/** Normalize a raster reference into the same 768px PNG canvas as SVG references. */
export async function renderRasterImage(inputPath: string, outputPath: string): Promise<void> {
  const mimeType = rasterMimeType(inputPath);
  const input = await readFile(inputPath);
  const dataUri = `data:${mimeType};base64,${input.toString("base64")}`;
  await mkdir(dirname(outputPath), { recursive: true });

  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 768, height: 768 }, deviceScaleFactor: 1 });
    await page.setContent(
      `<!doctype html><html><head><meta charset="utf-8"><style>html,body{margin:0;width:100%;height:100%;background:#fff;overflow:hidden}#reference{display:block;width:768px;height:768px;object-fit:contain;object-position:center}</style></head><body><img id="reference" alt="reference" src="${dataUri}"></body></html>`,
      { waitUntil: "load" }
    );
    await page.locator("#reference").waitFor({ state: "visible" });
    await page.locator("body").screenshot({ path: outputPath });
  } finally {
    await browser.close();
  }
}

/** Render either an SVG or raster reference into a normalized PNG. */
export async function renderReference(inputPath: string, outputPath: string): Promise<void> {
  if (extname(inputPath).toLowerCase() === ".svg") {
    await renderSvg(await readFile(inputPath, "utf8"), outputPath);
    return;
  }
  await renderRasterImage(inputPath, outputPath);
}
