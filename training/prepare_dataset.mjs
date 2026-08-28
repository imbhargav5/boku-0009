#!/usr/bin/env node

/**
 * Export the curated reference split into the image/caption layout consumed by
 * train_lora.py.  The collector's normalized reference.png files are used so
 * the training input is deterministic and independent of the original JPEG
 * dimensions.
 */

import { cp, mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

function parseArgs(argv) {
  const args = {
    manifest: "data/reference-manifest.jsonl",
    collection: "artifacts/collection-conditioned-train",
    split: "train",
    out: "artifacts/model/dataset",
    limit: null,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (!token.startsWith("--")) continue;
    const key = token.slice(2);
    const value = argv[i + 1];
    if (value === undefined || value.startsWith("--")) {
      throw new Error(`Missing value for --${key}`);
    }
    if (!(key in args)) throw new Error(`Unknown option --${key}`);
    args[key] = key === "limit" ? Number(value) : value;
    i += 1;
  }
  return args;
}

function lines(text) {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
}

function compact(value) {
  return String(value ?? "")
    .replace(/[\r\n]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

async function readJson(file) {
  return JSON.parse(await readFile(file, "utf8"));
}

function captionFor(record, analysis) {
  const fields = [
    analysis?.framing,
    analysis?.pose,
    analysis?.silhouette,
    analysis?.hair,
    analysis?.face,
    analysis?.clothing,
    analysis?.lineAndRendering,
    analysis?.background,
  ]
    .map(compact)
    .filter(Boolean);
  const category = compact(record.category).replace(/[^a-z0-9 -]/gi, "");
  return [
    "goku_manga",
    category,
    compact(record.prompt),
    ...fields,
  ]
    .filter(Boolean)
    .join(", ");
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const root = process.cwd();
  const manifestPath = path.resolve(root, args.manifest);
  const collectionPath = path.resolve(root, args.collection);
  const outPath = path.resolve(root, args.out);
  const imagePath = path.join(outPath, "images");
  await mkdir(imagePath, { recursive: true });

  const records = lines(await readFile(manifestPath, "utf8"))
    .map((line) => JSON.parse(line))
    .filter((record) => record.split === args.split)
    .slice(0, args.limit ?? undefined);
  if (records.length === 0) throw new Error(`No ${args.split} records found in ${manifestPath}`);

  const metadata = [];
  const skipped = [];
  for (const record of records) {
    const recordDir = path.join(collectionPath, record.id);
    const sourceImage = path.join(recordDir, "reference.png");
    const analysisPath = path.join(recordDir, "reference-analysis.json");
    const winnerPath = path.join(recordDir, "winner.json");
    try {
      const [analysis, winner] = await Promise.all([readJson(analysisPath), readJson(winnerPath)]);
      const targetName = `${record.id}.png`;
      await cp(sourceImage, path.join(imagePath, targetName));
      const text = captionFor(record, analysis);
      await writeFile(path.join(imagePath, `${record.id}.txt`), `${text}\n`);
      metadata.push({
        file_name: `images/${targetName}`,
        text,
        source_id: record.id,
        category: record.category,
        source: record.source,
        winner_score: winner.score ?? null,
      });
    } catch (error) {
      skipped.push({ id: record.id, reason: error instanceof Error ? error.message : String(error) });
    }
  }

  await writeFile(
    path.join(outPath, "metadata.jsonl"),
    metadata.map((row) => JSON.stringify(row)).join("\n") + (metadata.length ? "\n" : ""),
  );
  await writeFile(
    path.join(outPath, "dataset-summary.json"),
    JSON.stringify(
      {
        manifest: path.relative(root, manifestPath),
        collection: path.relative(root, collectionPath),
        split: args.split,
        requested: records.length,
        exported: metadata.length,
        skipped,
        triggerToken: "goku_manga",
      },
      null,
      2,
    ) + "\n",
  );
  console.log(`Exported ${metadata.length}/${records.length} ${args.split} images to ${path.relative(root, outPath)}`);
  if (skipped.length) console.warn(`Skipped ${skipped.length} records; see ${path.join(outPath, "dataset-summary.json")}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack : error);
  process.exitCode = 1;
});
