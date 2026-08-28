import { appendFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import type { ArtistProvider, JudgeProvider, ReferenceAnalyzer } from "./contracts.js";
import { runLoop } from "./loop.js";

export type ReferenceRecord = {
  id: string;
  path: string;
  category: string;
  prompt: string;
  split: "train" | "valid" | string;
  source?: string;
  visualNote?: string;
};

export type CollectionOptions = {
  manifestPath: string;
  outputDirectory: string;
  candidatesPerRound: number;
  rounds: number;
  split?: string;
  maxImages?: number;
  start?: number;
  baseDirectory?: string;
  artist: ArtistProvider;
  judge: JudgeProvider;
  analyzer: ReferenceAnalyzer;
};

export type CollectionSummary = {
  outputDirectory: string;
  manifestPath: string;
  selected: number;
  completed: number;
  skipped: number;
  failed: number;
};

function parseManifestLine(line: string, lineNumber: number): ReferenceRecord {
  let parsed: unknown;
  try {
    parsed = JSON.parse(line);
  } catch (error) {
    throw new Error(`Invalid JSON in manifest line ${lineNumber}: ${error instanceof Error ? error.message : String(error)}`);
  }
  if (!parsed || typeof parsed !== "object") throw new Error(`Manifest line ${lineNumber} must be an object.`);
  const record = parsed as Partial<ReferenceRecord>;
  if (typeof record.id !== "string" || !record.id.trim()) throw new Error(`Manifest line ${lineNumber} is missing a string id.`);
  if (typeof record.path !== "string" || !record.path.trim()) throw new Error(`Manifest line ${lineNumber} is missing a string path.`);
  if (typeof record.prompt !== "string" || !record.prompt.trim()) throw new Error(`Manifest line ${lineNumber} is missing a prompt.`);
  if (typeof record.split !== "string") throw new Error(`Manifest line ${lineNumber} is missing a split.`);
  return {
    id: record.id,
    path: record.path,
    category: typeof record.category === "string" ? record.category : "other",
    prompt: record.prompt,
    split: record.split,
    ...(typeof record.source === "string" ? { source: record.source } : {}),
    ...(typeof record.visualNote === "string" ? { visualNote: record.visualNote } : {})
  };
}

export async function readReferenceManifest(manifestPath: string): Promise<ReferenceRecord[]> {
  const contents = await readFile(manifestPath, "utf8");
  return contents
    .split(/\r?\n/)
    .map((line, index) => ({ line: line.trim(), lineNumber: index + 1 }))
    .filter(({ line }) => line.length > 0 && !line.startsWith("#"))
    .map(({ line, lineNumber }) => parseManifestLine(line, lineNumber));
}

function safeId(id: string): string {
  return id.replace(/[^a-zA-Z0-9_-]/g, "_");
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

async function appendStatus(path: string, status: Record<string, unknown>): Promise<void> {
  await appendFile(path, `${JSON.stringify({ ...status, timestamp: new Date().toISOString() })}\n`, "utf8");
}

export async function collectManifest(options: CollectionOptions): Promise<CollectionSummary> {
  const split = options.split ?? "train";
  const start = options.start ?? 0;
  if (!Number.isInteger(start) || start < 0) throw new Error("--start must be a non-negative integer.");
  if (options.maxImages !== undefined && (!Number.isInteger(options.maxImages) || options.maxImages < 1)) {
    throw new Error("--max-images must be a positive integer.");
  }

  const manifestPath = resolve(options.manifestPath);
  const baseDirectory = resolve(options.baseDirectory ?? process.cwd());
  const outputDirectory = resolve(options.outputDirectory);
  await mkdir(outputDirectory, { recursive: true });
  const indexPath = `${outputDirectory}/collection-index.jsonl`;
  const allRecords = await readReferenceManifest(manifestPath);
  const filtered = split === "all" ? allRecords : allRecords.filter((record) => record.split === split);
  const records = filtered.slice(start, options.maxImages === undefined ? undefined : start + options.maxImages);

  const summary: CollectionSummary = {
    outputDirectory,
    manifestPath,
    selected: records.length,
    completed: 0,
    skipped: 0,
    failed: 0
  };

  await writeFile(joinSummaryPath(outputDirectory), `${JSON.stringify({
    manifestPath,
    split,
    start,
    maxImages: options.maxImages ?? null,
    candidatesPerRound: options.candidatesPerRound,
    rounds: options.rounds,
    selected: records.length,
    startedAt: new Date().toISOString()
  }, null, 2)}\n`, "utf8");

  for (const [position, record] of records.entries()) {
    const id = safeId(record.id);
    const recordOutput = resolve(outputDirectory, id);
    const winnerPath = resolve(recordOutput, "winner.json");
    const referencePath = resolve(baseDirectory, record.path);
    console.log(`[${position + 1}/${records.length}] ${record.id} ${record.path}`);
    await mkdir(recordOutput, { recursive: true });
    await writeFile(resolve(recordOutput, "manifest-record.json"), `${JSON.stringify({
      ...record,
      referencePath,
      outputDirectory: recordOutput
    }, null, 2)}\n`, "utf8");

    if (existsSync(winnerPath)) {
      summary.skipped += 1;
      await appendStatus(indexPath, {
        id: record.id,
        path: record.path,
        prompt: record.prompt,
        split: record.split,
        outputDirectory: recordOutput,
        status: "skipped-existing"
      });
      console.log(`  skipped (winner already exists)`);
      continue;
    }

    if (!existsSync(referencePath)) {
      summary.failed += 1;
      const error = `Reference file does not exist: ${referencePath}`;
      await appendStatus(indexPath, {
        id: record.id,
        path: record.path,
        prompt: record.prompt,
        split: record.split,
        outputDirectory: recordOutput,
        status: "failed",
        error
      });
      console.error(`  failed: ${error}`);
      continue;
    }

    try {
      const result = await runLoop({
        prompt: record.prompt,
        referencePath,
        candidatesPerRound: options.candidatesPerRound,
        rounds: options.rounds,
        outputDirectory: recordOutput,
        artist: options.artist,
        judge: options.judge,
        analyzer: options.analyzer
      });
      summary.completed += 1;
      await appendStatus(indexPath, {
        id: record.id,
        path: record.path,
        prompt: record.prompt,
        split: record.split,
        outputDirectory: recordOutput,
        status: "completed",
        winner: {
          id: result.winner.id,
          score: result.winner.score.total,
          png: result.winner.pngPath
        }
      });
      console.log(`  completed: ${result.winner.score.total}/100`);
    } catch (error) {
      summary.failed += 1;
      await appendStatus(indexPath, {
        id: record.id,
        path: record.path,
        prompt: record.prompt,
        split: record.split,
        outputDirectory: recordOutput,
        status: "failed",
        error: message(error)
      });
      console.error(`  failed: ${message(error)}`);
    }
  }

  await writeFile(joinSummaryPath(outputDirectory), `${JSON.stringify({
    ...summary,
    finishedAt: new Date().toISOString()
  }, null, 2)}\n`, "utf8");
  return summary;
}

function joinSummaryPath(outputDirectory: string): string {
  return `${outputDirectory}/collection-summary.json`;
}

export function collectionRunName(): string {
  return `collection-${new Date().toISOString().replace(/[:.]/g, "-")}`;
}
