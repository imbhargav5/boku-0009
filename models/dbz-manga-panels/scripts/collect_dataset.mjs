#!/usr/bin/env node

import { createHash } from "node:crypto";
import { mkdir, readFile, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const modelRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const defaults = {
  output: path.join(modelRoot, "data/raw"),
  limit: 48,
  pages: 4,
};
const queries = ["dragon_ball_z comic", "dragon_ball 4koma", "dragon_ball speech_bubble"];
const api = "https://safebooru.org/index.php?page=dapi&s=post&q=index&json=1";
const imageExtensions = new Set([".jpg", ".jpeg", ".png", ".webp"]);
const excludedFranchiseTags = new Set([
  "bleach",
  "fairy_tail",
  "genshin_impact",
  "hunter_x_hunter",
  "kantai_collection",
  "kinnikuman",
  "my_hero_academia",
  "naruto",
  "one-piece_man",
  "one_piece",
  "pokemon",
  "touhou",
]);

function parseArgs(argv) {
  const args = { ...defaults };
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith("--")) throw new Error(`Unexpected argument ${token}`);
    const key = token.slice(2);
    const value = argv[index + 1];
    if (value === undefined || value.startsWith("--")) throw new Error(`Missing value for --${key}`);
    if (!(key in args)) throw new Error(`Unknown option --${key}`);
    args[key] = key === "limit" || key === "pages" ? Number(value) : path.resolve(value);
    index += 1;
  }
  if (!Number.isInteger(args.limit) || args.limit < 1) throw new Error("--limit must be a positive integer");
  if (!Number.isInteger(args.pages) || args.pages < 1) throw new Error("--pages must be a positive integer");
  return args;
}

function tagSet(post) {
  return new Set(String(post.tags ?? "").split(/\s+/).filter(Boolean));
}

function isCandidate(post) {
  const tags = tagSet(post);
  const width = Number(post.width ?? 0);
  const height = Number(post.height ?? 0);
  return Boolean(post.file_url)
    && tags.has("dragon_ball_z")
    && tags.has("comic")
    && (tags.has("monochrome") || tags.has("greyscale"))
    && !tags.has("crossover")
    && !tags.has("parody")
    && ![...excludedFranchiseTags].some((tag) => tags.has(tag))
    && Math.min(width, height) >= 300
    && Math.max(width, height) >= 800
    && width * height >= 450_000;
}

function candidateScore(post) {
  const tags = tagSet(post);
  const area = Number(post.width) * Number(post.height);
  return (tags.has("4koma") ? 8 : 0)
    + (tags.has("speech_bubble") ? 5 : 0)
    + (tags.has("greyscale") ? 2 : 0)
    + Math.min(Math.log2(Math.max(area, 1)) / 4, 6);
}

async function fetchJson(url, attempts = 3) {
  let lastError;
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      const response = await fetch(url, { headers: { "user-agent": "dbz-manga-panel-experiment/1.0" } });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const payload = await response.json();
      return Array.isArray(payload) ? payload : [];
    } catch (error) {
      lastError = error;
      if (attempt < attempts) await new Promise((resolve) => setTimeout(resolve, 500 * attempt));
    }
  }
  throw lastError;
}

async function fetchPool(pageCount) {
  const pool = new Map();
  for (const query of queries) {
    for (let page = 0; page < pageCount; page += 1) {
      const url = new URL(api);
      url.searchParams.set("tags", query);
      url.searchParams.set("limit", "100");
      url.searchParams.set("pid", String(page));
      const posts = await fetchJson(url);
      if (posts.length === 0) break;
      for (const post of posts) {
        if (!isCandidate(post)) continue;
        const previous = pool.get(String(post.id));
        const sourceQueries = new Set(previous?.sourceQueries ?? []);
        sourceQueries.add(query);
        pool.set(String(post.id), { ...post, sourceQueries: [...sourceQueries] });
      }
    }
  }
  return [...pool.values()].sort((left, right) => candidateScore(right) - candidateScore(left) || Number(right.id) - Number(left.id));
}

function extensionFor(post, contentType) {
  const fromUrl = path.extname(new URL(post.file_url).pathname).toLowerCase();
  if (imageExtensions.has(fromUrl)) return fromUrl === ".jpeg" ? ".jpg" : fromUrl;
  if (contentType.includes("png")) return ".png";
  if (contentType.includes("webp")) return ".webp";
  return ".jpg";
}

async function validExisting(output, record) {
  try {
    const info = await stat(path.join(output, record.file));
    return info.isFile() && info.size === record.bytes;
  } catch {
    return false;
  }
}

async function loadExisting(output) {
  try {
    const text = await readFile(path.join(output, "source-manifest.jsonl"), "utf8");
    return new Map(text.split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line)).map((row) => [String(row.id), row]));
  } catch {
    return new Map();
  }
}

async function download(post, output) {
  const response = await fetch(post.file_url, { headers: { "user-agent": "dbz-manga-panel-experiment/1.0" } });
  const contentType = response.headers.get("content-type") ?? "";
  if (!response.ok || !contentType.startsWith("image/")) throw new Error(`Image ${post.id} failed: HTTP ${response.status}`);
  const bytes = Buffer.from(await response.arrayBuffer());
  if (bytes.length < 10_000) throw new Error(`Image ${post.id} is unexpectedly small`);
  const extension = extensionFor(post, contentType);
  const filename = `${String(post.id).padStart(8, "0")}${extension}`;
  await writeFile(path.join(output, filename), bytes);
  return {
    id: String(post.id),
    file: filename,
    post_url: `https://safebooru.org/index.php?page=post&s=view&id=${post.id}`,
    source_url: post.file_url,
    source_queries: post.sourceQueries,
    width: Number(post.width),
    height: Number(post.height),
    bytes: bytes.length,
    sha256: createHash("sha256").update(bytes).digest("hex"),
    tags: String(post.tags ?? "").split(/\s+/).filter(Boolean),
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  await mkdir(args.output, { recursive: true });
  const existing = await loadExisting(args.output);
  const candidates = await fetchPool(args.pages);
  if (candidates.length < args.limit) throw new Error(`Only ${candidates.length} matching references found for requested limit ${args.limit}`);
  const selected = candidates.slice(0, args.limit);
  const records = [];
  for (let offset = 0; offset < selected.length; offset += 4) {
    const batch = selected.slice(offset, offset + 4);
    const downloaded = await Promise.all(batch.map(async (post) => {
      const cached = existing.get(String(post.id));
      if (cached && await validExisting(args.output, cached)) return cached;
      return download(post, args.output);
    }));
    records.push(...downloaded);
    console.log(`Collected ${records.length}/${selected.length} source images`);
  }
  const manifest = records.map((record) => JSON.stringify(record)).join("\n") + "\n";
  await writeFile(path.join(args.output, "source-manifest.jsonl"), manifest);
  await writeFile(path.join(args.output, "collection-summary.json"), `${JSON.stringify({
    collected_at: new Date().toISOString(),
    api,
    queries,
    pages_per_query: args.pages,
    candidate_count: candidates.length,
    selected_count: records.length,
  }, null, 2)}\n`);
  console.log(`Dataset source manifest: ${path.join(args.output, "source-manifest.jsonl")}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack : error);
  process.exitCode = 1;
});
