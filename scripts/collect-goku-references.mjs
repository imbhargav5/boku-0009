import { createHash } from "node:crypto";
import { mkdir, readFile, readdir, stat, writeFile } from "node:fs/promises";
import { basename, extname, join } from "node:path";

const output = process.argv[2] ?? "references/goku";
const target = Number(process.argv[3] ?? 200);
const api = "https://safebooru.org/index.php?page=dapi&s=post&q=index&json=1";
const manifestPath = join(output, "bulk-metadata.jsonl");

const imageExtensions = new Set([".jpg", ".jpeg", ".png", ".webp"]);
const categories = ["base", "action", "transformation", "portrait", "kid", "manga-line", "other"];
const quotas = new Map([
  ["base", 50],
  ["action", 40],
  ["transformation", 30],
  ["portrait", 25],
  ["kid", 15],
  ["manga-line", 15],
  ["other", 25]
]);

function categoryFor(tags) {
  if (tags.includes("monochrome")) return "manga-line";
  if (tags.includes("super_saiyan")) return "transformation";
  if (tags.includes("child") || tags.includes("dragon_ball_(classic)")) return "kid";
  if (/(fighting|kamehameha|flying|punch|kick|battle|duel|attack)/.test(tags)) return "action";
  if (/(upper_body|close-up|portrait|face|headshot)/.test(tags)) return "portrait";
  if (/(solo|standing)/.test(tags)) return "base";
  return "other";
}

function cleanExtension(url) {
  const extension = extname(new URL(url).pathname).toLowerCase();
  return imageExtensions.has(extension) ? extension : ".jpg";
}

async function fetchPosts(page) {
  const url = new URL(api);
  url.searchParams.set("tags", "son_goku rating:general");
  url.searchParams.set("limit", "100");
  url.searchParams.set("pid", String(page));
  const response = await fetch(url, { headers: { "user-agent": "local-goku-reference-collector/1.0" } });
  if (!response.ok) throw new Error(`Safebooru page ${page} failed: ${response.status}`);
  return response.json();
}

async function localImages(directory) {
  const names = await readdir(directory).catch(() => []);
  const files = await Promise.all(names.filter((name) => imageExtensions.has(extname(name).toLowerCase())).map(async (name) => {
    const path = join(directory, name);
    const info = await stat(path);
    const hash = createHash("sha256").update(await readFile(path)).digest("hex");
    return { name, bytes: info.size, hash };
  }));
  return files;
}

async function download(post, bucket, seenHashes) {
  if (!post.file_url || post.width < 400 || post.height < 400) return null;
  const response = await fetch(post.file_url, { headers: { "user-agent": "local-goku-reference-collector/1.0" } });
  const type = response.headers.get("content-type") ?? "";
  if (!response.ok || !type.startsWith("image/")) return null;
  const bytes = Buffer.from(await response.arrayBuffer());
  if (bytes.length < 10_000) return null;
  const hash = createHash("sha256").update(bytes).digest("hex");
  if (seenHashes.has(hash)) return null;
  const filename = `${String(post.id).padStart(8, "0")}-${bucket}${cleanExtension(post.file_url)}`;
  await writeFile(join(output, filename), bytes);
  seenHashes.add(hash);
  return {
    file: filename,
    id: post.id,
    category: bucket,
    source: post.file_url,
    width: post.width,
    height: post.height,
    tags: post.tags
  };
}

async function main() {
  await mkdir(output, { recursive: true });
  const existing = await localImages(output);
  if (existing.length >= target) {
    console.log(`Already have ${existing.length} image files in ${output}.`);
    return;
  }

  const posts = [];
  for (let page = 0; page < 8; page += 1) {
    posts.push(...await fetchPosts(page));
  }

  const selected = [];
  const categoryCounts = new Map(categories.map((category) => [category, 0]));
  const seenIds = new Set();
  const sorted = posts
    .filter((post) => post.file_url && post.width >= 400 && post.height >= 400)
    .sort((left, right) => (right.width * right.height) - (left.width * left.height));

  for (const post of sorted) {
    if (selected.length >= target - existing.length || seenIds.has(post.id)) continue;
    const bucket = categoryFor(post.tags ?? "");
    if ((categoryCounts.get(bucket) ?? 0) >= (quotas.get(bucket) ?? 0)) continue;
    selected.push({ post, bucket });
    seenIds.add(post.id);
    categoryCounts.set(bucket, (categoryCounts.get(bucket) ?? 0) + 1);
  }
  for (const post of sorted) {
    if (selected.length >= target - existing.length || seenIds.has(post.id)) continue;
    selected.push({ post, bucket: categoryFor(post.tags ?? "") });
    seenIds.add(post.id);
  }

  const seenHashes = new Set(existing.map((file) => file.hash));
  const records = [];
  for (let index = 0; index < selected.length; index += 4) {
    const batch = selected.slice(index, index + 4);
    const downloaded = await Promise.all(batch.map(({ post, bucket }) => download(post, bucket, seenHashes).catch(() => null)));
    records.push(...downloaded.filter(Boolean));
    console.log(`Downloaded ${records.length}/${selected.length} bulk images`);
  }
  await writeFile(manifestPath, `${records.map((record) => JSON.stringify(record)).join("\n")}\n`);
  console.log(`Added ${records.length} images; ${existing.length + records.length} total images in ${output}.`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
