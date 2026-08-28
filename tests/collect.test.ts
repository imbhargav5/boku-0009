import { existsSync } from "node:fs";
import test from "node:test";
import assert from "node:assert/strict";
import { resolve } from "node:path";
import { readReferenceManifest } from "../src/collect.js";

test("curated reference manifest is readable and every asset exists", async () => {
  const records = await readReferenceManifest(resolve("data/reference-manifest.jsonl"));
  assert.equal(records.length, 76);
  assert.equal(records.filter((record) => record.split === "train").length, 62);
  assert.equal(records.filter((record) => record.split === "valid").length, 14);
  assert.ok(records.every((record) => existsSync(resolve(record.path))));
});
