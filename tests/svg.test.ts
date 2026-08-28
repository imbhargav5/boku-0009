import assert from "node:assert/strict";
import test from "node:test";
import { artistScaffoldSvg, extractSvg, fixtureSvg, validateSvg } from "../src/svg.js";

test("fixture SVG is accepted by the restricted renderer", () => {
  const svg = fixtureSvg(0.8, 2);
  assert.equal(validateSvg(svg), svg);
});

test("Artist scaffold stays within the restricted SVG contract", () => {
  assert.doesNotThrow(() => validateSvg(artistScaffoldSvg()));
});

test("extractSvg removes incidental model prose", () => {
  const svg = extractSvg("Here you go:\n<svg viewBox='0 0 10 10'><circle cx='5' cy='5' r='4'/></svg>\nDone");
  assert.match(svg, /^<svg/);
});

test("unsafe SVG features are rejected", () => {
  assert.throws(
    () => validateSvg("<svg viewBox='0 0 10 10'><script>alert(1)</script></svg>"),
    /forbidden/
  );
  assert.throws(
    () => validateSvg("<svg viewBox='0 0 10 10'><image href='https://example.test/a.png'/></svg>"),
    /forbidden/
  );
});

test("runaway path geometry is rejected", () => {
  assert.throws(
    () => validateSvg("<svg viewBox='0 0 512 512'><path d='M0 0 L900 900'/></svg>"),
    /outside the 0..512 canvas/
  );
});
