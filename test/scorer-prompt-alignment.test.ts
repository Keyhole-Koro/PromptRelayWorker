import assert from "node:assert/strict";
import test from "node:test";
import { normalizeCosineToScore01 } from "../src/scorer/prompt-alignment.js";

test("normalizeCosineToScore01 maps cosine to 0..1", () => {
  assert.equal(normalizeCosineToScore01(-1), 0);
  assert.equal(normalizeCosineToScore01(0), 0.5);
  assert.equal(normalizeCosineToScore01(1), 1);
});

test("normalizeCosineToScore01 clamps out-of-range values", () => {
  assert.equal(normalizeCosineToScore01(2), 1);
  assert.equal(normalizeCosineToScore01(-3), 0);
});
