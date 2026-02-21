import test from "node:test";
import assert from "node:assert/strict";
import { cosineSimilarity, cosineToScore100 } from "../src/scorer/score-math.js";

test("cosine similarity maps to 0..100 score", () => {
  const a = [1, 0, 0];
  const b = [1, 0, 0];
  const c = [-1, 0, 0];

  const cosineSame = cosineSimilarity(a, b);
  const cosineOpposite = cosineSimilarity(a, c);

  assert.equal(cosineSame, 1);
  assert.equal(cosineOpposite, -1);
  assert.equal(cosineToScore100(cosineSame), 100);
  assert.equal(cosineToScore100(cosineOpposite), 0);
});

test("cosine similarity validates vector shape", () => {
  assert.throws(() => cosineSimilarity([], []), /Embedding vectors must be non-empty and same length/);
  assert.throws(() => cosineSimilarity([1, 2], [1]), /Embedding vectors must be non-empty and same length/);
});
