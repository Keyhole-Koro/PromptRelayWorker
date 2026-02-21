import test from "node:test";
import assert from "node:assert/strict";
import { computeFitness } from "../src/scorer/fitness.js";

test("computeFitness uses weighted formula", () => {
  const result = computeFitness(0.8, 0.6, 0.4);
  assert.ok(Math.abs(result - 0.67) < 1e-9);
});

test("computeFitness clamps to [0,1]", () => {
  assert.equal(computeFitness(2, 2, 2), 1);
  assert.equal(computeFitness(-1, -1, -1), 0);
});
