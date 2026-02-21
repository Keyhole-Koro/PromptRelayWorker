import test from "node:test";
import assert from "node:assert/strict";
import { parseGeminiScoreText } from "../src/scorer/gemini-parser.js";

test("parseGeminiScoreText parses structured output", () => {
  const score = parseGeminiScoreText(
    JSON.stringify({
      readability: 0.9,
      twist: 0.7,
      aesthetic: 0.8,
      labels: { scene: "park", subject: "cat", action: "running" },
      flags: ["ok"],
      short_reason: "clear and odd",
    }),
  );

  assert.equal(score.readability, 0.9);
  assert.equal(score.twist, 0.7);
  assert.equal(score.aesthetic, 0.8);
  assert.equal(score.fitness, 0.815);
  assert.deepEqual(score.labels, { scene: "park", subject: "cat", action: "running" });
  assert.deepEqual(score.flags, ["ok"]);
  assert.equal(score.short_reason, "clear and odd");
});

test("parseGeminiScoreText applies defaults and clamps", () => {
  const score = parseGeminiScoreText(
    JSON.stringify({
      readability: 2,
      twist: -1,
      aesthetic: 0.4,
      labels: {},
      flags: ["x", 1],
    }),
  );

  assert.equal(score.readability, 1);
  assert.equal(score.twist, 0);
  assert.equal(score.aesthetic, 0.4);
  assert.equal(score.fitness, 0.56);
  assert.deepEqual(score.flags, ["x"]);
  assert.equal(score.short_reason, "no reason");
});

test("parseGeminiScoreText throws on invalid JSON", () => {
  assert.throws(() => parseGeminiScoreText("not-json"), /gemini structured output parse failed/);
});
