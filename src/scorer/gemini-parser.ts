import type { ScoreBreakdown } from "../domain/types.js";
import { clamp01 } from "../shared/utils.js";
import { computeFitness } from "./fitness.js";

type JsonObject = Record<string, unknown>;

function asObject(value: unknown): JsonObject {
  return typeof value === "object" && value !== null ? (value as JsonObject) : {};
}

export function parseGeminiScoreText(text: string): ScoreBreakdown {
  let parsed: JsonObject;
  try {
    parsed = asObject(JSON.parse(text));
  } catch {
    throw new Error(`gemini structured output parse failed: ${text}`);
  }

  const readability = clamp01(Number(parsed.readability ?? 0));
  const twist = clamp01(Number(parsed.twist ?? 0));
  const aesthetic = clamp01(Number(parsed.aesthetic ?? 0));
  const labels = asObject(parsed.labels);
  const flagsRaw = Array.isArray(parsed.flags) ? parsed.flags : [];

  return {
    readability,
    twist,
    aesthetic,
    fitness: computeFitness(readability, twist, aesthetic),
    labels: {
      scene: typeof labels.scene === "string" ? labels.scene : undefined,
      subject: typeof labels.subject === "string" ? labels.subject : undefined,
      action: typeof labels.action === "string" ? labels.action : undefined,
    },
    flags: flagsRaw.filter((f): f is string => typeof f === "string"),
    short_reason:
      typeof parsed.short_reason === "string" && parsed.short_reason.length > 0
        ? parsed.short_reason
        : "no reason",
  };
}
