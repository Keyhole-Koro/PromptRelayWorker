import { readFile } from "node:fs/promises";
import Bottleneck from "bottleneck";
import { GoogleAuth } from "google-auth-library";
import { z } from "zod";
import { config } from "../config/app-config.js";
import type { ScoreBreakdown } from "../domain/types.js";
import { log } from "../observability/logger.js";
import { clamp01, withRetry } from "../shared/utils.js";

const auth = new GoogleAuth({
  scopes: ["https://www.googleapis.com/auth/cloud-platform"],
});

type JsonObject = Record<string, unknown>;

const geminiLimiter = new Bottleneck({
  maxConcurrent: config.MAX_CONCURRENCY_GEMINI,
  reservoir: config.RATE_LIMIT_GEMINI_RPM,
  reservoirRefreshAmount: config.RATE_LIMIT_GEMINI_RPM,
  reservoirRefreshInterval: 60_000,
});

const scoreSchema = z.object({
  readability: z.number(),
  twist: z.number(),
  aesthetic: z.number(),
  labels: z
    .object({
      scene: z.string().optional(),
      subject: z.string().optional(),
      action: z.string().optional(),
    })
    .optional(),
  flags: z.array(z.string()).optional(),
  short_reason: z.string().min(1).optional(),
});

function asObject(value: unknown): JsonObject {
  return typeof value === "object" && value !== null ? (value as JsonObject) : {};
}

function isRetriableStatus(status: number): boolean {
  return status === 429 || status === 503;
}

function parseStatus(error: unknown): number | undefined {
  if (!(error instanceof Error)) {
    return undefined;
  }
  const match = error.message.match(/status=(\d{3})/);
  if (!match) {
    return undefined;
  }
  return Number.parseInt(match[1], 10);
}

function calcFitness(readability: number, twist: number, aesthetic: number): number {
  return clamp01(0.5 * readability + 0.35 * twist + 0.15 * aesthetic);
}

async function accessToken(): Promise<string> {
  const client = await auth.getClient();
  const token = await client.getAccessToken();
  if (!token.token) {
    throw new Error("failed to get ADC access token");
  }
  return token.token;
}

async function vertexPost(url: string, body: unknown): Promise<unknown> {
  const token = await accessToken();
  const res = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
  const text = await res.text();
  if (!res.ok) {
    throw new Error(`vertex request failed status=${res.status} body=${text}`);
  }
  try {
    return JSON.parse(text);
  } catch {
    throw new Error("vertex response was not valid JSON");
  }
}

function extractModelJson(json: unknown): unknown {
  const top = asObject(json);
  const candidates = Array.isArray(top.candidates) ? top.candidates : [];
  const first = asObject(candidates[0]);
  const content = asObject(first.content);
  const parts = Array.isArray(content.parts) ? content.parts : [];

  for (const part of parts) {
    const obj = asObject(part);
    if (obj.text && typeof obj.text === "string") {
      try {
        return JSON.parse(obj.text);
      } catch {
        // fall through and continue scanning parts
      }
    }
    if (obj.inlineData && typeof obj.inlineData === "object") {
      const inline = asObject(obj.inlineData);
      const maybeText = inline.data;
      if (typeof maybeText === "string") {
        try {
          return JSON.parse(maybeText);
        } catch {
          // ignore
        }
      }
    }
  }

  throw new Error("gemini response missing parseable JSON body");
}

async function scoreWithGeminiStructured(params: {
  prompt: string;
  imageBase64: string;
  mimeType: string;
  runId: string;
  generation: number;
  candidateIndex: number;
}): Promise<ScoreBreakdown> {
  const url = `https://aiplatform.googleapis.com/v1/projects/${config.GOOGLE_CLOUD_PROJECT}/locations/global/publishers/google/models/${config.GEMINI_MODEL}:generateContent`;

  const body = {
    contents: [
      {
        role: "user",
        parts: [
          {
            text: [
              "Score the image against the prompt and return ONLY JSON.",
              "readability: how clearly a player can identify scene/subject/action.",
              "twist: how noticeable and interesting the twist is.",
              "aesthetic: overall visual quality.",
              "Each score must be number between 0 and 1.",
              `Prompt: ${params.prompt}`,
            ].join("\n"),
          },
          {
            inlineData: {
              mimeType: params.mimeType,
              data: params.imageBase64,
            },
          },
        ],
      },
    ],
    generationConfig: {
      temperature: 0,
      topP: 0,
      maxOutputTokens: 256,
      responseMimeType: "application/json",
      responseSchema: {
        type: "OBJECT",
        required: ["readability", "twist", "aesthetic", "short_reason"],
        properties: {
          readability: { type: "NUMBER" },
          twist: { type: "NUMBER" },
          aesthetic: { type: "NUMBER" },
          labels: {
            type: "OBJECT",
            properties: {
              scene: { type: "STRING" },
              subject: { type: "STRING" },
              action: { type: "STRING" },
            },
          },
          flags: {
            type: "ARRAY",
            items: { type: "STRING" },
          },
          short_reason: { type: "STRING" },
        },
      },
    },
  };

  const raw = await withRetry(
    () =>
      geminiLimiter.schedule(async () => {
        log("info", "gemini_score_request", {
          runId: params.runId,
          generation: params.generation,
          candidateIndex: params.candidateIndex,
          model: config.GEMINI_MODEL,
        });
        return vertexPost(url, body);
      }),
    (error) => {
      const status = parseStatus(error);
      return status !== undefined && isRetriableStatus(status);
    },
    config.MAX_VERTEX_RETRIES,
  );

  const parsed = scoreSchema.parse(extractModelJson(raw));
  const readability = clamp01(parsed.readability);
  const twist = clamp01(parsed.twist);
  const aesthetic = clamp01(parsed.aesthetic);

  return {
    readability,
    twist,
    aesthetic,
    fitness: calcFitness(readability, twist, aesthetic),
    labels: parsed.labels ?? {},
    flags: parsed.flags ?? [],
    short_reason: parsed.short_reason ?? "scored by gemini",
  };
}

function fileUriToPath(uri: string): string {
  if (!uri.startsWith("file://")) {
    throw new Error("evaluateWithGemini currently supports only file:// imageUri");
  }
  return decodeURIComponent(uri.slice("file://".length));
}

export async function evaluateWithGemini(params: {
  imageUri: string;
  prompt: string;
  runId: string;
  generation: number;
  candidateIndex: number;
}): Promise<ScoreBreakdown> {
  const filePath = fileUriToPath(params.imageUri);
  const imageBuffer = await readFile(filePath);
  const imageBase64 = imageBuffer.toString("base64");

  return evaluateWithGeminiInline({
    imageBase64,
    mimeType: "image/png",
    prompt: params.prompt,
    runId: params.runId,
    generation: params.generation,
    candidateIndex: params.candidateIndex,
  });
}

export async function evaluateWithGeminiInline(params: {
  imageBase64: string;
  mimeType: string;
  prompt: string;
  runId: string;
  generation: number;
  candidateIndex: number;
}): Promise<ScoreBreakdown> {
  try {
    return await scoreWithGeminiStructured({
      prompt: params.prompt,
      imageBase64: params.imageBase64,
      mimeType: params.mimeType,
      runId: params.runId,
      generation: params.generation,
      candidateIndex: params.candidateIndex,
    });
  } catch (error) {
    log("warn", "gemini_score_failed", {
      runId: params.runId,
      generation: params.generation,
      candidateIndex: params.candidateIndex,
      error: error instanceof Error ? error.message : String(error),
    });

    return {
      readability: 0,
      twist: 0,
      aesthetic: 0,
      fitness: 0,
      labels: {},
      flags: ["gemini_score_failed_fallback"],
      short_reason: "fallback: scoring failed",
    };
  }
}
