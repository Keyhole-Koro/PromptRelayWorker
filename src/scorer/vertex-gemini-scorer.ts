import Bottleneck from "bottleneck";
import { GoogleAuth } from "google-auth-library";
import { config } from "../config/app-config.js";
import { log } from "../observability/logger.js";
import type { Genome, ScoreBreakdown } from "../domain/types.js";
import { withRetry } from "../shared/utils.js";
import { parseGeminiScoreText } from "./gemini-parser.js";

const auth = new GoogleAuth({
  scopes: ["https://www.googleapis.com/auth/cloud-platform"],
});

type JsonObject = Record<string, unknown>;

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

const geminiLimiter = new Bottleneck({
  maxConcurrent: config.MAX_CONCURRENCY_GEMINI,
  reservoir: config.RATE_LIMIT_GEMINI_RPM,
  reservoirRefreshAmount: config.RATE_LIMIT_GEMINI_RPM,
  reservoirRefreshInterval: 60_000,
});

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

const scoreSchema = {
  type: "OBJECT",
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
  required: ["readability", "twist", "aesthetic", "labels", "flags", "short_reason"],
};

export async function evaluateWithGemini(params: {
  imageUri: string;
  prompt: string;
  genome: Genome;
  runId: string;
  generation: number;
  candidateIndex: number;
}): Promise<ScoreBreakdown> {
  return requestGeminiScore({
    prompt: params.prompt,
    genome: params.genome,
    runId: params.runId,
    generation: params.generation,
    candidateIndex: params.candidateIndex,
    imagePart: {
      fileData: {
        mimeType: "image/png",
        fileUri: params.imageUri,
      },
    },
  });
}

export async function evaluateWithGeminiInline(params: {
  imageBase64: string;
  mimeType: string;
  prompt: string;
  genome: Genome;
  runId: string;
  generation: number;
  candidateIndex: number;
}): Promise<ScoreBreakdown> {
  return requestGeminiScore({
    prompt: params.prompt,
    genome: params.genome,
    runId: params.runId,
    generation: params.generation,
    candidateIndex: params.candidateIndex,
    imagePart: {
      inlineData: {
        mimeType: params.mimeType,
        data: params.imageBase64,
      },
    },
  });
}

async function requestGeminiScore(params: {
  prompt: string;
  genome: Genome;
  runId: string;
  generation: number;
  candidateIndex: number;
  imagePart: Record<string, unknown>;
}): Promise<ScoreBreakdown> {
  const url = `https://aiplatform.googleapis.com/v1/projects/${config.GOOGLE_CLOUD_PROJECT}/locations/global/publishers/google/models/${config.GEMINI_MODEL}:generateContent`;

  const body = {
    contents: [
      {
        role: "user",
        parts: [
          {
            text:
              "あなたは画像審査員です。readability/twist/aestheticを0..1で厳密採点し、指定JSONのみを返してください。",
          },
          {
            text: `Prompt: ${params.prompt}`,
          },
          {
            text: `Genome: ${JSON.stringify(params.genome)}`,
          },
          params.imagePart,
        ],
      },
    ],
    generationConfig: {
      temperature: 0,
      maxOutputTokens: 256,
      responseMimeType: "application/json",
      responseSchema: scoreSchema,
    },
  };

  const json = await withRetry(
    () =>
      geminiLimiter.schedule(async () => {
        log("info", "gemini_request", {
          runId: params.runId,
          generation: params.generation,
          candidateIndex: params.candidateIndex,
        });
        return vertexPost(url, body);
      }),
    (error) => {
      const status = parseStatus(error);
      return status !== undefined && isRetriableStatus(status);
    },
    config.MAX_VERTEX_RETRIES,
  );

  const obj = asObject(json);
  const candidates = Array.isArray(obj.candidates) ? obj.candidates : [];
  const first = asObject(candidates[0]);
  const content = asObject(first.content);
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const firstPart = asObject(parts[0]);
  const text = typeof firstPart.text === "string" ? firstPart.text : "{}";

  return parseGeminiScoreText(text);
}
