import Bottleneck from "bottleneck";
import { GoogleAuth } from "google-auth-library";
import { config } from "../config/app-config.js";
import { log } from "../observability/logger.js";
import { cosineSimilarity } from "./score-math.js";
import { withRetry } from "../shared/utils.js";

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

const embedLimiter = new Bottleneck({
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

export function normalizeCosineToScore01(cosine: number): number {
  const normalized = (Math.max(-1, Math.min(1, cosine)) + 1) / 2;
  return Math.max(0, Math.min(1, normalized));
}

function parseEmbeddingVector(json: unknown): number[] {
  const top = asObject(json);
  const predictions = Array.isArray(top.predictions) ? top.predictions : [];
  const first = asObject(predictions[0]);

  const embObj = asObject(first.embeddings);
  const valuesFromEmbeddings = Array.isArray(embObj.values) ? embObj.values : undefined;
  const embVals = Array.isArray(first.embedding) ? first.embedding : undefined;
  const genericVals = Array.isArray(first.values) ? first.values : undefined;

  const vector = valuesFromEmbeddings ?? embVals ?? genericVals;
  const numeric = (vector ?? []).filter((v): v is number => typeof v === "number");
  if (numeric.length === 0) {
    throw new Error("embedding response missing numeric vector");
  }
  return numeric;
}

async function embedText(text: string, runId: string): Promise<number[]> {
  const url = `https://aiplatform.googleapis.com/v1/projects/${config.GOOGLE_CLOUD_PROJECT}/locations/global/publishers/google/models/${config.TEXT_EMBEDDING_MODEL}:predict`;
  const body = {
    instances: [{ content: text }],
  };

  const json = await withRetry(
    () =>
      embedLimiter.schedule(async () => {
        log("info", "text_embedding_request", { runId, model: config.TEXT_EMBEDDING_MODEL });
        return vertexPost(url, body);
      }),
    (error) => {
      const status = parseStatus(error);
      return status !== undefined && isRetriableStatus(status);
    },
    config.MAX_VERTEX_RETRIES,
  );

  return parseEmbeddingVector(json);
}

export async function scorePromptAlignment(params: {
  promptA: string;
  promptB: string;
  runId: string;
}): Promise<{ cosine: number; score01: number }> {
  const [a, b] = await Promise.all([
    embedText(params.promptA, params.runId),
    embedText(params.promptB, params.runId),
  ]);
  const cosine = cosineSimilarity(a, b);
  return {
    cosine,
    score01: normalizeCosineToScore01(cosine),
  };
}
