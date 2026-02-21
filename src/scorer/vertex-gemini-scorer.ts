import { readFile } from "node:fs/promises";
import Bottleneck from "bottleneck";
import { GoogleAuth } from "google-auth-library";
import { config } from "../config/app-config.js";
import type { ScoreBreakdown } from "../domain/types.js";
import { log } from "../observability/logger.js";
import { withRetry } from "../shared/utils.js";
import { cosineSimilarity } from "./score-math.js";

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

function normalizeCosineToScore01(cosine: number): number {
  const normalized = (Math.max(-1, Math.min(1, cosine)) + 1) / 2;
  return Math.max(0, Math.min(1, normalized));
}

function toNumericVector(value: unknown): number[] | undefined {
  if (Array.isArray(value)) {
    const vector = value.filter((item): item is number => typeof item === "number");
    return vector.length > 0 ? vector : undefined;
  }
  const obj = asObject(value);
  const nestedValues = obj.values;
  if (Array.isArray(nestedValues)) {
    const vector = nestedValues.filter((item): item is number => typeof item === "number");
    return vector.length > 0 ? vector : undefined;
  }
  return undefined;
}

function parseMultimodalVectors(json: unknown): { text: number[]; image: number[] } {
  const top = asObject(json);
  const predictions = Array.isArray(top.predictions) ? top.predictions : [];
  const first = asObject(predictions[0]);
  const embeddings = asObject(first.embeddings);

  const textCandidates = [
    first.textEmbedding,
    asObject(first.textEmbedding).values,
    embeddings.textEmbedding,
    asObject(embeddings.textEmbedding).values,
  ];

  const imageCandidates = [
    first.imageEmbedding,
    asObject(first.imageEmbedding).values,
    embeddings.imageEmbedding,
    asObject(embeddings.imageEmbedding).values,
  ];

  const text = textCandidates.map(toNumericVector).find((vector): vector is number[] => Array.isArray(vector));
  const image = imageCandidates.map(toNumericVector).find((vector): vector is number[] => Array.isArray(vector));

  if (!text || !image) {
    throw new Error("multimodal embedding response missing text/image vectors");
  }

  return { text, image };
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

async function scoreWithMultimodalEmbedding(params: {
  prompt: string;
  imageBase64: string;
  runId: string;
  generation: number;
  candidateIndex: number;
}): Promise<ScoreBreakdown> {
  const url = `https://aiplatform.googleapis.com/v1/projects/${config.GOOGLE_CLOUD_PROJECT}/locations/global/publishers/google/models/${config.MULTIMODAL_EMBEDDING_MODEL}:predict`;

  const body = {
    instances: [
      {
        text: params.prompt,
        image: {
          bytesBase64Encoded: params.imageBase64,
        },
      },
    ],
  };

  const json = await withRetry(
    () =>
      geminiLimiter.schedule(async () => {
        log("info", "multimodal_embedding_request", {
          runId: params.runId,
          generation: params.generation,
          candidateIndex: params.candidateIndex,
          model: config.MULTIMODAL_EMBEDDING_MODEL,
        });
        return vertexPost(url, body);
      }),
    (error) => {
      const status = parseStatus(error);
      return status !== undefined && isRetriableStatus(status);
    },
    config.MAX_VERTEX_RETRIES,
  );

  const vectors = parseMultimodalVectors(json);
  const cosine = cosineSimilarity(vectors.text, vectors.image);
  const score01 = normalizeCosineToScore01(cosine);

  return {
    cosine,
    score01,
    flags: [],
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
  void params.mimeType;

  try {
    return await scoreWithMultimodalEmbedding({
      prompt: params.prompt,
      imageBase64: params.imageBase64,
      runId: params.runId,
      generation: params.generation,
      candidateIndex: params.candidateIndex,
    });
  } catch (error) {
    log("warn", "multimodal_embedding_failed", {
      runId: params.runId,
      generation: params.generation,
      candidateIndex: params.candidateIndex,
      error: error instanceof Error ? error.message : String(error),
    });

    return {
      cosine: 0,
      score01: 0,
      flags: ["multimodal_embedding_failed_fallback"],
    };
  }
}
