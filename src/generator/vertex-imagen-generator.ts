import { mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import Bottleneck from "bottleneck";
import { GoogleAuth } from "google-auth-library";
import { config } from "../config/app-config.js";
import { log } from "../observability/logger.js";
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

const imagenLimiter = new Bottleneck({
  maxConcurrent: config.MAX_CONCURRENCY_IMAGEN,
  reservoir: config.RATE_LIMIT_IMAGEN_RPM,
  reservoirRefreshAmount: config.RATE_LIMIT_IMAGEN_RPM,
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

export async function generateImagenBase64(params: {
  prompt: string;
  aspectRatio: "1:1" | "16:9";
  runId: string;
  generation: number;
  candidateIndex: number;
}): Promise<string> {
  const url = `https://${config.IMAGEN_REGION}-aiplatform.googleapis.com/v1/projects/${config.GOOGLE_CLOUD_PROJECT}/locations/${config.IMAGEN_REGION}/publishers/google/models/${config.IMAGEN_MODEL}:predict`;
  const body = {
    instances: [{ prompt: params.prompt }],
    parameters: {
      sampleCount: 1,
      aspectRatio: params.aspectRatio,
    },
  };

  const json = await withRetry(
    () =>
      imagenLimiter.schedule(async () => {
        log("info", "imagen_request", {
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

  const top = asObject(json);
  const predictions = Array.isArray(top.predictions) ? top.predictions : [];
  const first = asObject(predictions[0]);
  const nested = asObject(first.image);
  const bytes =
    (typeof first.bytesBase64Encoded === "string" ? first.bytesBase64Encoded : undefined) ??
    (typeof nested.bytesBase64Encoded === "string" ? nested.bytesBase64Encoded : undefined);

  if (!bytes) {
    throw new Error("imagen response missing bytesBase64Encoded");
  }

  return bytes;
}

export async function savePngBase64ToDisk(base64Png: string, filePath: string): Promise<void> {
  await mkdir(dirname(filePath), { recursive: true });
  await writeFile(filePath, Buffer.from(base64Png, "base64"));
}
