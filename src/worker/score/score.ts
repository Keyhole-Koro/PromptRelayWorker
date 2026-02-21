import { createHash } from "node:crypto";
import { LRUCache } from "lru-cache";
import { ssim as calculateSsim } from "ssim.js";
import { getVertexImageEmbedding } from "../vertex/embeddings.js";
import { cosineSimilarity, normalizeCosine01 } from "./cosine.js";
import { buildHsvHistogram, histogramIntersection } from "./hist.js";
import { deterministicEmbeddingFromImageBytes } from "./mockEmbedding.js";
import { makeImageVariants } from "./preprocess.js";

const IMAGE_TIMEOUT_MS = 5_000;
const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
const MAX_PARALLEL_SCORING = 3;
const DEFAULT_CACHE_TTL_MS = 60_000;

export class ScoreHttpError extends Error {
  readonly status: number;
  readonly code: string;

  constructor(status: number, code: string, message: string) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

export type ScoreBreakdown = {
  semantic: number;
  composition: number;
  color: number;
  detail: number;
};

export type ScoreResult = {
  score100: number;
  cosine: number;
  breakdown: ScoreBreakdown;
  debug: {
    vertexUsed: boolean;
    timingsMs: {
      total: number;
      download: number;
      preprocess: number;
      semantic: number;
      composition: number;
      color: number;
      detail: number;
    };
  };
};

export type ScoreRequest = {
  playerImageUrl: string;
  aiImageUrl: string;
  promptText?: string;
};

type DownloadedImage = {
  buffer: Buffer;
  contentType: string;
};

type ScoreDeps = {
  fetcher?: typeof fetch;
  now?: () => number;
  embedder?: typeof getVertexImageEmbedding;
};

class Semaphore {
  private running = 0;
  private queue: Array<() => void> = [];

  constructor(private readonly max: number) {}

  async acquire(): Promise<void> {
    if (this.running < this.max) {
      this.running += 1;
      return;
    }

    await new Promise<void>((resolve) => {
      this.queue.push(() => {
        this.running += 1;
        resolve();
      });
    });
  }

  release(): void {
    this.running = Math.max(0, this.running - 1);
    const next = this.queue.shift();
    if (next) {
      next();
    }
  }
}

const scoreSemaphore = new Semaphore(MAX_PARALLEL_SCORING);

function cacheTtlMs(): number {
  const fromEnv = Number.parseInt(process.env.SCORE_CACHE_TTL_MS ?? "", 10);
  if (Number.isFinite(fromEnv) && fromEnv > 0) {
    return fromEnv;
  }
  return DEFAULT_CACHE_TTL_MS;
}

const imageCache = new LRUCache<string, DownloadedImage>({
  max: 128,
  ttl: cacheTtlMs(),
});

const embeddingCache = new LRUCache<string, { vector: number[]; vertexUsed: boolean }>({
  max: 512,
  ttl: cacheTtlMs(),
});

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function isHttpUrl(input: string): boolean {
  try {
    const url = new URL(input);
    return url.protocol === "http:" || url.protocol === "https:";
  } catch {
    return false;
  }
}

async function fetchWithTimeout(fetcher: typeof fetch, url: string, timeoutMs: number): Promise<Response> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    return await fetcher(url, { method: "GET", signal: ctrl.signal });
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      throw new ScoreHttpError(408, "IMAGE_TIMEOUT", `image download timeout: ${url}`);
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

async function readBodyWithLimit(res: Response, maxBytes: number): Promise<Buffer> {
  const reader = res.body?.getReader();
  if (!reader) {
    throw new ScoreHttpError(502, "IMAGE_STREAM_ERROR", "image response has no readable stream");
  }

  const chunks: Uint8Array[] = [];
  let total = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    if (!value) {
      continue;
    }
    total += value.byteLength;
    if (total > maxBytes) {
      throw new ScoreHttpError(413, "IMAGE_TOO_LARGE", `image exceeds ${maxBytes} bytes`);
    }
    chunks.push(value);
  }

  return Buffer.concat(chunks.map((c) => Buffer.from(c)));
}

async function downloadImage(url: string, fetcher: typeof fetch): Promise<DownloadedImage> {
  const cached = imageCache.get(url);
  if (cached) {
    return cached;
  }

  if (!isHttpUrl(url)) {
    throw new ScoreHttpError(400, "INVALID_IMAGE_URL", `invalid image url: ${url}`);
  }

  const res = await fetchWithTimeout(fetcher, url, IMAGE_TIMEOUT_MS);
  if (!res.ok) {
    throw new ScoreHttpError(400, "IMAGE_FETCH_FAILED", `failed to fetch image (${res.status})`);
  }

  const contentType = res.headers.get("content-type") ?? "";
  if (!contentType.toLowerCase().startsWith("image/")) {
    throw new ScoreHttpError(400, "INVALID_IMAGE_CONTENT_TYPE", `content-type is not image: ${contentType}`);
  }

  const contentLength = Number.parseInt(res.headers.get("content-length") ?? "", 10);
  if (Number.isFinite(contentLength) && contentLength > MAX_IMAGE_BYTES) {
    throw new ScoreHttpError(413, "IMAGE_TOO_LARGE", `image exceeds ${MAX_IMAGE_BYTES} bytes`);
  }

  const buffer = await readBodyWithLimit(res, MAX_IMAGE_BYTES);
  const result: DownloadedImage = { buffer, contentType };
  imageCache.set(url, result);
  return result;
}

function cacheKeyForEmbedding(imagePng: Buffer, promptText?: string): string {
  const digest = createHash("sha256").update(imagePng).digest("hex");
  const prompt = (promptText ?? "").trim();
  return `${digest}:${prompt}`;
}

async function imageEmbedding(
  imagePng: Buffer,
  sourceBytes: Buffer,
  promptText: string | undefined,
  fetcher: typeof fetch,
  embedder: typeof getVertexImageEmbedding,
): Promise<{ vector: number[]; vertexUsed: boolean }> {
  const key = cacheKeyForEmbedding(imagePng, promptText);
  const cached = embeddingCache.get(key);
  if (cached) {
    return cached;
  }

  const vertexVector = await embedder({
    imagePng,
    promptText,
    fetcher,
  });

  if (vertexVector) {
    const value = { vector: vertexVector, vertexUsed: true };
    embeddingCache.set(key, value);
    return value;
  }

  const mock = deterministicEmbeddingFromImageBytes(sourceBytes);
  const value = { vector: mock, vertexUsed: false };
  embeddingCache.set(key, value);
  return value;
}

function ssimScore(a: { width: number; height: number; data: Uint8ClampedArray }, b: { width: number; height: number; data: Uint8ClampedArray }): number {
  const { mssim } = calculateSsim(
    { width: a.width, height: a.height, data: a.data },
    { width: b.width, height: b.height, data: b.data },
  );
  return clamp01(mssim);
}

function weightedTotal(breakdown: ScoreBreakdown): number {
  return clamp01(
    breakdown.semantic * 0.5 +
      breakdown.composition * 0.2 +
      breakdown.color * 0.15 +
      breakdown.detail * 0.15,
  );
}

export async function scoreImagesByUrl(input: ScoreRequest, deps: ScoreDeps = {}): Promise<ScoreResult> {
  const fetcher = deps.fetcher ?? fetch;
  const embedder = deps.embedder ?? getVertexImageEmbedding;
  const now = deps.now ?? (() => Date.now());

  const started = now();
  const timingsMs = {
    total: 0,
    download: 0,
    preprocess: 0,
    semantic: 0,
    composition: 0,
    color: 0,
    detail: 0,
  };

  await scoreSemaphore.acquire();
  try {
    const d0 = now();
    const [playerImg, aiImg] = await Promise.all([
      downloadImage(input.playerImageUrl, fetcher),
      downloadImage(input.aiImageUrl, fetcher),
    ]);
    timingsMs.download = now() - d0;

    const p0 = now();
    const [player, ai] = await Promise.all([
      makeImageVariants(playerImg.buffer),
      makeImageVariants(aiImg.buffer),
    ]);
    timingsMs.preprocess = now() - p0;

    const s0 = now();
    const [playerEmb, aiEmb] = await Promise.all([
      imageEmbedding(player.embeddingPng, playerImg.buffer, input.promptText, fetcher, embedder),
      imageEmbedding(ai.embeddingPng, aiImg.buffer, input.promptText, fetcher, embedder),
    ]);
    timingsMs.semantic = now() - s0;

    const cosine = cosineSimilarity(playerEmb.vector, aiEmb.vector);
    const semantic = normalizeCosine01(cosine);
    const vertexUsed = playerEmb.vertexUsed && aiEmb.vertexUsed;

    const c0 = now();
    const composition = ssimScore(player.composition, ai.composition);
    timingsMs.composition = now() - c0;

    const d1 = now();
    const detail = ssimScore(player.detail, ai.detail);
    timingsMs.detail = now() - d1;

    const col0 = now();
    const colorA = buildHsvHistogram(player.color.data);
    const colorB = buildHsvHistogram(ai.color.data);
    const color = histogramIntersection(colorA, colorB);
    timingsMs.color = now() - col0;

    const breakdown: ScoreBreakdown = {
      semantic,
      composition,
      color,
      detail,
    };

    const total = weightedTotal(breakdown);
    const score100 = Math.round(total * 100);

    timingsMs.total = now() - started;

    return {
      score100,
      cosine,
      breakdown,
      debug: {
        vertexUsed,
        timingsMs,
      },
    };
  } finally {
    scoreSemaphore.release();
  }
}

export function __resetScoreCachesForTest(): void {
  imageCache.clear();
  embeddingCache.clear();
}
