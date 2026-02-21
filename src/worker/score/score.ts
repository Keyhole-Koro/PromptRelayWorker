import { createHash } from "node:crypto";
import { LRUCache } from "lru-cache";
import { ssim as calculateSsim } from "ssim.js";
import { getVertexImageEmbedding } from "../vertex/embeddings.js";
import { cosineSimilarity, normalizeCosine01 } from "./cosine.js";
import { buildHsvHistogram, histogramIntersection } from "./hist.js";
import { deterministicEmbeddingFromImageBytes } from "./mockEmbedding.js";
import { makeImageVariants } from "./preprocess.js";

import { downloadImage, ImageFetchHttpError } from "../../shared/image-fetch.js";

const MAX_PARALLEL_SCORING = 3;
const DEFAULT_CACHE_TTL_MS = 60_000;

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

type ScoreDeps = {
  fetcher?: typeof fetch;
  now?: () => number;
  embedder?: typeof getVertexImageEmbedding;
};

class Semaphore {
  private running = 0;
  private queue: Array<() => void> = [];

  constructor(private readonly max: number) { }

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

const embeddingCache = new LRUCache<string, { vector: number[]; vertexUsed: boolean }>({
  max: 512,
  ttl: cacheTtlMs(),
});

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

export { ImageFetchHttpError as ScoreHttpError };

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
  embeddingCache.clear();
}
