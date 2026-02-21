import { readFile } from "node:fs/promises";
import { config } from "../config/app-config.js";
import type { AspectRatio, EvolveBudget, GenerateRequest, GenerateResponse, PrewarmRequest, PrewarmResponse } from "../domain/types.js";
import { evolveOneItem } from "../generator/evolution.js";
import { acquireAndMoveOneAvailableItem, saveFinalItemToAvailable, signGsUri } from "../infra/storage/pool-store.js";
import { log } from "../observability/logger.js";
import { elapsedMs, mapLimit, nowMs, randomId, withTimeout } from "../shared/utils.js";

function normalizeAspectRatio(input: string | undefined): AspectRatio {
  return input === "16:9" ? "16:9" : config.DEFAULT_ASPECT_RATIO;
}

function normalizeBudget(input?: Partial<EvolveBudget>): EvolveBudget {
  return {
    generations: input?.generations ?? config.EVOLVE_GENERATIONS,
    population: input?.population ?? config.EVOLVE_POPULATION,
    parents: input?.parents ?? config.EVOLVE_PARENTS,
  };
}

export async function prewarmPool(input: PrewarmRequest): Promise<PrewarmResponse> {
  const started = nowMs();
  const runId = randomId("run");
  const count = Math.max(1, Math.min(200, input.count ?? 1));
  const aspectRatio = normalizeAspectRatio(input.aspectRatio);
  const budget = normalizeBudget(input.budget);

  const created: Array<{ itemId: string; gcsPrefix: string }> = [];

  const results = await withTimeout(
    mapLimit(Array.from({ length: count }, (_, i) => i), 2, async (i) => {
      const itemId = randomId(`item-${i}`);
      try {
        const evolved = await evolveOneItem({
          runId,
          itemId,
          budget,
          aspectRatio,
        });
        const saved = await saveFinalItemToAvailable(evolved);
        created.push({ itemId, gcsPrefix: saved.gcsPrefix });
        return { ok: true as const, itemId };
      } catch (error) {
        log("error", "prewarm_item_failed", {
          runId,
          itemId,
          error: error instanceof Error ? error.message : String(error),
        });
        return { ok: false as const, itemId };
      }
    }),
    config.REQUEST_TIMEOUT_MS,
    "prewarm timeout",
  ).catch((error) => {
    log("warn", "prewarm_timeout_or_failure", {
      runId,
      error: error instanceof Error ? error.message : String(error),
    });
    return [] as Array<{ ok: boolean; itemId: string }>;
  });

  const failed = results.filter((r) => !r.ok).length;
  const timingsMs = {
    total: elapsedMs(started),
  };

  log("info", "prewarm_done", {
    runId,
    requested: count,
    created: created.length,
    failed,
    timingsMs,
  });

  return {
    created: created.length,
    items: created,
    timingsMs,
  };
}

export async function generateFromPool(input: GenerateRequest): Promise<GenerateResponse> {
  const started = nowMs();
  const aspectRatio = normalizeAspectRatio(input.aspectRatio);

  const storeStarted = nowMs();
  const moved = await acquireAndMoveOneAvailableItem(aspectRatio);
  const storeMs = elapsedMs(storeStarted);

  const signStarted = nowMs();
  const signedUrl = await signGsUri(moved.usedImageUri);
  const signMs = elapsedMs(signStarted);

  const imageReadStarted = nowMs();
  const imageBuffer = await readFile(moved.usedImageUri);
  const imageBase64 = imageBuffer.toString("base64");
  const imageReadMs = elapsedMs(imageReadStarted);

  return {
    topic: {
      imageUri: moved.usedImageUri,
      signedUrl,
      imageBase64,
      mimeType: "image/png",
      prompt: moved.meta.prompt,
      genome: moved.meta.genome,
      scores: moved.meta.scores,
    },
    itemId: moved.itemId,
    timingsMs: {
      total: elapsedMs(started),
      gcs: storeMs + imageReadMs,
      sign: signMs,
    },
  };
}
