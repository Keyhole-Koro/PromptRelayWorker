import { join } from "node:path";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import express from "express";
import { z } from "zod";
import { generateFromPool, prewarmPool } from "../application/pool-service.js";
import { config } from "../config/app-config.js";
import { generateImagenBase64, savePngBase64ToDisk } from "../generator/vertex-imagen-generator.js";
import { filePathToPublicUrl } from "../infra/storage/pool-store.js";
import { log } from "../observability/logger.js";
import { scorePromptAlignment } from "../scorer/prompt-alignment.js";
import { evaluateWithGeminiInline } from "../scorer/vertex-gemini-scorer.js";
import { randomId, withTimeout } from "../shared/utils.js";
import { judgeByTopicSimilarity } from "../worker/judge/judge.js";

const app = express();
app.use(express.json({ limit: "15mb" }));
app.use("/files", express.static(config.LOCAL_DATA_DIR));

function imageMissingSvg(itemId: string): string {
  const escaped = itemId.replace(/[<>&"']/g, "");
  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="768" viewBox="0 0 1024 768">
  <rect width="1024" height="768" fill="#f3f6fb"/>
  <rect x="64" y="64" width="896" height="640" rx="24" fill="#ffffff" stroke="#d4dbe8"/>
  <text x="512" y="340" text-anchor="middle" font-size="40" font-family="sans-serif" fill="#30445f">画像が見つかりません</text>
  <text x="512" y="392" text-anchor="middle" font-size="22" font-family="sans-serif" fill="#5d728c">itemId: ${escaped}</text>
</svg>`;
}

const prewarmSchema = z.object({
  count: z.number().int().positive().max(200).optional(),
  budget: z
    .object({
      generations: z.number().int().positive().max(10).optional(),
      population: z.number().int().positive().max(24).optional(),
      parents: z.number().int().positive().max(10).optional(),
    })
    .optional(),
  aspectRatio: z.enum(["1:1", "16:9"]).optional(),
});

const generateSchema = z.object({
  aspectRatio: z.enum(["1:1", "16:9"]).optional(),
});

const debugGenerateSchema = z.object({
  prompt: z.string().min(1),
  aspectRatio: z.enum(["1:1", "16:9"]).optional(),
});

const promptScoreSchema = z.object({
  promptA: z.string().min(1),
  promptB: z.string().min(1),
});

const playerPreviewSchema = z.object({
  requestId: z.string().min(1).optional(),
  prompt: z.string().min(1),
});

const aiPreviewSchema = z.object({
  requestId: z.string().min(1).optional(),
  themeImageUrl: z.string().url(),
  recentImageUrl: z.string().url(),
  recentPrompt: z.string().min(1),
});

const judgeSchema = z.object({
  topicImageUrl: z.string().min(1),
  playerImageUrl: z.string().min(1),
  aiImageUrl: z.string().min(1),
});

app.post("/playerPreview", async (req, res) => {
  const parsed = playerPreviewSchema.safeParse(req.body ?? {});
  if (!parsed.success) {
    res.status(400).json({ error: "invalid request", details: parsed.error.flatten() });
    return;
  }

  const runId = parsed.data.requestId ?? randomId("playerPreview");
  const itemId = randomId("item");
  const filePath = join(config.LOCAL_DATA_DIR, "preview", "player", itemId, "generated.png");

  try {
    const imageBase64 = await withTimeout(
      generateImagenBase64({
        prompt: parsed.data.prompt,
        aspectRatio: "1:1",
        runId,
        generation: 0,
        candidateIndex: 0,
      }),
      config.REQUEST_TIMEOUT_MS,
      "request timeout",
    );
    await savePngBase64ToDisk(imageBase64, filePath);
    const imageUrl = filePathToPublicUrl(filePath);
    res.status(200).json({ imageUrl, prompt: parsed.data.prompt, requestId: runId });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    log("error", "playerPreview_generate_error", { runId, message });
    res.status(500).json({ error: message, runId });
  }
});

import { updatePromptWithGemini } from "../generator/gemini-prompt-updater.js";

app.post("/aiPreview", async (req, res) => {
  const parsed = aiPreviewSchema.safeParse(req.body ?? {});
  if (!parsed.success) {
    res.status(400).json({ error: "invalid request", details: parsed.error.flatten() });
    return;
  }

  const runId = parsed.data.requestId ?? randomId("aiPreview");
  const itemId = randomId("item");
  const filePath = join(config.LOCAL_DATA_DIR, "preview", "ai", itemId, "generated.png");

  try {
    const newPrompt = await withTimeout(
      updatePromptWithGemini({
        themeImageUrl: parsed.data.themeImageUrl,
        recentImageUrl: parsed.data.recentImageUrl,
        recentPrompt: parsed.data.recentPrompt,
        runId,
      }),
      config.REQUEST_TIMEOUT_MS,
      "gemini prompt update timeout",
    );

    const imageBase64 = await withTimeout(
      generateImagenBase64({
        prompt: newPrompt,
        aspectRatio: "1:1",
        runId,
        generation: 0,
        candidateIndex: 0,
      }),
      config.REQUEST_TIMEOUT_MS,
      "imagen generation timeout",
    );

    await savePngBase64ToDisk(imageBase64, filePath);
    const imageUrl = filePathToPublicUrl(filePath);
    res.status(200).json({ imageUrl, prompt: newPrompt, requestId: runId });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    log("error", "aiPreview_generate_error", { runId, message });
    res.status(500).json({ error: message, runId });
  }
});

app.post("/v1/pool/prewarm", async (req, res) => {
  const parsed = prewarmSchema.safeParse(req.body ?? {});
  if (!parsed.success) {
    res.status(400).json({ error: "invalid request", details: parsed.error.flatten() });
    return;
  }

  try {
    const response = await withTimeout(prewarmPool(parsed.data), config.REQUEST_TIMEOUT_MS, "request timeout");
    res.status(200).json(response);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    log("error", "prewarm_error", { message });
    res.status(500).json({ error: message });
  }
});

app.get("/v1/topic/image/:itemId", async (req, res) => {
  const itemId = String(req.params.itemId ?? "");
  if (!/^[A-Za-z0-9._-]+$/.test(itemId)) {
    res.status(200).setHeader("content-type", "image/svg+xml; charset=utf-8");
    res.setHeader("cache-control", "no-store");
    res.send(imageMissingSvg(itemId));
    return;
  }

  const filePath = join(config.LOCAL_DATA_DIR, config.GCS_PREFIX_USED, itemId, "final.png");
  try {
    const image = await readFile(filePath);
    res.status(200).setHeader("content-type", "image/png");
    res.setHeader("cache-control", "no-store");
    res.send(image);
  } catch {
    res.status(200).setHeader("content-type", "image/svg+xml; charset=utf-8");
    res.setHeader("cache-control", "no-store");
    res.send(imageMissingSvg(itemId));
  }
});

async function handleGenerate(req: express.Request, res: express.Response): Promise<void> {
  const parsed = generateSchema.safeParse(req.body ?? {});
  if (!parsed.success) {
    res.status(400).json({ error: "invalid request", details: parsed.error.flatten() });
    return;
  }

  try {
    const response = await withTimeout(generateFromPool(parsed.data), config.REQUEST_TIMEOUT_MS, "request timeout");
    res.status(200).json(response);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (message === "pool_empty") {
      res
        .status(503)
        .json({ error: "pool empty。/v1/pool/prewarm を実行してください" });
      return;
    }
    log("error", "generate_error", { message });
    res.status(500).json({ error: message });
  }
}

app.post("/generate", (req, res) => {
  void handleGenerate(req, res);
});

app.post("/v1/topic/generate", (req, res) => {
  void handleGenerate(req, res);
});

app.post("/judge", async (req, res) => {
  const parsed = judgeSchema.safeParse(req.body ?? {});
  if (!parsed.success) {
    res.status(400).json({ error: "invalid request", details: parsed.error.flatten() });
    return;
  }

  try {
    const judged = await withTimeout(
      judgeByTopicSimilarity(parsed.data),
      config.REQUEST_TIMEOUT_MS,
      "judge timeout",
    );
    res.status(200).json(judged);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    log("error", "judge_error", { message });
    if (message === "vertex_embedding_unavailable") {
      res.status(503).json({ error: message });
      return;
    }
    res.status(500).json({ error: message });
  }
});

app.listen(config.PORT, config.HOST, () => {
  log("info", "worker_started", {
    host: config.HOST,
    port: config.PORT,
  });
});
