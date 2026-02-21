import { join } from "node:path";
import { mkdir, writeFile } from "node:fs/promises";
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
import { scoreImagesByUrl, ScoreHttpError } from "../worker/score/score.js";

const app = express();
app.use(express.json({ limit: "15mb" }));
app.use("/files", express.static(config.LOCAL_DATA_DIR));

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

app.listen(config.PORT, config.HOST, () => {
  log("info", "worker_started", {
    host: config.HOST,
    port: config.PORT,
  });
});
