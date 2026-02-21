import express from "express";
import { z } from "zod";
import { config } from "../config/app-config.js";
import { log } from "../observability/logger.js";
import { generateFromPool, prewarmPool } from "../application/pool-service.js";
import { withTimeout } from "../shared/utils.js";

const app = express();
app.use(express.json({ limit: "1mb" }));

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

app.get("/healthz", (_req, res) => {
  res.status(200).type("text/plain; charset=utf-8").send("ok");
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
