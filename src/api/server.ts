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

const previewSchema = z.object({
  requestId: z.string().min(1),
  kind: z.enum(["player", "ai"]),
  prompt: z.string().min(1),
  isFinal: z.boolean().optional(),
});

const debugScoreSchema = z.object({
  prompt: z.string().min(1),
  imageBase64: z.string().min(1),
  mimeType: z.string().min(1),
});

const previewSeqByKind: Record<"player" | "ai", number> = {
  player: 0,
  ai: 0,
};

function escapeXml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;")
    .replaceAll("'", "&apos;");
}

function mockPreviewSvg(params: {
  kind: "player" | "ai";
  seq: number;
  prompt: string;
  requestId: string;
  isFinal: boolean;
}): string {
  const bg = params.kind === "player" ? "#0ea5e9" : "#9333ea";
  const panel = params.kind === "player" ? "#082f49" : "#3b0764";
  const promptPreview = params.prompt.slice(0, 160);
  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 1024 1024">
  <rect width="1024" height="1024" fill="${bg}" />
  <rect x="64" y="64" width="896" height="896" rx="28" fill="${panel}" />
  <text x="96" y="150" fill="#ffffff" font-size="48" font-family="Arial, sans-serif" font-weight="700">/preview mock (${params.kind})</text>
  <text x="96" y="220" fill="#dbeafe" font-size="34" font-family="Arial, sans-serif">seq: ${params.seq}</text>
  <text x="96" y="270" fill="#dbeafe" font-size="28" font-family="Arial, sans-serif">requestId: ${escapeXml(params.requestId)}</text>
  <text x="96" y="320" fill="#dbeafe" font-size="28" font-family="Arial, sans-serif">final: ${params.isFinal ? "true" : "false"}</text>
  <foreignObject x="96" y="380" width="832" height="540">
    <div xmlns="http://www.w3.org/1999/xhtml" style="color:#ffffff;font-size:30px;line-height:1.4;font-family:Arial,sans-serif;white-space:pre-wrap;word-break:break-word;">
      ${escapeXml(promptPreview)}
    </div>
  </foreignObject>
</svg>`;
}

function workerPageHtml(): string {
  return `<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Worker Visual Debug</title>
  <style>
    :root {
      --bg: #f4f6fb;
      --card: #ffffff;
      --ink: #10203a;
      --accent: #0f5ef7;
      --muted: #60708a;
      --line: #d6deea;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--ink);
      background: radial-gradient(1200px 700px at 20% -20%, #dce7ff 0%, var(--bg) 50%);
    }
    .wrap {
      max-width: 980px;
      margin: 24px auto;
      padding: 0 16px 32px;
    }
    h1 { margin: 8px 0 20px; font-size: 30px; }
    .grid {
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 10px 22px rgba(16, 32, 58, 0.08);
    }
    label { display: block; font-weight: 600; margin-top: 8px; }
    input, textarea, select, button {
      width: 100%;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid var(--line);
      margin-top: 6px;
      font: inherit;
      background: #fff;
    }
    textarea { min-height: 88px; resize: vertical; }
    button {
      margin-top: 12px;
      cursor: pointer;
      border: 0;
      background: linear-gradient(120deg, #0f5ef7, #2782ff);
      color: #fff;
      font-weight: 700;
    }
    .note { color: var(--muted); font-size: 13px; }
    .img {
      margin-top: 10px;
      width: 100%;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: #f8fbff;
    }
    pre {
      background: #0d1a2e;
      color: #d8e6ff;
      padding: 10px;
      border-radius: 10px;
      overflow: auto;
      min-height: 120px;
      font-size: 12px;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Worker Visual Debug (/worker)</h1>
    <div class="grid">
      <section class="card">
        <h2>1) 画像×テキスト類似度 (Multimodal Embedding)</h2>
        <p class="note">画像ファイルとプロンプトを送信して cosine/score01 を確認します。</p>
        <label>画像</label>
        <input id="scoreFile" type="file" accept="image/*" />
        <label>プロンプト</label>
        <textarea id="scorePrompt" placeholder="A tiny robot chef in a ramen shop"></textarea>
        <button id="scoreBtn" type="button">スコア実行</button>
        <img id="scorePreview" class="img" alt="uploaded preview" />
        <pre id="scoreOut"></pre>
      </section>

      <section class="card">
        <h2>2) 画像生成 (Imagen Generator)</h2>
        <p class="note">プロンプトから1枚生成してローカルdisk保存・表示します。</p>
        <label>プロンプト</label>
        <textarea id="genPrompt" placeholder="A cat detective in Tokyo rain"></textarea>
        <label>アスペクト比</label>
        <select id="genAspect">
          <option value="1:1">1:1</option>
          <option value="16:9">16:9</option>
        </select>
        <button id="genBtn" type="button">画像生成</button>
        <img id="genImg" class="img" alt="generated image" />
        <pre id="genOut"></pre>
      </section>

      <section class="card">
        <h2>3) プール消費 (/generate)</h2>
        <p class="note">available から1つ取り出し、used に移動した結果を確認します。</p>
        <label>アスペクト比</label>
        <select id="poolAspect">
          <option value="1:1">1:1</option>
          <option value="16:9">16:9</option>
        </select>
        <button id="poolBtn" type="button">/generate 実行</button>
        <img id="poolImg" class="img" alt="pooled image" />
        <pre id="poolOut"></pre>
      </section>

      <section class="card">
        <h2>4) プロンプト類似度 (Embedding)</h2>
        <p class="note">2つのプロンプトを embedding 化して cosine 類似度を返します。</p>
        <label>Prompt A</label>
        <textarea id="promptA" placeholder="A cat detective in Tokyo rain"></textarea>
        <label>Prompt B</label>
        <textarea id="promptB" placeholder="A detective cat in rainy Tokyo street"></textarea>
        <button id="promptBtn" type="button">類似度を計算</button>
        <pre id="promptOut"></pre>
      </section>
    </div>
  </div>

  <script>
    const byId = (id) => document.getElementById(id);

    async function fileToPayload(file) {
      const dataUrl = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });
      const raw = String(dataUrl);
      const parts = raw.split(',');
      if (parts.length < 2) throw new Error('invalid data url');
      const meta = parts[0];
      const imageBase64 = parts[1];
      const m = /data:(.*);base64/.exec(meta);
      const mimeType = m ? m[1] : 'image/png';
      return { imageBase64, mimeType, preview: raw };
    }

    byId('scoreBtn').addEventListener('click', async () => {
      const out = byId('scoreOut');
      const img = byId('scorePreview');
      out.textContent = 'running...';
      try {
        const file = byId('scoreFile').files[0];
        if (!file) throw new Error('画像ファイルを選択してください');
        const payload = await fileToPayload(file);
        img.src = payload.preview;
        const res = await fetch('/v1/debug/score', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({
            prompt: byId('scorePrompt').value || 'test prompt',
            imageBase64: payload.imageBase64,
            mimeType: payload.mimeType,
          }),
        });
        const json = await res.json();
        out.textContent = JSON.stringify(json, null, 2);
      } catch (e) {
        out.textContent = String(e);
      }
    });

    byId('genBtn').addEventListener('click', async () => {
      const out = byId('genOut');
      const img = byId('genImg');
      out.textContent = 'running...';
      try {
        const res = await fetch('/v1/debug/generate', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({
            prompt: byId('genPrompt').value || 'test prompt',
            aspectRatio: byId('genAspect').value,
          }),
        });
        const json = await res.json();
        if (json?.signedUrl) img.src = json.signedUrl;
        out.textContent = JSON.stringify(json, null, 2);
      } catch (e) {
        out.textContent = String(e);
      }
    });

    byId('poolBtn').addEventListener('click', async () => {
      const out = byId('poolOut');
      const img = byId('poolImg');
      out.textContent = 'running...';
      try {
        const res = await fetch('/generate', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ aspectRatio: byId('poolAspect').value }),
        });
        const json = await res.json();
        if (json?.topic?.signedUrl) img.src = json.topic.signedUrl;
        out.textContent = JSON.stringify(json, null, 2);
      } catch (e) {
        out.textContent = String(e);
      }
    });

    byId('promptBtn').addEventListener('click', async () => {
      const out = byId('promptOut');
      out.textContent = 'running...';
      try {
        const res = await fetch('/v1/debug/prompt-score', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({
            promptA: byId('promptA').value || 'test prompt a',
            promptB: byId('promptB').value || 'test prompt b',
          }),
        });
        const json = await res.json();
        out.textContent = JSON.stringify(json, null, 2);
      } catch (e) {
        out.textContent = String(e);
      }
    });
  </script>
</body>
</html>`;
}

app.get("/healthz", (_req, res) => {
  res.status(200).type("text/plain; charset=utf-8").send("ok");
});

app.get("/worker", (_req, res) => {
  res.status(200).type("text/html; charset=utf-8").send(workerPageHtml());
});

app.get("/worker/", (_req, res) => {
  res.status(200).type("text/html; charset=utf-8").send(workerPageHtml());
});

app.post("/v1/debug/score", async (req, res) => {
  const parsed = debugScoreSchema.safeParse(req.body ?? {});
  if (!parsed.success) {
    res.status(400).json({ error: "invalid request", details: parsed.error.flatten() });
    return;
  }

  const runId = randomId("debug-score");
  try {
    const scores = await withTimeout(
      evaluateWithGeminiInline({
        imageBase64: parsed.data.imageBase64,
        mimeType: parsed.data.mimeType,
        prompt: parsed.data.prompt,
        runId,
        generation: 0,
        candidateIndex: 0,
      }),
      config.REQUEST_TIMEOUT_MS,
      "request timeout",
    );
    res.status(200).json({ runId, prompt: parsed.data.prompt, scores });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    log("error", "debug_score_error", { runId, message });
    res.status(500).json({ error: message, runId });
  }
});

app.post("/v1/debug/prompt-score", async (req, res) => {
  const parsed = promptScoreSchema.safeParse(req.body ?? {});
  if (!parsed.success) {
    res.status(400).json({ error: "invalid request", details: parsed.error.flatten() });
    return;
  }

  const runId = randomId("debug-prompt-score");
  try {
    const alignment = await withTimeout(
      scorePromptAlignment({
        promptA: parsed.data.promptA,
        promptB: parsed.data.promptB,
        runId,
      }),
      config.REQUEST_TIMEOUT_MS,
      "request timeout",
    );
    res.status(200).json({
      runId,
      promptA: parsed.data.promptA,
      promptB: parsed.data.promptB,
      ...alignment,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    log("error", "debug_prompt_score_error", { runId, message });
    res.status(500).json({ error: message, runId });
  }
});

app.post("/v1/debug/generate", async (req, res) => {
  const parsed = debugGenerateSchema.safeParse(req.body ?? {});
  if (!parsed.success) {
    res.status(400).json({ error: "invalid request", details: parsed.error.flatten() });
    return;
  }

  const runId = randomId("debug-generate");
  const itemId = randomId("item");
  const aspectRatio = parsed.data.aspectRatio ?? "1:1";
  const filePath = join(config.LOCAL_DATA_DIR, "debug", "generated", itemId, "generated.png");

  try {
    const imageBase64 = await withTimeout(
      generateImagenBase64({
        prompt: parsed.data.prompt,
        aspectRatio,
        runId,
        generation: 0,
        candidateIndex: 0,
      }),
      config.REQUEST_TIMEOUT_MS,
      "request timeout",
    );
    await savePngBase64ToDisk(imageBase64, filePath);
    const imageUri = `file://${filePath}`;
    const signedUrl = filePathToPublicUrl(filePath);
    res.status(200).json({ runId, itemId, imageUri, signedUrl });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    log("error", "debug_generate_error", { runId, message });
    res.status(500).json({ error: message, runId });
  }
});

app.post("/preview", async (req, res) => {
  const parsed = previewSchema.safeParse(req.body ?? {});
  if (!parsed.success) {
    res.status(400).json({ error: "invalid request", details: parsed.error.flatten() });
    return;
  }

  const runId = randomId("preview");
  const { kind, prompt, requestId, isFinal = false } = parsed.data;
  const seq = ++previewSeqByKind[kind];
  const filePath = join(config.LOCAL_DATA_DIR, "preview", kind, `${runId}-${seq}.svg`);

  try {
    await mkdir(join(config.LOCAL_DATA_DIR, "preview", kind), { recursive: true });
    const svg = mockPreviewSvg({ kind, seq, prompt, requestId, isFinal });
    await writeFile(filePath, svg, "utf8");
    const imageUrl = filePathToPublicUrl(filePath);
    res.status(200).json({ imageUrl, requestId });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    log("error", "preview_generate_error", { runId, message, kind });
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
