import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { cosineSimilarity, cosineToScore100 } from "./score.js";
import { MockProvider } from "./mock-provider.js";
import { VertexProvider } from "./vertex-provider.js";
import type {
  GenerateImageRequest,
  GenerateTopicRequest,
  ScoreRequest,
  WorkerProvider,
} from "./types.js";

const HOST = "127.0.0.1";
const PORT = 8091;

const provider: WorkerProvider = process.env.USE_VERTEX === "1" ? new VertexProvider() : new MockProvider();

const server = createServer(async (req, res) => {
  try {
    if (!req.url || !req.method) {
      sendJson(res, 400, { error: "Invalid request" });
      return;
    }

    if (req.method === "GET" && req.url === "/health") {
      sendJson(res, 200, { ok: true, provider: process.env.USE_VERTEX === "1" ? "vertex" : "mock" });
      return;
    }

    if (req.method === "POST" && req.url === "/v1/topic:generate") {
      const body = await readJson<GenerateTopicRequest>(req);
      if (!body.roomCode) {
        sendJson(res, 400, { error: "roomCode is required" });
        return;
      }
      const result = await provider.generateTopic(body);
      sendJson(res, 200, result);
      return;
    }

    if (req.method === "POST" && req.url === "/v1/image:generate") {
      const body = await readJson<GenerateImageRequest>(req);
      if (!body.requestId || !body.prompt || (body.kind !== "player" && body.kind !== "ai") || typeof body.isFinal !== "boolean") {
        sendJson(res, 400, { error: "Invalid image generation payload" });
        return;
      }
      const result = await provider.generateImage(body);
      sendJson(res, 200, result);
      return;
    }

    if (req.method === "POST" && req.url === "/v1/score") {
      const body = await readJson<ScoreRequest>(req);
      if (!body.playerImageUrl || !body.aiImageUrl) {
        sendJson(res, 400, { error: "playerImageUrl and aiImageUrl are required" });
        return;
      }

      const [playerEmbedding, aiEmbedding] = await Promise.all([
        provider.embedImageFromUrl(body.playerImageUrl),
        provider.embedImageFromUrl(body.aiImageUrl),
      ]);

      const cosine = cosineSimilarity(playerEmbedding, aiEmbedding);
      const score100 = cosineToScore100(cosine);
      sendJson(res, 200, { cosine, score100 });
      return;
    }

    sendJson(res, 404, { error: "Not found" });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    sendJson(res, 500, { error: message });
  }
});

server.listen(PORT, HOST, () => {
  console.log(`worker listening on http://${HOST}:${PORT}`);
});

async function readJson<T>(req: IncomingMessage): Promise<T> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  const raw = Buffer.concat(chunks).toString("utf8");
  return (raw ? JSON.parse(raw) : {}) as T;
}

function sendJson(res: ServerResponse, statusCode: number, body: unknown): void {
  res.statusCode = statusCode;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.end(JSON.stringify(body));
}
