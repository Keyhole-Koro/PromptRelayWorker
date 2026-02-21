import { execFile as execFileCallback } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import Bottleneck from "bottleneck";
import { config } from "../config/app-config.js";
import { log } from "../observability/logger.js";
import { withRetry } from "../shared/utils.js";

const execFile = promisify(execFileCallback);

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

const pythonScriptPath = fileURLToPath(new URL("./imagen_generate.py", import.meta.url));

async function callImagenPython(params: {
  prompt: string;
  aspectRatio: "1:1" | "16:9";
  runId: string;
  generation: number;
  candidateIndex: number;
}): Promise<string> {
  const { stdout } = await execFile("python3", [
    pythonScriptPath,
    "--project",
    config.GOOGLE_CLOUD_PROJECT,
    "--region",
    config.IMAGEN_REGION,
    "--model",
    config.IMAGEN_MODEL,
    "--prompt",
    params.prompt,
    "--aspectRatio",
    params.aspectRatio,
  ]);

  const parsed = JSON.parse(stdout) as { bytesBase64Encoded?: string; error?: string };
  if (!parsed.bytesBase64Encoded) {
    throw new Error(parsed.error ?? "python imagen response missing bytesBase64Encoded");
  }
  return parsed.bytesBase64Encoded;
}

export async function generateImagenBase64(params: {
  prompt: string;
  aspectRatio: "1:1" | "16:9";
  runId: string;
  generation: number;
  candidateIndex: number;
}): Promise<string> {
  return withRetry(
    () =>
      imagenLimiter.schedule(async () => {
        log("info", "imagen_request", {
          runId: params.runId,
          generation: params.generation,
          candidateIndex: params.candidateIndex,
        });
        return callImagenPython(params);
      }),
    (error) => {
      const status = parseStatus(error);
      return status !== undefined && isRetriableStatus(status);
    },
    config.MAX_VERTEX_RETRIES,
  );
}

export async function savePngBase64ToDisk(base64Png: string, filePath: string): Promise<void> {
  await mkdir(dirname(filePath), { recursive: true });
  await writeFile(filePath, Buffer.from(base64Png, "base64"));
}
