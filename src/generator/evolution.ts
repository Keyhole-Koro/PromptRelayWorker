import { execFile as execFileCallback } from "node:child_process";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { config } from "../config/app-config.js";
import type { AspectRatio, EvolveBudget, PoolItemMeta } from "../domain/types.js";
import { log } from "../observability/logger.js";

const execFile = promisify(execFileCallback);
const pythonScriptPath = fileURLToPath(new URL("./evolve_item.py", import.meta.url));

export async function evolveOneItem(params: {
  runId: string;
  itemId: string;
  budget: EvolveBudget;
  aspectRatio: AspectRatio;
}): Promise<PoolItemMeta & { tempImagePath: string }> {
  let stdout = "";
  let stderr = "";
  try {
    const output = await execFile("python3", [
      pythonScriptPath,
      "--project",
      config.GOOGLE_CLOUD_PROJECT,
      "--imagenRegion",
      config.IMAGEN_REGION,
      "--imagenModel",
      config.IMAGEN_MODEL,
      "--geminiModel",
      config.GEMINI_MODEL,
      "--localDataDir",
      config.LOCAL_DATA_DIR,
      "--runId",
      params.runId,
      "--itemId",
      params.itemId,
      "--aspectRatio",
      params.aspectRatio,
      "--generations",
      String(params.budget.generations),
      "--population",
      String(params.budget.population),
      "--parents",
      String(params.budget.parents),
      "--maxVertexRetries",
      String(config.MAX_VERTEX_RETRIES),
      "--settingsPath",
      config.EVOLVE_SETTINGS_PATH,
    ]);
    stdout = output.stdout;
    stderr = output.stderr;
  } catch (error) {
    const err = error as { stdout?: string; stderr?: string; message?: string };
    stderr = err.stderr ?? "";
    const message = err.message ?? "python evolve execution failed";
    if (stderr.trim()) {
      for (const line of stderr.split(/\r?\n/)) {
        const trimmed = line.trim();
        if (!trimmed) {
          continue;
        }
        log("warn", "evolve_item_progress", { runId: params.runId, itemId: params.itemId, line: trimmed });
      }
    }
    throw new Error(message);
  }

  if (stderr.trim()) {
    for (const line of stderr.split(/\r?\n/)) {
      const trimmed = line.trim();
      if (!trimmed) {
        continue;
      }
      log("info", "evolve_item_progress", { runId: params.runId, itemId: params.itemId, line: trimmed });
    }
  }

  const parsed = JSON.parse(stdout) as PoolItemMeta & { tempImagePath: string };
  return parsed;
}
