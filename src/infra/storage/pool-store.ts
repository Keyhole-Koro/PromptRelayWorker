import { constants } from "node:fs";
import { copyFile, mkdir, open, readFile, readdir, rename, rm, unlink, writeFile } from "node:fs/promises";
import { basename, join } from "node:path";
import { config } from "../../config/app-config.js";
import { log } from "../../observability/logger.js";
import type { AspectRatio, PoolItemMeta } from "../../domain/types.js";
import { randomId, sleep } from "../../shared/utils.js";

function trimSlash(s: string): string {
  return s.endsWith("/") ? s.slice(0, -1) : s;
}

function availableRoot(): string {
  return join(config.LOCAL_DATA_DIR, trimSlash(config.GCS_PREFIX_AVAILABLE));
}

function usedRoot(): string {
  return join(config.LOCAL_DATA_DIR, trimSlash(config.GCS_PREFIX_USED));
}

async function ensureDataRoots(): Promise<void> {
  await Promise.all([
    mkdir(availableRoot(), { recursive: true }),
    mkdir(usedRoot(), { recursive: true }),
    mkdir(join(config.LOCAL_DATA_DIR, "debug", "generated"), { recursive: true }),
  ]);
}

function itemDir(root: string, itemId: string): string {
  return join(root, itemId);
}

export function filePathToPublicUrl(filePath: string): string {
  const rel = filePath.replace(`${config.LOCAL_DATA_DIR}/`, "");
  return `/files/${rel.split("\\").join("/")}`;
}

export async function saveFinalItemToAvailable(meta: PoolItemMeta & { tempImagePath: string }): Promise<{ gcsPrefix: string }> {
  await ensureDataRoots();
  const dir = itemDir(availableRoot(), meta.itemId);
  await mkdir(dir, { recursive: true });

  const finalPng = join(dir, "final.png");
  const metaJson = join(dir, "meta.json");

  await copyFile(meta.tempImagePath, finalPng, constants.COPYFILE_EXCL);
  await writeFile(
    metaJson,
    JSON.stringify(
      {
        itemId: meta.itemId,
        runId: meta.runId,
        createdAt: meta.createdAt,
        aspectRatio: meta.aspectRatio,
        prompt: meta.prompt,
        genome: meta.genome,
        scores: meta.scores,
        generation: meta.generation,
      },
      null,
      2,
    ),
    "utf8",
  );

  return { gcsPrefix: `file://${dir}/` };
}

type ClaimedItem = {
  itemId: string;
  meta: PoolItemMeta;
  dirPath: string;
  claimPath: string;
};

async function listItemDirs(root: string): Promise<string[]> {
  await mkdir(root, { recursive: true });
  const entries = await readdir(root, { withFileTypes: true });
  return entries.filter((e) => e.isDirectory()).map((e) => join(root, e.name));
}

async function tryClaimItem(dirPath: string): Promise<ClaimedItem | undefined> {
  const itemId = basename(dirPath);
  const claimPath = join(dirPath, ".claim");

  try {
    const handle = await open(claimPath, "wx");
    await handle.writeFile(JSON.stringify({ ts: new Date().toISOString(), claimId: randomId("claim") }), "utf8");
    await handle.close();
  } catch {
    return undefined;
  }

  try {
    const metaRaw = await readFile(join(dirPath, "meta.json"), "utf8");
    const meta = JSON.parse(metaRaw) as PoolItemMeta;
    return { itemId, meta, dirPath, claimPath };
  } catch (error) {
    log("warn", "claim_read_failed", { itemId, error: error instanceof Error ? error.message : String(error) });
    await unlink(claimPath).catch(() => undefined);
    return undefined;
  }
}

export async function acquireAndMoveOneAvailableItem(aspectRatio: AspectRatio): Promise<{
  itemId: string;
  usedImageUri: string;
  usedMetaUri: string;
  meta: PoolItemMeta;
}> {
  await ensureDataRoots();

  for (let attempt = 0; attempt < config.MAX_MOVE_RETRIES; attempt += 1) {
    const dirs = await listItemDirs(availableRoot());
    if (dirs.length === 0) {
      throw new Error("pool_empty");
    }

    const shuffled = [...dirs].sort(() => Math.random() - 0.5);
    for (const dirPath of shuffled) {
      const claimed = await tryClaimItem(dirPath);
      if (!claimed) {
        continue;
      }

      if (claimed.meta.aspectRatio !== aspectRatio) {
        await unlink(claimed.claimPath).catch(() => undefined);
        continue;
      }

      const targetDir = itemDir(usedRoot(), claimed.itemId);
      try {
        await rename(claimed.dirPath, targetDir);
      } catch (error) {
        await unlink(claimed.claimPath).catch(() => undefined);
        log("warn", "move_conflict_retry", {
          itemId: claimed.itemId,
          error: error instanceof Error ? error.message : String(error),
        });
        continue;
      }

      await unlink(join(targetDir, ".claim")).catch(() => undefined);

      const usedImagePath = join(targetDir, "final.png");
      const usedMetaPath = join(targetDir, "meta.json");
      return {
        itemId: claimed.itemId,
        usedImageUri: usedImagePath,
        usedMetaUri: usedMetaPath,
        meta: claimed.meta,
      };
    }

    await sleep(100 + attempt * 120);
  }

  throw new Error("pool_busy_retry_exhausted");
}

export async function signGsUri(pathOrUri: string): Promise<string> {
  if (pathOrUri.startsWith("/")) {
    return filePathToPublicUrl(pathOrUri);
  }
  if (pathOrUri.startsWith("file://")) {
    return filePathToPublicUrl(pathOrUri.replace("file://", ""));
  }
  return pathOrUri;
}

export async function clearDebugGenerated(): Promise<void> {
  await rm(join(config.LOCAL_DATA_DIR, "debug", "generated"), { recursive: true, force: true });
  await mkdir(join(config.LOCAL_DATA_DIR, "debug", "generated"), { recursive: true });
}
