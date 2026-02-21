import { Storage } from "@google-cloud/storage";
import { config } from "../../config/app-config.js";
import { log } from "../../observability/logger.js";
import type { AspectRatio, PoolItemMeta } from "../../domain/types.js";
import { randomId, sleep } from "../../shared/utils.js";

const storage = new Storage({ projectId: config.GOOGLE_CLOUD_PROJECT });
const bucket = storage.bucket(config.GCS_BUCKET);

function trimSlash(s: string): string {
  return s.endsWith("/") ? s.slice(0, -1) : s;
}

function joinPrefix(prefix: string, itemId: string): string {
  return `${trimSlash(prefix)}/${itemId}/`;
}

function gsUri(objectName: string): string {
  return `gs://${config.GCS_BUCKET}/${objectName}`;
}

export async function saveFinalItemToAvailable(meta: PoolItemMeta & { tempImageUri: string }): Promise<{ gcsPrefix: string }> {
  const itemPrefix = joinPrefix(config.GCS_PREFIX_AVAILABLE, meta.itemId);
  const finalPngName = `${itemPrefix}final.png`;
  const metaName = `${itemPrefix}meta.json`;

  const sourcePath = meta.tempImageUri.replace(`gs://${config.GCS_BUCKET}/`, "");
  await bucket.file(sourcePath).copy(bucket.file(finalPngName), {
    preconditionOpts: { ifGenerationMatch: 0 },
  });

  await bucket.file(metaName).save(
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
    {
      resumable: false,
      contentType: "application/json; charset=utf-8",
      preconditionOpts: { ifGenerationMatch: 0 },
    },
  );

  return { gcsPrefix: gsUri(itemPrefix) };
}

type ClaimedItem = {
  itemId: string;
  meta: PoolItemMeta;
  availableMetaName: string;
  availableFinalName: string;
  claimName: string;
};

async function listAvailableMetaFiles(): Promise<string[]> {
  const [files] = await bucket.getFiles({ prefix: config.GCS_PREFIX_AVAILABLE });
  return files
    .map((f) => f.name)
    .filter((name) => name.endsWith("/meta.json"));
}

async function tryClaimMeta(metaName: string): Promise<ClaimedItem | undefined> {
  const itemId = metaName.split("/").slice(-2, -1)[0];
  const availablePrefix = joinPrefix(config.GCS_PREFIX_AVAILABLE, itemId);
  const claimName = `${availablePrefix}.claim`;
  const claimFile = bucket.file(claimName);

  try {
    await claimFile.save(JSON.stringify({ ts: new Date().toISOString(), claimId: randomId("claim") }), {
      resumable: false,
      contentType: "application/json",
      preconditionOpts: { ifGenerationMatch: 0 },
    });
  } catch {
    return undefined;
  }

  try {
    const [raw] = await bucket.file(metaName).download();
    const meta = JSON.parse(raw.toString("utf8")) as PoolItemMeta;
    return {
      itemId,
      meta,
      availableMetaName: metaName,
      availableFinalName: `${availablePrefix}final.png`,
      claimName,
    };
  } catch (error) {
    log("warn", "claim_read_failed", { itemId, error: error instanceof Error ? error.message : String(error) });
    await claimFile.delete({ ignoreNotFound: true });
    return undefined;
  }
}

async function copyDeleteSafe(srcName: string, dstName: string): Promise<void> {
  const srcFile = bucket.file(srcName);
  const [srcMeta] = await srcFile.getMetadata();
  const generation = Number.parseInt(String(srcMeta.generation), 10);

  await srcFile.copy(bucket.file(dstName), {
    preconditionOpts: {
      ifGenerationMatch: 0,
    },
  });

  await srcFile.delete({
    ignoreNotFound: false,
    ifGenerationMatch: generation,
  });
}

export async function acquireAndMoveOneAvailableItem(aspectRatio: AspectRatio): Promise<{
  itemId: string;
  usedImageUri: string;
  usedMetaUri: string;
  meta: PoolItemMeta;
}> {
  for (let attempt = 0; attempt < config.MAX_MOVE_RETRIES; attempt += 1) {
    const metaNames = await listAvailableMetaFiles();
    if (metaNames.length === 0) {
      throw new Error("pool_empty");
    }

    const shuffled = [...metaNames].sort(() => Math.random() - 0.5);
    for (const metaName of shuffled) {
      const claimed = await tryClaimMeta(metaName);
      if (!claimed) {
        continue;
      }

      if (claimed.meta.aspectRatio !== aspectRatio) {
        await bucket.file(claimed.claimName).delete({ ignoreNotFound: true });
        continue;
      }

      const usedPrefix = joinPrefix(config.GCS_PREFIX_USED, claimed.itemId);
      const usedFinalName = `${usedPrefix}final.png`;
      const usedMetaName = `${usedPrefix}meta.json`;

      try {
        await copyDeleteSafe(claimed.availableFinalName, usedFinalName);
        await copyDeleteSafe(claimed.availableMetaName, usedMetaName);
        await bucket.file(claimed.claimName).delete({ ignoreNotFound: true });

        return {
          itemId: claimed.itemId,
          usedImageUri: gsUri(usedFinalName),
          usedMetaUri: gsUri(usedMetaName),
          meta: claimed.meta,
        };
      } catch (error) {
        await bucket.file(claimed.claimName).delete({ ignoreNotFound: true });
        log("warn", "move_conflict_retry", {
          itemId: claimed.itemId,
          error: error instanceof Error ? error.message : String(error),
        });
      }
    }
    await sleep(100 + attempt * 120);
  }

  throw new Error("pool_busy_retry_exhausted");
}

export async function signGsUri(gsPath: string): Promise<string> {
  const objectName = gsPath.replace(`gs://${config.GCS_BUCKET}/`, "");
  const [url] = await bucket.file(objectName).getSignedUrl({
    version: "v4",
    action: "read",
    expires: Date.now() + config.SIGNED_URL_TTL_SEC * 1000,
  });
  return url;
}
