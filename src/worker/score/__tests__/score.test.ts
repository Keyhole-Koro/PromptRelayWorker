import { createHash } from "node:crypto";
import { describe, expect, test, beforeEach, vi } from "vitest";
import sharp from "sharp";
import { ScoreHttpError, __resetScoreCachesForTest, scoreImagesByUrl } from "../score.js";

async function makePng(options: { r: number; g: number; b: number; stripe?: boolean }): Promise<Buffer> {
  const base = sharp({
    create: {
      width: 256,
      height: 256,
      channels: 3,
      background: { r: options.r, g: options.g, b: options.b },
    },
  });

  if (!options.stripe) {
    return base.png().toBuffer();
  }

  const stripe = await sharp({
    create: {
      width: 256,
      height: 48,
      channels: 3,
      background: { r: 255, g: 255, b: 255 },
    },
  })
    .png()
    .toBuffer();

  return base
    .composite([{ input: stripe, top: 104, left: 0 }])
    .png()
    .toBuffer();
}

function imageFetcher(imageMap: Record<string, Buffer>, delayMs = 0): typeof fetch {
  return vi.fn(async (input: string | URL | Request) => {
    const url = String(input);
    if (delayMs > 0) {
      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
    const img = imageMap[url];
    if (!img) {
      return new Response("not found", { status: 404 });
    }
    return imageResponse(img);
  }) as unknown as typeof fetch;
}

function imageResponse(img: Buffer): Response {
  return new Response(new Uint8Array(img), {
    status: 200,
    headers: {
      "content-type": "image/png",
      "content-length": String(img.length),
    },
  });
}

function hash(buffer: Buffer): string {
  return createHash("sha256").update(buffer).digest("hex");
}

describe("scoreImagesByUrl", () => {
  beforeEach(() => {
    __resetScoreCachesForTest();
    delete process.env.VERTEX_ENABLED;
  });

  test("same image yields high score", async () => {
    const img = await makePng({ r: 30, g: 100, b: 200, stripe: true });
    const fetcher = imageFetcher({ "https://x/player.png": img, "https://x/ai.png": img });

    const result = await scoreImagesByUrl(
      {
        playerImageUrl: "https://x/player.png",
        aiImageUrl: "https://x/ai.png",
        promptText: "space cat",
      },
      {
        fetcher,
        embedder: async () => [1, 0, 0],
      },
    );

    expect(result.score100).toBeGreaterThanOrEqual(90);
    expect(result.debug.vertexUsed).toBe(true);
  });

  test("different images yields relatively low score", async () => {
    const player = await makePng({ r: 250, g: 20, b: 20, stripe: true });
    const ai = await makePng({ r: 20, g: 20, b: 250, stripe: false });
    const fetcher = imageFetcher({ "https://x/player.png": player, "https://x/ai.png": ai });

    const map = new Map<string, number[]>([
      [hash(await sharp(player).resize(512, 512, { fit: "contain", background: { r: 0, g: 0, b: 0, alpha: 1 } }).png().toBuffer()), [1, 0]],
      [hash(await sharp(ai).resize(512, 512, { fit: "contain", background: { r: 0, g: 0, b: 0, alpha: 1 } }).png().toBuffer()), [-1, 0]],
    ]);

    const result = await scoreImagesByUrl(
      {
        playerImageUrl: "https://x/player.png",
        aiImageUrl: "https://x/ai.png",
        promptText: "space cat",
      },
      {
        fetcher,
        embedder: async ({ imagePng }) => map.get(hash(imagePng)) ?? [0, 1],
      },
    );

    expect(result.score100).toBeLessThanOrEqual(60);
  });

  test("falls back to deterministic mock embedding when vertex is disabled", async () => {
    process.env.VERTEX_ENABLED = "0";
    const player = await makePng({ r: 120, g: 80, b: 40, stripe: true });
    const ai = await makePng({ r: 80, g: 120, b: 40, stripe: true });
    const fetcher = imageFetcher({ "https://x/player.png": player, "https://x/ai.png": ai });

    const result = await scoreImagesByUrl(
      {
        playerImageUrl: "https://x/player.png",
        aiImageUrl: "https://x/ai.png",
        promptText: "space cat",
      },
      {
        fetcher,
        embedder: async () => undefined,
      },
    );

    expect(result.score100).toBeGreaterThanOrEqual(0);
    expect(result.debug.vertexUsed).toBe(false);
  });

  test("404 image returns 4xx error without crashing", async () => {
    const player = await makePng({ r: 120, g: 80, b: 40, stripe: true });
    const fetcher = imageFetcher({ "https://x/player.png": player });

    await expect(
      scoreImagesByUrl(
        {
          playerImageUrl: "https://x/player.png",
          aiImageUrl: "https://x/missing.png",
        },
        {
          fetcher,
          embedder: async () => [0, 1],
        },
      ),
    ).rejects.toMatchObject({
      status: 400,
      code: "IMAGE_FETCH_FAILED",
    });
  });

  test("cache makes second identical request faster and avoids re-download", async () => {
    const player = await makePng({ r: 130, g: 100, b: 20, stripe: true });
    const ai = await makePng({ r: 20, g: 100, b: 130, stripe: true });
    const fetcher = imageFetcher({ "https://x/player.png": player, "https://x/ai.png": ai }, 80);

    const req = {
      playerImageUrl: "https://x/player.png",
      aiImageUrl: "https://x/ai.png",
      promptText: "space cat",
    };

    const t1 = Date.now();
    await scoreImagesByUrl(req, { fetcher, embedder: async () => [1, 0, 0] });
    const d1 = Date.now() - t1;

    const t2 = Date.now();
    await scoreImagesByUrl(req, { fetcher, embedder: async () => [1, 0, 0] });
    const d2 = Date.now() - t2;

    expect((fetcher as unknown as ReturnType<typeof vi.fn>).mock.calls.length).toBe(2);
    expect(d2).toBeLessThan(d1);
  });
});
