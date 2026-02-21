import { LRUCache } from "lru-cache";

const IMAGE_TIMEOUT_MS = 10_000;
const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
const DEFAULT_CACHE_TTL_MS = 60_000;

export class ImageFetchHttpError extends Error {
    readonly status: number;
    readonly code: string;

    constructor(status: number, code: string, message: string) {
        super(message);
        this.status = status;
        this.code = code;
    }
}

export type DownloadedImage = {
    buffer: Buffer;
    contentType: string;
};

function cacheTtlMs(): number {
    const fromEnv = Number.parseInt(process.env.IMAGE_FETCH_CACHE_TTL_MS ?? "", 10);
    if (Number.isFinite(fromEnv) && fromEnv > 0) {
        return fromEnv;
    }
    return DEFAULT_CACHE_TTL_MS;
}

const imageCache = new LRUCache<string, DownloadedImage>({
    max: 128,
    ttl: cacheTtlMs(),
});

function isHttpUrl(input: string): boolean {
    try {
        const url = new URL(input);
        return url.protocol === "http:" || url.protocol === "https:";
    } catch {
        return false;
    }
}

async function fetchWithTimeout(fetcher: typeof fetch, url: string, timeoutMs: number): Promise<Response> {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
        return await fetcher(url, { method: "GET", signal: ctrl.signal });
    } catch (error) {
        if (error instanceof Error && error.name === "AbortError") {
            throw new ImageFetchHttpError(408, "IMAGE_TIMEOUT", `image download timeout: ${url}`);
        }
        throw error;
    } finally {
        clearTimeout(timer);
    }
}

async function readBodyWithLimit(res: Response, maxBytes: number): Promise<Buffer> {
    const reader = res.body?.getReader();
    if (!reader) {
        throw new ImageFetchHttpError(502, "IMAGE_STREAM_ERROR", "image response has no readable stream");
    }

    const chunks: Uint8Array[] = [];
    let total = 0;

    while (true) {
        const { done, value } = await reader.read();
        if (done) {
            break;
        }
        if (!value) {
            continue;
        }
        total += value.byteLength;
        if (total > maxBytes) {
            throw new ImageFetchHttpError(413, "IMAGE_TOO_LARGE", `image exceeds ${maxBytes} bytes`);
        }
        chunks.push(value);
    }

    return Buffer.concat(chunks.map((c) => Buffer.from(c)));
}

export async function downloadImage(url: string, fetcher: typeof fetch = fetch): Promise<DownloadedImage> {
    const cached = imageCache.get(url);
    if (cached) {
        return cached;
    }

    if (!isHttpUrl(url)) {
        throw new ImageFetchHttpError(400, "INVALID_IMAGE_URL", `invalid image url: ${url}`);
    }

    const res = await fetchWithTimeout(fetcher, url, IMAGE_TIMEOUT_MS);
    if (!res.ok) {
        throw new ImageFetchHttpError(400, "IMAGE_FETCH_FAILED", `failed to fetch image (${res.status})`);
    }

    const contentType = res.headers.get("content-type") ?? "";
    if (!contentType.toLowerCase().startsWith("image/")) {
        throw new ImageFetchHttpError(400, "INVALID_IMAGE_CONTENT_TYPE", `content-type is not image: ${contentType}`);
    }

    const contentLength = Number.parseInt(res.headers.get("content-length") ?? "", 10);
    if (Number.isFinite(contentLength) && contentLength > MAX_IMAGE_BYTES) {
        throw new ImageFetchHttpError(413, "IMAGE_TOO_LARGE", `image exceeds ${MAX_IMAGE_BYTES} bytes`);
    }

    const buffer = await readBodyWithLimit(res, MAX_IMAGE_BYTES);
    const result: DownloadedImage = { buffer, contentType };
    imageCache.set(url, result);
    return result;
}
