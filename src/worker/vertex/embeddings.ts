const METADATA_TOKEN_URL =
  "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token";

export type VertexEmbeddingInput = {
  imagePng: Buffer;
  promptText?: string;
  fetcher?: typeof fetch;
};

type TokenResponse = {
  access_token?: string;
};

function envEnabled(): boolean {
  return process.env.VERTEX_ENABLED !== "0";
}

function projectId(): string | undefined {
  return process.env.VERTEX_PROJECT_ID ?? process.env.GOOGLE_CLOUD_PROJECT;
}

function location(): string {
  return process.env.VERTEX_LOCATION ?? "asia-northeast1";
}

function compactPrompt(promptText: string | undefined): string {
  const text = (promptText ?? "").trim();
  if (text.length === 0) {
    return "image similarity";
  }
  const words = text.split(/\s+/).slice(0, 32);
  return words.join(" ");
}

async function fetchWithTimeout(url: string, options: RequestInit, timeoutMs: number, fetcher: typeof fetch): Promise<Response> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    return await fetcher(url, { ...options, signal: ctrl.signal });
  } finally {
    clearTimeout(timer);
  }
}

async function accessToken(fetcher: typeof fetch): Promise<string | undefined> {
  if (process.env.VERTEX_ACCESS_TOKEN) {
    return process.env.VERTEX_ACCESS_TOKEN;
  }

  const res = await fetchWithTimeout(
    METADATA_TOKEN_URL,
    {
      method: "GET",
      headers: {
        "Metadata-Flavor": "Google",
      },
    },
    1500,
    fetcher,
  );

  if (!res.ok) {
    return undefined;
  }

  const json = (await res.json()) as TokenResponse;
  return typeof json.access_token === "string" ? json.access_token : undefined;
}

export async function getVertexImageEmbedding(input: VertexEmbeddingInput): Promise<number[] | undefined> {
  if (!envEnabled()) {
    return undefined;
  }

  const project = projectId();
  if (!project) {
    return undefined;
  }

  const fetcher = input.fetcher ?? fetch;
  const token = await accessToken(fetcher);
  if (!token) {
    return undefined;
  }

  const loc = location();
  const url = `https://${loc}-aiplatform.googleapis.com/v1/projects/${project}/locations/${loc}/publishers/google/models/multimodalembedding@001:predict`;

  const body = {
    instances: [
      {
        text: compactPrompt(input.promptText),
        image: {
          bytesBase64Encoded: input.imagePng.toString("base64"),
        },
      },
    ],
  };

  const res = await fetchWithTimeout(
    url,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    },
    10_000,
    fetcher,
  );

  if (!res.ok) {
    return undefined;
  }

  const json = (await res.json()) as { predictions?: Array<{ imageEmbedding?: number[] }> };
  const emb = json.predictions?.[0]?.imageEmbedding;
  if (!Array.isArray(emb) || emb.length === 0 || !emb.every((v) => typeof v === "number")) {
    return undefined;
  }

  return emb;
}
