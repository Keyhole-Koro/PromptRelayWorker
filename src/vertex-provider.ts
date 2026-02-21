import { GoogleAuth } from "google-auth-library";
import type {
  GenerateImageRequest,
  GenerateImageResponse,
  GenerateTopicRequest,
  GenerateTopicResponse,
  WorkerProvider,
} from "./types.js";

const DEFAULT_LOCATION = process.env.GOOGLE_CLOUD_LOCATION ?? "us-central1";
const PROJECT_ID = process.env.GOOGLE_CLOUD_PROJECT;
const IMAGEN_MODEL = process.env.VERTEX_IMAGEN_MODEL ?? "imagen-4.0-generate-001";
const EMBEDDING_MODEL = process.env.VERTEX_EMBED_MODEL ?? "multimodalembedding@001";

const auth = new GoogleAuth({
  scopes: ["https://www.googleapis.com/auth/cloud-platform"],
});

type JsonObject = Record<string, unknown>;

function asObject(value: unknown): JsonObject {
  return typeof value === "object" && value !== null ? (value as JsonObject) : {};
}

function stringAt(obj: JsonObject, key: string): string | undefined {
  const v = obj[key];
  return typeof v === "string" ? v : undefined;
}

function numberArrayAt(obj: JsonObject, key: string): number[] | undefined {
  const v = obj[key];
  return Array.isArray(v) && v.every((item) => typeof item === "number")
    ? (v as number[])
    : undefined;
}

async function fetchAccessToken(): Promise<string> {
  const client = await auth.getClient();
  const tokenResponse = await client.getAccessToken();
  if (!tokenResponse.token) {
    throw new Error("Failed to get Google Cloud access token");
  }
  return tokenResponse.token;
}

async function vertexPredict(model: string, body: unknown): Promise<unknown> {
  if (!PROJECT_ID) {
    throw new Error("GOOGLE_CLOUD_PROJECT is required when USE_VERTEX=1");
  }

  const token = await fetchAccessToken();
  const url = `https://${DEFAULT_LOCATION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${DEFAULT_LOCATION}/publishers/google/models/${model}:predict`;

  const res = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Vertex predict failed (${res.status}): ${text}`);
  }

  return res.json();
}

async function imageUrlToBase64(imageUrl: string): Promise<string> {
  const res = await fetch(imageUrl);
  if (!res.ok) {
    throw new Error(`Failed to fetch image for embedding: ${res.status}`);
  }
  const arrayBuffer = await res.arrayBuffer();
  return Buffer.from(arrayBuffer).toString("base64");
}

export class VertexProvider implements WorkerProvider {
  async generateTopic(input: GenerateTopicRequest): Promise<GenerateTopicResponse> {
    const topicText = `Illustration challenge about room ${input.roomCode}`;
    const generated = await this.generateImage({
      requestId: `${input.roomCode}-topic`,
      kind: "ai",
      prompt: topicText,
      isFinal: true,
    });
    return {
      topicText,
      topicImageUrl: generated.imageUrl,
    };
  }

  async generateImage(input: GenerateImageRequest): Promise<GenerateImageResponse> {
    const body = {
      instances: [
        {
          prompt: input.prompt,
        },
      ],
      parameters: {
        sampleCount: 1,
      },
    };

    const json = asObject(await vertexPredict(IMAGEN_MODEL, body));
    const predictions = Array.isArray(json.predictions) ? json.predictions : [];
    const first = asObject(predictions[0]);
    const nestedImage = asObject(first.image);

    const imageBase64 =
      stringAt(first, "bytesBase64Encoded") ?? stringAt(nestedImage, "bytesBase64Encoded");

    const uri = stringAt(first, "gcsUri") ?? stringAt(nestedImage, "gcsUri");

    if (imageBase64) {
      return { imageUrl: `data:image/png;base64,${imageBase64}` };
    }

    if (uri) {
      return { imageUrl: uri };
    }

    // TODO: Vertex Imagen response shape may vary by model/version.
    // If this throws in your environment, log `json` and map the exact field path here.
    throw new Error("Imagen response did not include a supported image field");
  }

  async embedImageFromUrl(imageUrl: string): Promise<number[]> {
    const imageBase64 = await imageUrlToBase64(imageUrl);

    const body = {
      instances: [
        {
          image: {
            bytesBase64Encoded: imageBase64,
          },
        },
      ],
    };

    const json = asObject(await vertexPredict(EMBEDDING_MODEL, body));
    const predictions = Array.isArray(json.predictions) ? json.predictions : [];
    const first = asObject(predictions[0]);

    const vector =
      numberArrayAt(first, "imageEmbedding") ??
      numberArrayAt(first, "embeddings") ??
      numberArrayAt(first, "embedding");

    if (!vector) {
      // TODO: Vertex embedding response shape may vary by model/version.
      // If this throws in your environment, log `json` and map the exact field path here.
      throw new Error("Embedding response did not include a numeric vector field");
    }

    return vector;
  }
}
