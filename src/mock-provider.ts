import { createHash } from "node:crypto";
import type {
  GenerateImageRequest,
  GenerateImageResponse,
  GenerateTopicRequest,
  GenerateTopicResponse,
  WorkerProvider,
} from "./types.js";

const TOPIC_TEXTS = [
  "A tiny robot chef in a ramen shop",
  "A floating island library at sunset",
  "Cyberpunk cat detective in Tokyo rain",
  "Retro game arcade inside a cave",
  "A giant koi fish flying over downtown",
];

function seededPicsum(seed: string, width = 1024, height = 1024): string {
  return `https://picsum.photos/seed/${encodeURIComponent(seed)}/${width}/${height}`;
}

function deterministicEmbeddingFromString(input: string, dimensions = 128): number[] {
  const values = new Array<number>(dimensions).fill(0);
  for (let i = 0; i < dimensions; i += 1) {
    const hash = createHash("sha256").update(`${input}:${i}`).digest();
    const n = hash.readUInt32BE(0) / 0xffffffff;
    values[i] = n * 2 - 1;
  }
  return values;
}

export class MockProvider implements WorkerProvider {
  async generateTopic(input: GenerateTopicRequest): Promise<GenerateTopicResponse> {
    const idx = Math.abs(hashCode(input.roomCode)) % TOPIC_TEXTS.length;
    const topicText = TOPIC_TEXTS[idx];
    return {
      topicText,
      topicImageUrl: seededPicsum(`topic-${input.roomCode}-${topicText}`),
    };
  }

  async generateImage(input: GenerateImageRequest): Promise<GenerateImageResponse> {
    const seed = `${input.kind}:${input.requestId}:${input.prompt}:${input.isFinal}`;
    return { imageUrl: seededPicsum(seed) };
  }

  async embedImageFromUrl(imageUrl: string): Promise<number[]> {
    return deterministicEmbeddingFromString(imageUrl);
  }
}

function hashCode(value: string): number {
  let hash = 0;
  for (let i = 0; i < value.length; i += 1) {
    hash = (hash << 5) - hash + value.charCodeAt(i);
    hash |= 0;
  }
  return hash;
}
