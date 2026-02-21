import { createHash } from "node:crypto";

function xorshift32(seed: number): () => number {
  let state = seed >>> 0;
  if (state === 0) {
    state = 0x9e3779b9;
  }

  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 0x100000000;
  };
}

export function deterministicEmbeddingFromImageBytes(bytes: Buffer, dim = 256): number[] {
  const digest = createHash("sha256").update(bytes).digest();
  const seed = digest.readUInt32BE(0);
  const rand = xorshift32(seed);

  const vector: number[] = [];
  for (let i = 0; i < dim; i += 1) {
    const value = rand() * 2 - 1;
    vector.push(value);
  }
  return vector;
}
