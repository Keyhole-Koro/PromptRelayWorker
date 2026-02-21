export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length === 0 || b.length === 0 || a.length !== b.length) {
    throw new Error("Embedding vectors must be non-empty and same length");
  }

  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i += 1) {
    const av = a[i];
    const bv = b[i];
    dot += av * bv;
    normA += av * av;
    normB += bv * bv;
  }

  if (normA === 0 || normB === 0) {
    return 0;
  }

  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

export function cosineToScore100(cosine: number): number {
  const normalized = (Math.max(-1, Math.min(1, cosine)) + 1) / 2;
  const score = Math.round(normalized * 100);
  return Math.max(0, Math.min(100, score));
}
