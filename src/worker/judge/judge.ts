import { downloadImage } from "../../shared/image-fetch.js";
import { getVertexImageEmbedding } from "../vertex/embeddings.js";
import { cosineSimilarity } from "../score/cosine.js";
import { makeImageVariants } from "../score/preprocess.js";

export type JudgeRequest = {
  topicImageUrl: string;
  playerImageUrl: string;
  aiImageUrl: string;
};

export type JudgePairResult = {
  cosine: number;
  score100: number;
};

export type JudgeResult = {
  player: JudgePairResult;
  ai: JudgePairResult;
  winner: "player" | "ai" | "draw";
};

function cosineToScore100(cosine: number): number {
  const normalized = (Math.max(-1, Math.min(1, cosine)) + 1) / 2;
  return Math.max(0, Math.min(100, Math.round(normalized * 100)));
}

async function embedImageUrl(url: string, fetcher: typeof fetch): Promise<number[]> {
  const downloaded = await downloadImage(url, fetcher);
  const variants = await makeImageVariants(downloaded.buffer);
  const vector = await getVertexImageEmbedding({
    imagePng: variants.embeddingPng,
    fetcher,
  });
  if (!vector) {
    throw new Error("vertex_embedding_unavailable");
  }
  return vector;
}

export async function judgeByTopicSimilarity(input: JudgeRequest, fetcher: typeof fetch = fetch): Promise<JudgeResult> {
  const [topicVector, playerVector, aiVector] = await Promise.all([
    embedImageUrl(input.topicImageUrl, fetcher),
    embedImageUrl(input.playerImageUrl, fetcher),
    embedImageUrl(input.aiImageUrl, fetcher),
  ]);

  const playerCosine = cosineSimilarity(topicVector, playerVector);
  const aiCosine = cosineSimilarity(topicVector, aiVector);

  const player: JudgePairResult = {
    cosine: playerCosine,
    score100: cosineToScore100(playerCosine),
  };
  const ai: JudgePairResult = {
    cosine: aiCosine,
    score100: cosineToScore100(aiCosine),
  };

  const winner = player.score100 === ai.score100 ? "draw" : player.score100 > ai.score100 ? "player" : "ai";
  return { player, ai, winner };
}
