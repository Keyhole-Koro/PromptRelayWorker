export type GenerateTopicRequest = {
  roomCode: string;
};

export type GenerateTopicResponse = {
  topicImageUrl: string;
  topicText?: string;
};

export type GenerateImageRequest = {
  requestId: string;
  kind: "player" | "ai";
  prompt: string;
  isFinal: boolean;
};

export type GenerateImageResponse = {
  imageUrl: string;
};

export type ScoreRequest = {
  playerImageUrl: string;
  aiImageUrl: string;
};

export type ScoreResponse = {
  cosine: number;
  score100: number;
};

export interface WorkerProvider {
  generateTopic(input: GenerateTopicRequest): Promise<GenerateTopicResponse>;
  generateImage(input: GenerateImageRequest): Promise<GenerateImageResponse>;
  embedImageFromUrl(imageUrl: string): Promise<number[]>;
}
