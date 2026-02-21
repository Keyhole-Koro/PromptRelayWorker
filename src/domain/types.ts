export type AspectRatio = "1:1" | "16:9";

export type TwistType = "material" | "scale" | "role" | "physics" | "causality";

export type Genome = {
  baseScene: string;
  subject: string;
  action: string;
  twist: Array<{
    type: TwistType;
    description: string;
    strength: 1 | 2;
  }>;
  style: {
    medium: "photo" | "illustration";
    mood: string;
    lighting: string;
    lens?: string;
  };
  composition: {
    focus: "single_subject";
    framing: "center" | "rule_of_thirds";
    backgroundClarity: "clear";
  };
  constraints: string[];
};

export type ScoreBreakdown = {
  readability: number;
  twist: number;
  aesthetic: number;
  fitness: number;
  novelty?: number;
  labels: {
    scene?: string;
    subject?: string;
    action?: string;
  };
  flags: string[];
  short_reason: string;
};

export type PoolItemMeta = {
  itemId: string;
  runId: string;
  createdAt: string;
  aspectRatio: AspectRatio;
  prompt: string;
  genome: Genome;
  scores: ScoreBreakdown;
  generation: number;
  promptHistory?: Array<{
    generation: number;
    candidateIndex: number;
    prompt: string;
    scores: ScoreBreakdown;
  }>;
};

export type EvolveBudget = {
  generations: number;
  population: number;
  parents: number;
};

export type PrewarmRequest = {
  count?: number;
  budget?: Partial<EvolveBudget>;
  aspectRatio?: AspectRatio;
};

export type PrewarmResponse = {
  created: number;
  items: Array<{ itemId: string; gcsPrefix: string }>;
  timingsMs: Record<string, number>;
};

export type GenerateRequest = {
  aspectRatio?: AspectRatio;
};

export type GenerateResponse = {
  topic: {
    imageUri: string;
    signedUrl: string;
    imageUrl: string;
    prompt: string;
    genome: Genome;
    scores: ScoreBreakdown;
  };
  itemId: string;
  timingsMs: {
    total: number;
    gcs: number;
    sign: number;
  };
};
