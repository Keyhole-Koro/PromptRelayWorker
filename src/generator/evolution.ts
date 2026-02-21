import { join } from "node:path";
import { config } from "../config/app-config.js";
import { log } from "../observability/logger.js";
import { computeFitness } from "../scorer/fitness.js";
import { evaluateWithGeminiInline } from "../scorer/vertex-gemini-scorer.js";
import type { AspectRatio, EvolveBudget, Genome, PoolItemMeta, ScoreBreakdown } from "../domain/types.js";
import { mapLimit } from "../shared/utils.js";
import { generateImagenBase64, savePngBase64ToDisk } from "./vertex-imagen-generator.js";

type Candidate = {
  genome: Genome;
  prompt: string;
  imageBase64: string;
  tempImagePath: string;
  scores: ScoreBreakdown;
  generation: number;
};

const BASE_SCENES = [
  "city street",
  "classroom",
  "kitchen",
  "park",
  "museum hall",
  "beach boardwalk",
];
const SUBJECTS = ["cat", "dog", "robot", "chef", "child", "astronaut", "teacher"];
const ACTIONS = ["running", "cooking", "reading", "pointing", "dancing", "sleeping"];
const MOODS = ["playful", "serious", "calm", "mysterious", "bright"];
const LIGHTINGS = ["soft daylight", "golden hour", "studio light", "backlit"];
const LENSES = ["35mm", "50mm", "85mm"];
const TWISTS = [
  { type: "material", description: "body made of transparent glass" },
  { type: "scale", description: "subject is toy-sized" },
  { type: "role", description: "subject acts as a school principal" },
  { type: "physics", description: "gravity is sideways" },
  { type: "causality", description: "shadow moves independently" },
] as const;

const DEFAULT_CONSTRAINTS = ["no text", "no logos", "no collage", "no extra subjects"];

function pick<T>(arr: readonly T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

function maybe<T>(value: T): T | undefined {
  return Math.random() < 0.5 ? value : undefined;
}

function randomTwist() {
  const count = Math.random() < 0.8 ? 1 : Math.random() < 0.5 ? 0 : 2;
  const pool = [...TWISTS];
  const twists: Genome["twist"] = [];
  for (let i = 0; i < count && pool.length > 0; i += 1) {
    const idx = Math.floor(Math.random() * pool.length);
    const t = pool.splice(idx, 1)[0];
    twists.push({ type: t.type, description: t.description, strength: Math.random() < 0.7 ? 1 : 2 });
  }
  return twists;
}

function randomGenome(baseScene: string): Genome {
  return {
    baseScene,
    subject: pick(SUBJECTS),
    action: pick(ACTIONS),
    twist: randomTwist(),
    style: {
      medium: Math.random() < 0.5 ? "photo" : "illustration",
      mood: pick(MOODS),
      lighting: pick(LIGHTINGS),
      lens: maybe(pick(LENSES)),
    },
    composition: {
      focus: "single_subject",
      framing: Math.random() < 0.5 ? "center" : "rule_of_thirds",
      backgroundClarity: "clear",
    },
    constraints: [...DEFAULT_CONSTRAINTS],
  };
}

function mutate(parent: Genome): Genome {
  const child: Genome = JSON.parse(JSON.stringify(parent));
  const p = Math.random();
  if (p < 0.2) child.subject = pick(SUBJECTS);
  else if (p < 0.4) child.action = pick(ACTIONS);
  else if (p < 0.55) child.style.mood = pick(MOODS);
  else if (p < 0.7) child.style.lighting = pick(LIGHTINGS);
  else if (p < 0.85) child.composition.framing = child.composition.framing === "center" ? "rule_of_thirds" : "center";
  else child.twist = randomTwist();
  child.constraints = [...DEFAULT_CONSTRAINTS];
  return child;
}

function buildPrompt(genome: Genome): string {
  const twistText = genome.twist.length
    ? genome.twist.map((t) => `${t.type}:${t.description}(strength=${t.strength})`).join(", ")
    : "none";
  const lens = genome.style.lens ? `, lens ${genome.style.lens}` : "";
  return [
    `Single ${genome.subject} in ${genome.baseScene}`,
    `Action: ${genome.action}`,
    `Twist: ${twistText}`,
    `Style: ${genome.style.medium}, mood ${genome.style.mood}, lighting ${genome.style.lighting}${lens}`,
    `Composition: ${genome.composition.framing}, clear background, single_subject focus`,
    `Constraints: ${genome.constraints.join(", ")}`,
  ].join(". ");
}

export async function evolveOneItem(params: {
  runId: string;
  itemId: string;
  budget: EvolveBudget;
  aspectRatio: AspectRatio;
}): Promise<PoolItemMeta & { tempImagePath: string }> {
  const baseScene = pick(BASE_SCENES);
  let population: Genome[] = Array.from({ length: params.budget.population }, () => randomGenome(baseScene));
  let bestOverall: Candidate | undefined;

  for (let generation = 0; generation < params.budget.generations; generation += 1) {
    const candidates = await mapLimit(population, config.MAX_CONCURRENCY_IMAGEN, async (genome, i) => {
      const candidateId = `${params.itemId}-g${generation}-i${i}`;
      const prompt = buildPrompt(genome);
      const imageBase64 = await generateImagenBase64({
        prompt,
        aspectRatio: params.aspectRatio,
        runId: params.runId,
        generation,
        candidateIndex: i,
      });
      const tempImagePath = join(config.LOCAL_DATA_DIR, "runs", params.runId, candidateId, "candidate.png");
      await savePngBase64ToDisk(imageBase64, tempImagePath);

      const scores = await evaluateWithGeminiInline({
        imageBase64,
        mimeType: "image/png",
        prompt,
        genome,
        runId: params.runId,
        generation,
        candidateIndex: i,
      });
      if (scores.readability < 0.65) {
        scores.flags = [...scores.flags, "filtered_readability_lt_0_65"];
      }
      log("info", "candidate_scored", {
        runId: params.runId,
        itemId: params.itemId,
        generation,
        candidateIndex: i,
        fitness: scores.fitness,
        readability: scores.readability,
        filtered: scores.readability < 0.65,
      });
      return { genome, prompt, imageBase64, tempImagePath, scores, generation } satisfies Candidate;
    });

    const viable = candidates.filter((c) => c.scores.readability >= 0.65);
    const ranked = [...(viable.length > 0 ? viable : candidates)].sort((a, b) => b.scores.fitness - a.scores.fitness);
    const generationBest = ranked[0];

    if (!bestOverall || generationBest.scores.fitness > bestOverall.scores.fitness) {
      bestOverall = generationBest;
    }

    const parentCount = Math.max(1, Math.min(params.budget.parents, ranked.length));
    const parents = ranked.slice(0, parentCount).map((c) => c.genome);
    population = Array.from({ length: params.budget.population }, (_, idx) => {
      if (idx < parents.length) {
        return parents[idx];
      }
      return mutate(pick(parents));
    });
  }

  if (!bestOverall) {
    throw new Error("evolution did not produce any candidate");
  }

  return {
    itemId: params.itemId,
    runId: params.runId,
    createdAt: new Date().toISOString(),
    aspectRatio: params.aspectRatio,
    prompt: bestOverall.prompt,
    genome: bestOverall.genome,
    scores: {
      ...bestOverall.scores,
      fitness: computeFitness(
        bestOverall.scores.readability,
        bestOverall.scores.twist,
        bestOverall.scores.aesthetic,
      ),
    },
    generation: bestOverall.generation,
    tempImagePath: bestOverall.tempImagePath,
  };
}
