import { clamp01 } from "../shared/utils.js";

export function computeFitness(readability: number, twist: number, aesthetic: number): number {
  return clamp01(0.5 * readability + 0.35 * twist + 0.15 * aesthetic);
}
