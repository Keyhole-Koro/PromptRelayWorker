import { z } from "zod";
import type { AspectRatio } from "../domain/types.js";

const ConfigSchema = z.object({
  PORT: z.number().int().positive(),
  GOOGLE_CLOUD_PROJECT: z.string().min(1),
  IMAGEN_REGION: z.string().min(1),
  GCS_BUCKET: z.string().min(1),
  LOCAL_DATA_DIR: z.string().min(1),
  SIGNED_URL_TTL_SEC: z.number().int().positive(),
  GCS_PREFIX_AVAILABLE: z.string().min(1),
  GCS_PREFIX_USED: z.string().min(1),
  RATE_LIMIT_IMAGEN_RPM: z.number().int().positive(),
  RATE_LIMIT_GEMINI_RPM: z.number().int().positive(),
  MAX_CONCURRENCY_IMAGEN: z.number().int().positive(),
  MAX_CONCURRENCY_GEMINI: z.number().int().positive(),
  EVOLVE_GENERATIONS: z.number().int().positive(),
  EVOLVE_POPULATION: z.number().int().positive(),
  EVOLVE_PARENTS: z.number().int().positive(),
  EVOLVE_SETTINGS_PATH: z.string().min(1),
  REQUEST_TIMEOUT_MS: z.number().int().positive(),
  HOST: z.string().min(1),
  DEFAULT_ASPECT_RATIO: z.enum(["1:1", "16:9"]),
  MAX_MOVE_RETRIES: z.number().int().positive(),
  MAX_VERTEX_RETRIES: z.number().int().positive(),
  GEMINI_MODEL: z.string().min(1),
  MULTIMODAL_EMBEDDING_MODEL: z.string().min(1),
  IMAGEN_MODEL: z.string().min(1),
  TEXT_EMBEDDING_MODEL: z.string().min(1),
});

export type Config = z.infer<typeof ConfigSchema>;

const baseConfig = {
  PORT: 8091,
  GOOGLE_CLOUD_PROJECT: "promptrelay",
  IMAGEN_REGION: "asia-northeast1",
  GCS_BUCKET: "not-used-when-disk-mode",
  LOCAL_DATA_DIR: "/home/korokororin47_gmail_com/PromptRelay/PromptRelayWorker/data",
  SIGNED_URL_TTL_SEC: 900,
  GCS_PREFIX_AVAILABLE: "pool/available/",
  GCS_PREFIX_USED: "pool/used/",
  RATE_LIMIT_IMAGEN_RPM: 60,
  RATE_LIMIT_GEMINI_RPM: 120,
  MAX_CONCURRENCY_IMAGEN: 4,
  MAX_CONCURRENCY_GEMINI: 8,
  EVOLVE_GENERATIONS: 3,
  EVOLVE_POPULATION: 8,
  EVOLVE_PARENTS: 3,
  EVOLVE_SETTINGS_PATH: "src/generator/evolve-settings.json",
  REQUEST_TIMEOUT_MS: 120000,
  HOST: "127.0.0.1",
  DEFAULT_ASPECT_RATIO: "1:1" as AspectRatio,
  MAX_MOVE_RETRIES: 8,
  MAX_VERTEX_RETRIES: 3,
  GEMINI_MODEL: "gemini-3-flash-preview",
  MULTIMODAL_EMBEDDING_MODEL: "multimodalembedding@001",
  IMAGEN_MODEL: "imagen-4.0-generate-001",
  TEXT_EMBEDDING_MODEL: "text-embedding-005",
} satisfies Config;

function asInt(name: string): number | undefined {
  const value = process.env[name];
  if (value === undefined) {
    return undefined;
  }
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function envOverride(): Partial<Config> {
  return {
    PORT: asInt("PORT"),
    GOOGLE_CLOUD_PROJECT: process.env.GOOGLE_CLOUD_PROJECT,
    IMAGEN_REGION: process.env.IMAGEN_REGION,
    GCS_BUCKET: process.env.GCS_BUCKET,
    LOCAL_DATA_DIR: process.env.LOCAL_DATA_DIR,
    SIGNED_URL_TTL_SEC: asInt("SIGNED_URL_TTL_SEC"),
    GCS_PREFIX_AVAILABLE: process.env.GCS_PREFIX_AVAILABLE,
    GCS_PREFIX_USED: process.env.GCS_PREFIX_USED,
    RATE_LIMIT_IMAGEN_RPM: asInt("RATE_LIMIT_IMAGEN_RPM"),
    RATE_LIMIT_GEMINI_RPM: asInt("RATE_LIMIT_GEMINI_RPM"),
    MAX_CONCURRENCY_IMAGEN: asInt("MAX_CONCURRENCY_IMAGEN"),
    MAX_CONCURRENCY_GEMINI: asInt("MAX_CONCURRENCY_GEMINI"),
    EVOLVE_GENERATIONS: asInt("EVOLVE_GENERATIONS"),
    EVOLVE_POPULATION: asInt("EVOLVE_POPULATION"),
    EVOLVE_PARENTS: asInt("EVOLVE_PARENTS"),
    EVOLVE_SETTINGS_PATH: process.env.EVOLVE_SETTINGS_PATH,
    REQUEST_TIMEOUT_MS: asInt("REQUEST_TIMEOUT_MS"),
    HOST: process.env.HOST,
    DEFAULT_ASPECT_RATIO: process.env.DEFAULT_ASPECT_RATIO as AspectRatio | undefined,
    MAX_MOVE_RETRIES: asInt("MAX_MOVE_RETRIES"),
    MAX_VERTEX_RETRIES: asInt("MAX_VERTEX_RETRIES"),
    GEMINI_MODEL: process.env.GEMINI_MODEL,
    MULTIMODAL_EMBEDDING_MODEL: process.env.MULTIMODAL_EMBEDDING_MODEL,
    IMAGEN_MODEL: process.env.IMAGEN_MODEL,
    TEXT_EMBEDDING_MODEL: process.env.TEXT_EMBEDDING_MODEL,
  };
}

const merged = {
  ...baseConfig,
  ...Object.fromEntries(Object.entries(envOverride()).filter(([, v]) => v !== undefined)),
};

export const config: Config = ConfigSchema.parse(merged);
