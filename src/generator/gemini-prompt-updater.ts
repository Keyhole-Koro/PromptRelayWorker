import { config } from "../config/app-config.js";
import { log } from "../observability/logger.js";
import { downloadImage } from "../shared/image-fetch.js";
import Bottleneck from "bottleneck";
import { withRetry } from "../shared/utils.js";
import { GoogleAuth } from "google-auth-library";

const auth = new GoogleAuth({
    scopes: ["https://www.googleapis.com/auth/cloud-platform"],
});

const geminiLimiter = new Bottleneck({
    maxConcurrent: config.MAX_CONCURRENCY_GEMINI,
    reservoir: config.RATE_LIMIT_GEMINI_RPM,
    reservoirRefreshAmount: config.RATE_LIMIT_GEMINI_RPM,
    reservoirRefreshInterval: 60_000,
});

async function accessToken(): Promise<string> {
    const client = await auth.getClient();
    const token = await client.getAccessToken();
    if (!token.token) {
        throw new Error("failed to get ADC access token");
    }
    return token.token;
}

function isRetriableStatus(status: number): boolean {
    return status === 429 || status === 503;
}

function parseStatus(error: unknown): number | undefined {
    if (!(error instanceof Error)) {
        return undefined;
    }
    const match = error.message.match(/status=(\d{3})/);
    if (!match) {
        return undefined;
    }
    return Number.parseInt(match[1], 10);
}

export async function updatePromptWithGemini(params: {
    themeImageUrl: string;
    recentImageUrl: string;
    recentPrompt: string;
    runId: string;
}): Promise<string> {
    const [themeImage, recentImage] = await Promise.all([
        downloadImage(params.themeImageUrl),
        downloadImage(params.recentImageUrl),
    ]);

    const url = `https://aiplatform.googleapis.com/v1/projects/${config.GOOGLE_CLOUD_PROJECT}/locations/global/publishers/google/models/${config.GEMINI_MODEL}:generateContent`;

    const instruction = `
You are playing a cooperative drawing game. Your goal is to guide the image generation to match the "theme image" as closely as possible.
You are given:
1. The Target Theme Image (Image 1)
2. The Recently Generated Image (Image 2)
3. The prompt used to generate the recent image.

Compare the recent image to the theme image. What is missing? What is different?
Provide an updated prompt that will result in an image closer to the theme image.
Keep the prompt concise, descriptive, and focused on the visual elements.

Output ONLY your new prompt. Do not include any explanations or extra text.
`.trim();

    const body = {
        contents: [
            {
                role: "user",
                parts: [
                    {
                        text: instruction,
                    },
                    {
                        inlineData: {
                            mimeType: themeImage.contentType,
                            data: themeImage.buffer.toString("base64"),
                        },
                    },
                    {
                        inlineData: {
                            mimeType: recentImage.contentType,
                            data: recentImage.buffer.toString("base64"),
                        },
                    },
                    {
                        text: `Recent Prompt: ${params.recentPrompt}`,
                    },
                ],
            },
        ],
        generationConfig: {
            temperature: 0.4,
            maxOutputTokens: 256,
            topP: 0.95,
            responseMimeType: "text/plain",
        },
    };

    const token = await accessToken();

    return withRetry(
        () =>
            geminiLimiter.schedule(async () => {
                log("info", "gemini_prompt_update_request", {
                    runId: params.runId,
                    model: config.GEMINI_MODEL,
                });

                const res = await fetch(url, {
                    method: "POST",
                    headers: {
                        Authorization: `Bearer ${token}`,
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify(body),
                });

                const text = await res.text();
                if (!res.ok) {
                    throw new Error(`vertex request failed status=${res.status} body=${text}`);
                }

                try {
                    const json = JSON.parse(text);
                    const candidates = json.candidates || [];
                    if (candidates.length === 0) {
                        throw new Error("gemini response has no candidates");
                    }
                    const parts = candidates[0]?.content?.parts || [];
                    const textPart = parts.find((p: any) => typeof p.text === "string");
                    if (!textPart) {
                        throw new Error("gemini response missing text part");
                    }
                    return textPart.text.trim();
                } catch (err) {
                    throw new Error(`failed to parse gemini response: ${err instanceof Error ? err.message : String(err)}`);
                }
            }),
        (error) => {
            const status = parseStatus(error);
            return status !== undefined && isRetriableStatus(status);
        },
        config.MAX_VERTEX_RETRIES,
    );
}
