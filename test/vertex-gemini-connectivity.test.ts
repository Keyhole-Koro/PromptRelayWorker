import assert from "node:assert/strict";
import test from "node:test";
import { GoogleAuth } from "google-auth-library";

type ConfigLike = {
  RUN_VERTEX_TEST?: string;
  RUN_VERTEX_CONNECTIVITY_TEST?: string;
  GOOGLE_CLOUD_PROJECT?: string;
  GEMINI_FLASH_MODEL?: string;
};

const cfg: ConfigLike = {};
const PROJECT_ID = process.env.GOOGLE_CLOUD_PROJECT ?? cfg.GOOGLE_CLOUD_PROJECT;
const MODEL = process.env.GEMINI_FLASH_MODEL ?? cfg.GEMINI_FLASH_MODEL ?? "gemini-3-flash-preview";

const shouldRun =
  (process.env.RUN_VERTEX_TEST ?? cfg.RUN_VERTEX_TEST) === "1" ||
  (process.env.RUN_VERTEX_CONNECTIVITY_TEST ?? cfg.RUN_VERTEX_CONNECTIVITY_TEST) === "1";

test("vertex ai gemini global connectivity", { skip: !shouldRun }, async (t) => {
  if (!PROJECT_ID) {
    t.skip("GOOGLE_CLOUD_PROJECT is not set");
    return;
  }

  const auth = new GoogleAuth({
    scopes: ["https://www.googleapis.com/auth/cloud-platform"],
  });
  const client = await auth.getClient();
  const tokenResponse = await client.getAccessToken();
  assert.ok(tokenResponse.token, "failed to acquire Google access token");

  const url = `https://aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/global/publishers/google/models/${MODEL}:generateContent`;

  const res = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${tokenResponse.token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      contents: [
        {
          role: "user",
          parts: [{ text: "Reply with exactly OK" }],
        },
      ],
      generationConfig: {
        maxOutputTokens: 8,
        temperature: 0,
      },
    }),
  });

  const raw = await res.text();
  assert.equal(res.ok, true, `Vertex request failed: status=${res.status} body=${raw}`);

  const json = JSON.parse(raw) as {
    candidates?: Array<{
      content?: { parts?: Array<{ text?: string }> };
    }>;
  };
  const text = json.candidates?.[0]?.content?.parts?.[0]?.text ?? "";
  assert.notEqual(text.trim(), "", "Gemini returned empty text");
});
