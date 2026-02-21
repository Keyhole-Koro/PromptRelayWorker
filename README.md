# PromptRelay Worker (GCE + Vertex AI + GCS pool)

Node.js + TypeScript worker for pre-generating topic images with Vertex AI and serving them from a GCS pool.

## Flow
1. Run `POST /v1/pool/prewarm` to create items in `pool/available/`.
2. Run `POST /generate` to atomically move one item to `pool/used/` and get signed URL.
3. `/generate` does not call evolution or image generation in real time.

## API
- `GET /healthz` -> `200 ok`
- `GET /worker` -> visual debug page (image score/generate/pool check)
- `POST /v1/pool/prewarm`
- `POST /generate`
- `POST /v1/topic/generate` (same behavior as `/generate`)
- `POST /v1/debug/score`
- `POST /v1/debug/generate`
- `POST /v1/debug/prompt-score`
- OpenAPI schema: `docs/api-schema.yaml`

## GCE Setup
1. Create VM (Node.js 20+ installed).
2. Attach a service account to VM.
3. Grant IAM roles to that service account:
- `roles/aiplatform.user`
- `roles/storage.objectAdmin`
- `roles/iam.serviceAccountTokenCreator` (for signed URL)
4. Enable APIs:
- Vertex AI API
- Cloud Storage API
- IAM Service Account Credentials API

## Config
Primary config is `src/config/app-config.ts` as a JSON-style object (`baseConfig`).

Required values to edit in `src/config/app-config.ts`:
- `GOOGLE_CLOUD_PROJECT`
- `IMAGEN_REGION`
- `GCS_BUCKET`

Optional env overrides are supported but not required.

## Install / Run
```bash
npm install
npm run typecheck
npm run start
```

Worker listens on `127.0.0.1:PORT`.

## Dev Mode Note
If `promptrelay-worker` is running via systemd, port `8091` is already in use and `npm run dev` fails with `EADDRINUSE`.

Stop/disable service before local dev:
```bash
sudo systemctl stop promptrelay-worker
sudo systemctl disable promptrelay-worker
npm run dev
```

Re-enable after dev:
```bash
sudo systemctl enable --now promptrelay-worker
```

## Quick Curl
Health:
```bash
curl -sS http://127.0.0.1:8091/healthz
```

Prewarm first:
```bash
curl -sS -X POST http://127.0.0.1:8091/v1/pool/prewarm \
  -H 'content-type: application/json' \
  -d '{
    "count": 3,
    "budget": {"generations": 3, "population": 8, "parents": 3},
    "aspectRatio": "1:1"
  }'
```

Then generate:
```bash
curl -sS -X POST http://127.0.0.1:8091/generate \
  -H 'content-type: application/json' \
  -d '{"aspectRatio":"1:1"}'
```

If pool is empty:
- status `503`
- message: `pool empty。/v1/pool/prewarm を実行してください`

## GCS Layout
- `gs://{BUCKET}/{GCS_PREFIX_AVAILABLE}{itemId}/final.png`
- `gs://{BUCKET}/{GCS_PREFIX_AVAILABLE}{itemId}/meta.json`
- `gs://{BUCKET}/{GCS_PREFIX_USED}{itemId}/final.png`
- `gs://{BUCKET}/{GCS_PREFIX_USED}{itemId}/meta.json`

## Logging
JSON logs include:
- `runId`, `itemId`
- generation/candidate index
- fitness/readability
- filtered reason
- API errors

## systemd
See `deploy/promptrelay-worker.service`.

## Caddy
Route example is in `deploy/Caddyfile` and exposes `/worker`, `/v1/*`, `/generate`, `/healthz`.
