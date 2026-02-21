# Delivery Plan

1. 設定層
- `src/config/app-config.ts` で JSON形式設定 + zod 検証

2. Vertex 層
- Imagen regional endpoint (`:predict`) + `storageUri`
- Gemini global endpoint (`:generateContent`) + structured JSON output
- Bottleneck で RPM/concurrency 制御
- 429/503 は指数バックオフ最大3回

3. 進化層 (prewarm only)
- Genome 初期集団
- Imagen 生成
- Gemini 採点
- readability<0.65 足切り
- fitness 計算と親選抜
- 突然変異で次世代

4. プール層
- available への保存
- /generate で claim + generation precondition による競合安全 move
- used へ移動後に signed URL 発行

5. API 層
- `/healthz`, `/v1/pool/prewarm`, `/generate`, `/v1/topic/generate`
- タイムアウトとエラーハンドリング

6. 運用層
- README, systemd, ticket一覧
