# Implementation Tickets

1. Ticket: `INFRA-001` GCE + IAM + API 有効化
- 目的: ADC（metadata server）で Vertex/GCS を利用可能にする
- 完了条件:
  - GCE VM にサービスアカウントを付与
  - IAM: `roles/aiplatform.user`, `roles/storage.objectAdmin`, `roles/iam.serviceAccountTokenCreator` を付与
  - API: Vertex AI API, IAM Service Account Credentials API, Cloud Storage API を有効化

2. Ticket: `APP-001` config.ts 中心の設定管理
- 目的: JSONオブジェクト設定 + optional env override + zod 起動時検証
- 完了条件:
  - `src/config/app-config.ts` 実装
  - 欠落値で起動失敗

3. Ticket: `APP-002` Vertex クライアント実装
- 目的: Gemini(global) + Imagen(regional) + RPM レート制限 + 429/503 リトライ
- 完了条件:
  - `gemini-3-flash-preview` を `locations/global` で利用
  - `imagen-4.0-generate-001` を `IMAGEN_REGION` endpoint で利用
  - Imagen を `storageUri` 出力

4. Ticket: `APP-003` 進化ループ prewarm 実装
- 目的: 事前生成でのみ突然変異/自然淘汰を実行
- 完了条件:
  - 初期集団生成、世代ループ、readability 足切り、fitness 計算、親選抜、突然変異を実装
  - `count` 件の完成品を一括生成可能

5. Ticket: `APP-004` GCS プールと競合安全 move
- 目的: available -> used の一意消費
- 完了条件:
  - `pool/available/{itemId}/` に `final.png` + `meta.json`
  - `/generate` で claim + copy/delete + generation precondition による競合回避
  - 空プール時 503

6. Ticket: `APP-005` HTTP API + 運用導線
- 目的: `/healthz`, `/v1/pool/prewarm`, `/generate`, `/v1/topic/generate` の提供
- 完了条件:
  - タイムアウト・ログ・timingsMs 返却
  - README と systemd 定義を追加
