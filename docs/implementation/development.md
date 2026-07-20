# 開発ガイド

この文書は、日常的な開発作業の入口である。
詳細な運用ルールは `rules/` と `AGENTS.md` を参照する。

## 読む場所

- project direction: `docs/vision/`
- architecture and roadmap: `docs/implementation/`
- project and service payloads: [data-models.md](data-models.md)
- coding conventions: `rules/coding-style.md`
- common commands and deployment flow: `rules/command.md`

## 環境構築

Python 環境と package 管理には `uv` を使う。

dependency group は責務ごとに分かれている。

- `embodiedlab`: 複数 service で使う shared library dependencies
- `server`: API service runtime dependencies
- `trainer`: training job runtime dependencies
- `notification`: WebSocket relay runtime dependencies
- `dev`: test、lint、local development tooling

通常は `embodiedlab` を shared base とし、必要な service group と組み合わせる。

全 group を同期する場合:

```bash
uv sync --frozen --all-groups
```

service 単位で同期する場合:

```bash
uv sync --frozen --group embodiedlab --group server
uv sync --frozen --group embodiedlab --group trainer
uv sync --frozen --group embodiedlab --group notification
```

Makefile 経由:

```bash
make local_setup
```

## ローカル開発

API をローカル起動する。

```bash
make server_local
```

一括 check を実行する。

```bash
make check
```

`make check` は以下を実行する。

```bash
uv run ruff check embodiedlab server trainer tests notification
uv run pymarkdown scan --recurse --respect-gitignore README.md AGENTS.md docs
uv run pytest
```

pytest のみを実行する場合:

```bash
make test
uv run pytest
```

## Manual End-To-End Flow

infrastructure が deploy 済みで `.env` が設定されている場合、
以下の流れで手動確認できる。

```bash
make submit
make train
make get_result
make get_result_ws
```

`make get_result_ws` は `.last_submission_id` を読み、
`tools/ws_client.py` を実行する。
`make submit` は `.last_submission_idempotency_key` と `.last_cancel_token` を先に生成し、
responseが失われても次回の実行で同じsubmissionを回収できる。新しいsubmissionを開始する
前に`make clear_submission_id`でこれらのlocal capability fileも削除する。

## `.env` の扱い

`.env` は repository に commit しない。

local check 系 target は `.env` なしで動作する。

cloud/API 系 target は、必要な環境変数が不足している場合、
`PROJECT_ID required` や `API_URL required` のように
不足名を表示して停止する。

## Model Artifacts

trainer job は完了した model artifact を `MODEL_BUCKET` の
`results/<submission_id>/` に upload する。

- `policy.zip`: Stable-Baselines3 saved model
- `policy.onnx`: deterministic continuous navigation policy の ONNX export
- `policy.sentis.onnx`: Unity Sentis-oriented continuous navigation ONNX export
- `replay/replay.jsonl`: EnvForge local replay 用の JSON Lines log

現在の bucket 作成処理は public object read を許可する。
これは prototype 用の挙動であり、user-specific result を扱う段階では
access control を見直す必要がある。

## `rules/` について

`rules/` は簡潔な coding / operational rules の置き場として残す。
`docs/` は、背景、設計判断、ロードマップ、説明文書の置き場とする。
