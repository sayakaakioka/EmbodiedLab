# サブエージェント運用

## 目的

EmbodiedLab の開発規模が大きくなったときに、親エージェントが方針と統合を
担当し、サブエージェントを短命の専門タスクに使うための運用ルールを
まとめる。

サブエージェントは常駐チームではなく、必要な局面で起動する補助役として
扱う。最終判断、統合、ユーザへの確認、commit や push の判断は親エージェントが
担当する。

## 親エージェントの責務

- Google Drive の `codex/AGENTS.md` とこのリポジトリの `AGENTS.md` を確認する。
- `docs/vision` と `docs/implementation` の現在方針を読む。
- 作業を分解し、サブエージェントに渡す範囲を決める。
- 重要判断はユーザに確認する。
- サブエージェントの結果を統合し、矛盾や重複を解消する。
- 最終的な実装、検証、ドキュメント更新、commit 候補を管理する。

## 役割

### EmbodiedLab Context Scout

使うタイミング:

- Phase 開始時。
- API、trainer、artifact、notification、docs の現状を整理したいとき。
- EnvForge との契約境界を確認したいとき。

読む範囲:

- `docs/vision`
- `docs/implementation`
- `embodiedlab`
- `server`
- `trainer`
- `notification`
- `tests`
- `Makefile`
- `pyproject.toml`

出力:

- 現状
- 変更候補
- リスク
- 未確認事項

### Implementation Worker

使うタイミング:

- 方針が決まっていて、編集範囲が明確なとき。
- 他の worker と書き込み範囲が重ならないとき。

指示に含めるもの:

- 担当ファイルまたは担当モジュール。
- 触ってよい範囲。
- 触ってはいけない範囲。
- 期待する検証方法。
- 他の作業者の変更を revert しないこと。

出力:

- 変更ファイル
- 実装内容
- 検証結果
- 残件

### Review Scout

使うタイミング:

- 実装後。
- PR 前。
- API contract、trainer runtime、artifact 生成の不整合がありそうなとき。

見る観点:

- 仕様とのズレ。
- Pydantic model と保存 payload の不整合。
- FastAPI route と repository protocol の不整合。
- trainer job、progress transition、artifact upload の失敗経路。
- `make check` で捕まらない test gap。
- ドキュメントと実装の食い違い。

出力:

- severity 順の finding
- test gap
- residual risk

#### WSL2 リポジトリ向けの注意

EmbodiedLab は WSL2 側で開発することがあるため、親エージェントの
作業ディレクトリが Windows 側にある場合、Review Scout にはアクセス方法を
明示する。

Review Scout を起動するときは、以下をプロンプトに含める。

- 作業ツリーは EmbodiedLab のリポジトリルートであること。
- 必要に応じて、現在の checkout の絶対パスを補助情報として渡すこと。
- PowerShell 側から確認する場合は、次の形でコマンドを実行すること。

    wsl -e bash -lc "cd EMBODIEDLAB_CHECKOUT && 実行するコマンド"

- 先に親エージェントが確認した `git status --short`、`git diff --stat`、
  対象ファイル一覧。
- 対象ファイルを絞り、広い探索を避けること。
- 詰まった場合は、長時間探索せず「何に詰まったか」を短く返すこと。

EmbodiedLab の Review Scout は、必要なら `fork_context: true` で起動する。
特に EnvForge 側から作業している会話では、WSL2 のリポジトリ配置、許可済み
コマンド、直前の検証結果を引き継がせた方がよい。

親エージェントは、Review Scout が戻らない場合に備えて、先に以下を手元で
確認しておく。

- `git status --short`
- `git diff --stat`
- review 対象ファイルの本文または主要 diff
- `make check` の結果

一定時間戻らない場合は、該当 Scout を停止し、親エージェントの確認と
テスト結果で代替してよい。ただし、停止した理由と代替した確認内容を
ユーザに報告する。

### Security Scout

使うタイミング:

- Cloud Run、Firestore、Pub/Sub、GCS、外部入力、成果物公開が絡むとき。
- `.env`、credential、token、signed URL、IAM、bucket policy を扱うとき。

見る観点:

- secret や `.env` の変更、漏えい、commit 混入。
- ユーザ提供 Scenario Bundle、reward spec、artifact path の検証不足。
- 任意コード実行につながる経路。
- GCS object の公開範囲。
- Firestore document や Pub/Sub message の権限境界。
- Cloud Run Job の環境変数と service account 権限。

出力:

- severity 順の finding
- 悪用シナリオ
- 推奨修正
- 保留できるリスク

### Refactor/Docs Worker

使うタイミング:

- 機能実装と検証が一段落したあと。
- PR 前に構造やドキュメントを整えるとき。

担当:

- 重複整理。
- 命名整理。
- 小さな責務分離。
- docs の追従。
- コメントや README の整理。
- test helper や fake repository の整理。

## 禁止事項

- 複数 worker に同じファイル群を同時に編集させない。
- サブエージェントに commit、push、PR 作成を任せない。
- サブエージェントに `.env` を変更させない。
- reviewer に勝手な修正をさせない。
- explorer に広すぎる調査を渡さない。
- 親エージェントがサブエージェント結果を未統合のまま採用しない。

## 起動プロンプト例

### プロンプト例: Context Scout

EmbodiedLab の Context Scout として、現在の `docs/vision`、
`docs/implementation`、API、trainer、notification、tests の関連ファイルを
読んでください。今回のテーマは `<テーマ>` です。出力は「現状」
「変更候補」「リスク」「未確認事項」に限定してください。ファイルは
変更しないでください。

### プロンプト例: Implementation Worker

EmbodiedLab の Implementation Worker として、`<担当範囲>` を実装してください。
編集してよい範囲は `<ファイルまたはディレクトリ>` です。ほかの作業者の
変更を revert しないでください。完了後に、変更ファイル、実装内容、検証結果、
残件を報告してください。

### プロンプト例: Review Scout

EmbodiedLab の Review Scout として、現在の差分をレビューしてください。
特に API contract、Firestore payload、trainer runtime、artifact upload、
progress transition、test gap を確認してください。出力は severity 順の
finding、test gap、residual risk にしてください。ファイルは変更しないでください。

### プロンプト例: Security Scout

EmbodiedLab の Security Scout として、現在の差分を確認してください。
特に `.env`、credential、Cloud Run、Firestore、Pub/Sub、GCS、ユーザ提供
Scenario Bundle、artifact path の扱いを見てください。出力は severity 順の
finding、悪用シナリオ、推奨修正、保留できるリスクにしてください。ファイルは
変更しないでください。
