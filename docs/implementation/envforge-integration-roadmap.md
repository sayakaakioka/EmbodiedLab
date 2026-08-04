# EnvForge 連携ロードマップ

## 目的と責務

EmbodiedLab は、Scenario Bundle を受け取り、学習を実行し、Result Bundle を返す
クラウドバックエンドである。外部 API とデータ契約の正本を持つ。

`EmbodiedLab.Unity` は、契約 DTO、HTTP / WebSocket client、成果物取得、
Replay Bundle 読み込みを Unity 向け公開 API として提供する。

EnvForge は、シナリオ編集、ジョブ履歴、結果可視化、ローカル推論を提供する
Unity アプリケーションであり、`EmbodiedLab.Unity` を利用する。

```text
EnvForge / another Unity frontend
  -> EmbodiedLab.Unity
  -> EmbodiedLab API
  -> Cloud training
  -> Model artifacts + Replay Bundle
  -> local replay and inference
```

## 現在の baseline

- FastAPI、Firestore、Cloud Run Jobs、GCS、Pub/Sub、WebSocket で学習ライフサイクルを
  実行する。
- Scenario Bundle を continuous navigation runtime へ変換し、Stable-Baselines3 PPO と
  `NavigationFinalPolicy` で学習する。
- 通常 ONNX は `obs_0` と `obs_1`、Sentis ONNX は固定長 `observation` を入力とする。
- 成果物は `results/<submission_id>/model/` と
  `results/<submission_id>/replay/` に保存する。
- Replay Bundle は manifest と train / eval の gzip JSONL chunk で構成する。
- EmbodiedLab の Pydantic model と versioned JSON Schema を wire contract の正本とし、
  SDK の generated DTO、canonical fixture、contract test を同期する。
- `EmbodiedLab.Unity` の契約同期と最小チュートリアルを実装済みである。

現行の Scenario、Result、Replay、ONNX 契約の詳細は `contracts/v0/`、
`tests/fixtures/envforge/`、`docs/implementation/unity-sdk-roadmap.md` を正本として参照する。
過去の段階的な実装経緯は Git 履歴と pull request に残す。

## 次の実装順序

### 1. server-owned job lifecycle

submission 作成後の training dispatch、失敗記録、再試行判断はサーバー側で完結させる。
Unity client は一つの submit 操作を呼び、submission id と authoritative な Result を
受け取る。クラウド内部の二段階操作や部分失敗の後処理を client に要求しない。

cancel は永続化された capability と terminal state に基づいて安全に処理する。

### 2. Unity 公開 API の整理

`EmbodiedLab.Unity` は tutorial 固有の補助クラスへ責務を隠さず、次を小さな公開 API として
提供する。

- job submit、状態監視、再同期、cancel
- model / replay artifact の取得と検証
- Replay Bundle の逐次読み込み
- ONNX metadata と Unity 推論入力の検証
- semantic camera と観測組み立ての再利用可能な境界

公開 API は server-owned lifecycle を表現し、artifact の size と SHA-256 を検証する。
HTTP response、Scenario / Replay の要素数と文字列長には明示的な上限を設ける。

### 3. 契約値の明示

実行結果へ影響する既定値をコードの magic number にしない。寸法、goal radius、camera、
解像度、semantic mode、PPO、environment 数、CPU、PyTorch thread 数など、ユーザーが
将来調整する値は Scenario Bundle に明記し、runtime と SDK はその値から構成する。

ONNX の入出力 shape と layout は metadata だけでなく実 graph と照合する。

### 4. EnvForge の SDK 移行

公開 API と tutorial の整理後、EnvForge の重複 client / contract / replay / inference
実装を `EmbodiedLab.Unity` 利用へ置き換える。移行時は以下を横断検証する。

- canonical Scenario と generated DTO の一致
- submit から Result の terminal state までの監視
- model と Replay Bundle の取得、digest 検証、ローカル再生
- `policy.onnx` を使うローカル推論
- Unity 2022.3.19f1 と Unity 6.3 LTS の対応範囲

### 5. 公開運用の hardening

- 認証、所有者単位の authorization、quota、billing
- private GCS と期限付き artifact access
- 学習計算量、payload、download、Replay 展開の resource limit
- Cloud Run / Firestore / GCS の保持と削除運用

cloud resource を削除する前に、EnvForge 側の
`docs/implementation/cloud-result-retention.md` と保持台帳を確認する。

## 将来候補

- versioned な宣言的規則と seed を持つ generated environment mode
- Replay Bundle の streaming load と部分取得
- 複数 robot / sensor 構成
- CPU 別 Cloud Run Job 選択または安全な job definition 更新
- SDK の release、tag、UPM package distribution

任意のユーザー提供 C# / Python code のクラウド実行は対象外とする。
