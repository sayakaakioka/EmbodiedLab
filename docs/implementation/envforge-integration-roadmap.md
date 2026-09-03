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
- ONNX の input 名と shape は Scenario から導出する。canonical fixture では
  `obs_0` と `obs_1` である。
- 成果物は `results/<submission_id>/model/` と
  `results/<submission_id>/replay/` に保存する。
- Replay Bundle は manifest と train / eval の gzip JSONL chunk で構成する。
- `randomize_start` の開始可能面積を submission 受理前に近似検証し、受理後は有限回抽選と
  検証済みの決定的な安全位置で reset を完了する。
- EmbodiedLab の Pydantic model と versioned JSON Schema を wire contract の正本とし、
  SDK の generated DTO、canonical fixture、contract test を同期する。
- `EmbodiedLab.Unity` は現在の厳密な v0 schema、generated DTO、semantic validator、
  bounded artifact／Replay reader、段階的チュートリアルへ同期済みである。
- EnvForge は同じ SDK revision を固定し、cloud job、artifact、Replay、local inference の
  主導線を SDK 利用へ移行済みである。

現行の Scenario、Result、Replay、ONNX 契約の詳細は `contracts/v0/`、
`tests/fixtures/envforge/`、`docs/implementation/unity-sdk-roadmap.md` を正本として参照する。
過去の段階的な実装経緯は Git 履歴と pull request に残す。

## 実装順序と進捗

### 1. server-owned job lifecycle

完了。submission 作成を一つの server-owned 操作として受理し、dispatch、調停、cancel、
terminal Result までを server と trainer が所有する。

submission 作成後の training dispatch、失敗記録、再試行判断はサーバー側で完結させる。
Unity client は一つの submit 操作を呼び、submission id と authoritative な Result を
受け取る。クラウド内部の二段階操作や部分失敗の後処理を client に要求しない。

cancel は永続化された capability と terminal state に基づいて安全に処理する。

### 2. Unity 公開 API の整理

公開 API と厳密な Result / artifact contract への追従は実装済みである。

`EmbodiedLab.Unity` は tutorial 固有の補助クラスへ責務を隠さず、次を小さな公開 API として
提供する。

- job submit、状態監視、再同期、cancel
- model / replay artifact の取得と検証
- Replay Bundle の逐次読み込み
- ONNX metadata と Unity 推論入力の検証
- semantic camera と観測組み立ての再利用可能な境界

公開 API は server-owned lifecycle を表現し、artifact の size と SHA-256 を検証する。
HTTP response、Scenario / Replay の要素数と文字列長には明示的な上限を設ける。
JSON Schema だけでは表現できない train chunk の
`start_step <= end_step`、`checkpoint_step == end_step`、Replay path の一意性は、
generated DTO の後段に置く Unity semantic validator と contract test で検証する。

### 3. 契約値の明示

EmbodiedLab 側は完了。action step、camera、numeric goal input、reward 判定値、PPO、
resource、Replay 設定を Scenario Bundle から runtime へ渡し、解決済み training 設定を
Result Bundle に記録する。ONNX export は保存済み policy と Scenario の observation contract
が一致しない場合に失敗する。

ONNX、Replay manifest と Replay chunk は `size_bytes` と `sha256` を持つ。
同じ schema、fixture、download 検証を `EmbodiedLab.Unity` と EnvForge へ同期済みである。

実行結果へ影響する既定値をコードの magic number にしない。寸法、goal radius、camera、
解像度、semantic mode、PPO、environment 数、CPU、PyTorch thread 数など、ユーザーが
将来調整する値は Scenario Bundle に明記し、runtime と SDK はその値から構成する。

ONNX の入出力 shape と layout は metadata だけでなく実 graph と照合する。

### 4. EnvForge の SDK 移行

完了。EnvForge は `EmbodiedLab.Unity` の確定済み SDK revision を固定し、cloud job、
artifact、Replay、local inference の主導線を SDK 公開 API 利用へ移行した。

重複 client、contract DTO、artifact download、Replay parse、ONNX Runtime binary は
移行後に削除した。以下は移行済み構成を維持するための横断検証項目である。

- canonical Scenario と generated DTO の一致
- submit から Result の terminal state までの監視
- model と Replay Bundle の取得、digest 検証、ローカル再生
- `policy.onnx` を使うローカル推論
- Unity 2022.3.19f1 と Unity 6.3 LTS の対応範囲

現在は三つのリポジトリと Quickstart を人間が順にレビューし、実 cloud job を使う
end-to-end 操作と対象 platform の build／inference を確認する段階である。

### 5. 公開運用の hardening

- Scenario、Result、Replay の現在の wire contract には要素数、文字列長、training resource
  の上限を実装済みである。以下は service 公開前に残る運用・転送境界の課題である。
- 認証、所有者単位の authorization、quota、billing
- private GCS と期限付き artifact access
- HTTP body、artifact download、Replay 展開後 bytes の service-level resource limit
- Cloud Run / Firestore / GCS の保持と削除運用

cloud resource を削除する前に、EnvForge 側の
`docs/implementation/cloud-result-retention.md` と保持台帳を確認する。

## 将来候補

- versioned な宣言的規則と seed を持つ generated environment mode
- Replay Bundle の streaming load と部分取得
- 複数 robot / sensor 構成
- `DistanceSensor` の LiDAR 的な angular scan。現行は前方1本だけである。versioned contract
  として水平視野角（180度／360度など）、Ray 数または角度分解能、最小／最大距離、出力順、
  必要なら mount height と垂直 layer を定義し、EmbodiedLab runtime、Replay、
  `EmbodiedLab.Unity`、EnvForge を同時に更新する。また、現行の Ray 進行方向 0.005 m
  point sampling は、その間に収まる薄い障害物を見逃し得る。角度分解能とは別の課題として、
  解析的な ray-object intersection など、薄い形状を落とさない距離判定へ置き換える。
- CPU 別 Cloud Run Job 選択または安全な job definition 更新
- SDK の release、tag、UPM package distribution

任意のユーザー提供 C# / Python code のクラウド実行は対象外とする。
