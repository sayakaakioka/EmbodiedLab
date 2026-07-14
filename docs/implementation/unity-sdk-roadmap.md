# EmbodiedLab Unity SDK ロードマップ

## 目的

EnvForge に直接実装されている EmbodiedLab client 機能を独立した Unity Package へ
移し、EnvForge 以外の Unity プロジェクトも EmbodiedLab のジョブ投入、進捗監視、
成果物取得を利用できるようにする。

新規リポジトリ名は `EmbodiedLab.Unity`、UPM package ID は
`com.embodiedlab.unity` を第一候補とする。

## リポジトリ境界

### EmbodiedLab

- 外部 API と versioned contract の正本
- submission、result、job execution、notification
- 学習環境、学習、評価、artifact 生成
- 宣言的な episode environment generation の検証と実行

### EmbodiedLab.Unity

- Scenario / Result / Replay Bundle の Unity DTO と serializer
- submission 作成、training 開始、result 取得
- WebSocket 監視と HTTP による authoritative state への再同期
- artifact location の解決と download
- Replay Bundle manifest / chunk の取得、展開、parse
- API error、retry、timeout、cancellation、compatibility check
- UPM package、Unity 向け test、最小サンプル

Cloud Run、Firestore、Pub/Sub、GCS は SDK の公開 API に露出させない。

### EnvForge

- world editor と Unity scene
- scene から Scenario Bundle を構築する処理
- クラウド操作 UI とユーザ向け job history
- Replay の可視化と操作
- ONNX Runtime 推論
- EnvForge 固有の保存先と画面状態

Replay の取得、展開、parse は SDK、Unity scene 上での再生は EnvForge の責務とする。

## 設計・移行原則

- 現在必要な最小限の公開 API だけを設計する。
- 将来の可能性だけを理由に抽象化、extension point、汎用 layer を追加しない。
- 重複実装、不要な互換 layer、到達不能な旧コードを残さない。
- 既存動作を contract fixture と test で固定してから、責務を段階的に切り出す。
- 各段階で対象リポジトリの test と lint を実行し、成功を次段階の開始条件とする。
- 公開 API、contract、責務境界などの設計判断が必要な場合は、2 から 3 案の
  メリットとデメリット、推奨案と推奨理由を提示する。
- 設計判断はユーザとの合意前に実装しない。

## 学習環境モード

既存の固定マップを壊さず、Scenario Bundle で次の mode を明示する。

- `fixed`: 全 episode で同じ world definition を使う。既定値。
- `generated`: 宣言的な生成規則から episode ごとに world definition を構成する。

4 分割したマップの各領域に壁パーツをランダム配置する方式は `generated` mode の
一例であり、専用機能として固定しない。最初の schema は、領域、候補パーツ、選択方式、
seed、制約を表現できる最小範囲から設計する。選択結果は Replay または関連 metadata に
記録し、同じ episode を再現できるようにする。

ユーザ提供の C# や Python の任意コード実行は、このロードマップの対象外とする。

## Contract 生成方針

EmbodiedLab の Pydantic model を wire contract の正本とし、versioned JSON Schema を
`contracts/v0` へ決定的に出力する。Scenario Bundle、Result Document、Result Bundle、
Replay Bundle、Replay Log と API response を対象とする。

`EmbodiedLab.Unity` はこの JSON Schema から C# DTO を生成する。生成済み DTO は
UPM package に含め、package 利用者に generator の実行を要求しない。通信と job lifecycle
を扱う公開 API は手書きとし、生成された REST client をそのまま公開 API にしない。
canonical fixture は code generation 後も cross-repository の適合 test として維持する。
contract の再生成と差分検査には次を使う。

    uv run python -m tools.export_contract_schemas
    uv run python -m tools.export_contract_schemas --check

## 実装順序

1. EmbodiedLab API と現在の EnvForge client の動作を fixture と test で固定する。
2. `EmbodiedLab.Unity` リポジトリと UPM package の最小構造を作る。
3. DTO、HTTP submission、training start、result fetch を移す。
4. WebSocket 監視、HTTP 再同期、artifact download、Replay Bundle 読み込みを移す。
5. EnvForge を SDK 利用へ変更し、移行済みの重複コードを削除する。
6. 三者の compatibility と contract fixture test を追加する。
7. `fixed` mode を明文化した後、`generated` mode の schema と runtime を追加する。
8. EnvForge または別サンプルから両 mode を選択して投入できるようにする。

各段階は独立した Issue と PR にし、受入条件、非対象、変更可能範囲、検証コマンドを
Issue 本文に記載する。Codex は実装、test、lint、review、文書追従まで進め、公開 API、
互換性、schema の重要判断が必要な場合だけユーザ確認を求める。

## 人間の確認点

人間の判断を次の三点に限定する。

1. リポジトリ境界と責務分担。
2. SDK の最小公開 API と contract versioning。
3. EnvForge 移行後の操作と end-to-end 結果。

## 進行状況

- Issue #24 で、SDK が読む completed Result Document と Replay Bundle manifest の
  canonical fixture を追加し、現行 wire format を test で固定した。
- Issue #26 で、現行 API response と Replay Bundle を Pydantic model に結び付け、
  Unity DTO 生成元となる versioned JSON Schema の公開に着手した。
- Issue #28 で、合意した `T-B2 + C-B` の前提となる capability token 付き
  クラウドジョブキャンセル、正確な Cloud Run Execution name の保存、
  `cancelling` / `cancelled` 契約と WebSocket 通知を実装する。

状態監視は WebSocket を通常経路とし、接続失敗、切断、無通信、明示更新時だけ
HTTP の Result Document へ再同期する。正常な WebSocket 接続中に定期 HTTP polling は
行わない。ジョブ作成、学習開始、キャンセル、artifact download は一回性の
request/response として HTTP を使う。

## 保留事項

- SDK repository の公開範囲、release、tag、package distribution の運用。
- C# DTO generator の固定方法と、生成差分を SDK 側で検証する CI。
- 一般ユーザ認証導入後の token storage と signed artifact URL。
- quota、cost control。
- `generated` mode の最初の schema と、生成結果を Replay のどこへ記録するか。
