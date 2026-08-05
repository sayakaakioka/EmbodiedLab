# プロダクト方針

## 概要

EmbodiedLab は、身体性を持つ AI 実験のためのクラウド側学習基盤である。

現在の実装は、Scenario Bundle を受け取る end-to-end 学習ループを持つ。
Scenario Bundle を continuous navigation runtime へ変換し、Cloud Run Job 上で
Stable-Baselines3 PPO により方策を学習する。成果物は Google Cloud Storage に
保存し、Firestore、Pub/Sub、WebSocket を通して状態更新を届ける。

現在のプロダクト方針は、EmbodiedLab、`EmbodiedLab.Unity`、EnvForge を
明示的な契約で接続し、同じ学習結果を再利用可能な Unity SDK と専用 frontend の
両方から扱える状態を維持することである。
EnvForge は、ユーザがロボット学習用シナリオを作る Unity アプリである。
ユーザは壁や障害物を配置し、用意されたロボットとセンサを設定し、
報酬体系を定義し、シナリオをクラウドへ送信する。
その後、学習済みモデルと Replay Bundle をダウンロードし、
EnvForge 上で結果を確認する。

EmbodiedLab は、Unity や ML-Agents をクラウド上で実行する必要はない。
EnvForge から渡されたシナリオ条件を保持し、それをクラウド実行に適した
学習環境で十分に再現し、EnvForge が利用できる成果物を返すことが役割である。

## 境界

- EnvForge はシナリオ作成とリプレイ再生のアプリケーションである。
- EnvForge は Unity バイナリとしてユーザに配布される。
- EmbodiedLab はクラウド学習と成果物提供のバックエンドである。
- EmbodiedLab は EnvForge とは別リポジトリとして維持する。
- 両者は Scenario Bundle、Result Bundle、Replay Bundle で接続する。
- Unity 向け共通機能は、独立した `EmbodiedLab.Unity` リポジトリの UPM package とする。
- EnvForge は `EmbodiedLab.Unity` を利用する一つのフロントエンドとして扱う。

## 現在の方向性

現在は、API、ジョブ起動、成果物アップロード、状態保存、通知経路に加えて、
Scenario Bundle、continuous navigation runtime、宣言的 reward component、
Result Bundle、Replay Bundle、ONNX artifact の主経路まで実装済みである。

汎用 Unity client 機能は、独立した `EmbodiedLab.Unity` UPM package として
分離済みである。EmbodiedLab の Pydantic model と versioned JSON Schema を
wire contract の正本とし、SDK の generated DTO、canonical fixture、transport test を
同じ契約へ同期している。Scenario の実行値、学習値、観測 shape は JSON を正本とし、
downloadable artifact は bytes 数と SHA-256 を持つ。

`EmbodiedLab.Unity` は server-owned job lifecycle、artifact 検証、Replay 読み込み、
Windows x64 ONNX inference を提供し、固定環境の Quickstart は SDK の主要 API を順に
確認できるチュートリアルへ整理済みである。EnvForge も同じ SDK revision と厳密な
v0 contract へ移行し、旧 client、重複 DTO、重複 Replay／artifact 実装を削除済みである。

現在は、三つのリポジトリを横断する人間のレビューと end-to-end 検証を行い、
package version、tag、release、対応 platform の運用を確定する段階である。

その後、学習環境の生成モードを次の二つへ拡張する。

- `fixed`: 全 episode で同じマップを使う既定モード。
- `generated`: versioned な宣言的生成規則に従って episode ごとに環境を構成するモード。

生成規則には seed と選択結果を記録し、学習、評価、Replay の再現性を維持する。

## 次フェーズの非目標

- EnvForge Unity バイナリを EmbodiedLab 内で実行すること。
- クラウド学習基盤を ML-Agents に固定すること。
- 任意のユーザ提供 reward code を実行すること。
- ユーザ提供の C#、Python、その他の任意コードを学習ジョブ内で実行すること。
- 最初の連携 MVP で動的人間や動的障害物を扱うこと。
- 初期段階で認証、quota、billing を本番品質にすること。

## 設計原則

- EnvForge、EmbodiedLab、`EmbodiedLab.Unity` はリポジトリ単位で分離する。
- データ契約は明示的かつ versioned にする。
- 任意コードより宣言的なシナリオ・報酬設定を優先する。
- 固定マップと生成マップを暗黙に切り替えず、Scenario Bundle で明示する。
- リプレイは動画ではなく構造化ログとして保存する。
- EnvForge がモデルやリプレイの互換性を検証できる metadata を残す。
- ダウンロード対象の size と digest を契約に含め、利用前に実 bytes と照合する。
- 学習時に解決した library、hyperparameter、resource 値を Result Bundle に残す。
- 成果物は cloud storage に保存し、状態は Firestore に保存する。
