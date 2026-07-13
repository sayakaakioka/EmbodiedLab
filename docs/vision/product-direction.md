# プロダクト方針

## 概要

EmbodiedLab は、身体性を持つ AI 実験のためのクラウド側学習基盤である。

現在の実装は、EnvForge Scenario Bundle を受け取る end-to-end 学習ループを持つ。
Scenario Bundle を continuous navigation runtime へ変換し、Cloud Run Job 上で
Stable-Baselines3 PPO により方策を学習する。成果物は Google Cloud Storage に
保存し、Firestore、Pub/Sub、WebSocket を通して状態更新を届ける。

次のプロダクト方針は、EmbodiedLab を EnvForge と接続することである。
EnvForge は、ユーザがロボット学習用シナリオを作る Unity アプリである。
ユーザは壁や障害物を配置し、用意されたロボットとセンサを設定し、
報酬体系を定義し、シナリオをクラウドへ送信する。
その後、学習済みモデルと Replay Log をダウンロードし、
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
EnvForge Scenario Bundle、continuous navigation runtime、宣言的 reward component、
Result Bundle、Replay Bundle、ONNX artifact の主経路まで実装済みである。

次のフェーズでは、EnvForge に直接実装されている汎用 Unity client 機能を
`EmbodiedLab.Unity` へ切り出し、複数の Unity フロントエンドから EmbodiedLab を
利用できるようにする。その後、学習環境の生成モードを次の二つへ拡張する。

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
- 成果物は cloud storage に保存し、状態は Firestore に保存する。
