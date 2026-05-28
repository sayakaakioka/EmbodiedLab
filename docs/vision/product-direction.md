# プロダクト方針

## 概要

EmbodiedLab は、身体性を持つ AI 実験のためのクラウド側学習基盤である。

現在の実装は、最小限の end-to-end 学習ループを持つ。
grid-world の training submission を受け取り、Cloud Run Job を起動し、
Stable-Baselines3 PPO で方策を学習し、成果物を Google Cloud Storage に
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
- 両者は Scenario Bundle、Result Bundle、Replay Log で接続する。

## 現在の方向性

現在の grid-world 実装は、最終的な環境モデルではなく、
クラウド学習ループの試作品として扱う。

価値があるのは、API、ジョブ起動、成果物アップロード、状態保存、
通知経路がすでに検証されている点である。

次のフェーズでは、この基盤を維持しつつ、ドメインモデルを
EnvForge 連携向けに拡張する。

- grid-world submission から EnvForge Scenario Bundle へ移行する。
- 固定 reward logic から宣言的 reward component へ移行する。
- scalar summary だけでなく Replay Log を出力する。
- simple robot descriptor を versioned robot / sensor spec に拡張する。
- model artifact を compatibility metadata 付き Result Bundle に整理する。

## 次フェーズの非目標

- EnvForge Unity バイナリを EmbodiedLab 内で実行すること。
- クラウド学習基盤を ML-Agents に固定すること。
- 任意のユーザ提供 reward code を実行すること。
- 最初の連携 MVP で動的人間や動的障害物を扱うこと。
- 初期段階で認証、quota、billing を本番品質にすること。

## 設計原則

- EnvForge と EmbodiedLab はリポジトリ単位で分離する。
- データ契約は明示的かつ versioned にする。
- 任意コードより宣言的なシナリオ・報酬設定を優先する。
- リプレイは動画ではなく構造化ログとして保存する。
- EnvForge がモデルやリプレイの互換性を検証できる metadata を残す。
- 成果物は cloud storage に保存し、状態は Firestore に保存する。
