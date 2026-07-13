# EnvForge 連携ロードマップ

## 背景

EnvForge と EmbodiedLab は別リポジトリとして維持する。

EnvForge は、ユーザがシナリオを作成し、学習中の挙動を再生する
Unity アプリケーションである。

EmbodiedLab は、EnvForge から受け取ったシナリオをもとに
クラウド側で学習を実行し、成果物を保存し、状態更新を通知する
バックエンドである。

現在の EmbodiedLab は、FastAPI submission endpoint、Firestore persistence、
Cloud Run Job execution、Stable-Baselines3 PPO training、GCS artifact storage、
Pub/Sub event、WebSocket fan-out を持つ。
これは EnvForge 連携に使える学習実行基盤である。

## 現在の EmbodiedLab baseline

- 入力モデルは EnvForge Scenario Bundle へ置き換える。
- training backend は continuous navigation runtime と
  Stable-Baselines3 PPO である。
- 成果物は `results/<submission_id>/` に保存される。
- 保存される artifact は `policy.zip`、`policy.onnx`、
  `policy.sentis.onnx`、`replay/replay.jsonl` である。
- 結果通知は Firestore result document、Pub/Sub、WebSocket で行う。
- Notification service は WebSocket 接続時に Firestore の最新 result document を
  送信し、Pub/Sub event を取り逃がしても EnvForge が authoritative な
  状態へ復帰できるようにする。
- tests は API routes、trainer transitions、artifact flow、schemas、
  notification fan-out を cover している。

## Target Integration Shape

EnvForge などの Unity frontend は Unity build ではなく Scenario Bundle を送信する。
共通のジョブ投入、進捗監視、成果物取得には独立した `EmbodiedLab.Unity` package を
利用する。EmbodiedLab は Scenario Bundle を検証し、クラウド向きの学習環境へ変換し、
学習を実行し、Result Bundle を返す。

    EnvForge / another Unity frontend
      -> EmbodiedLab.Unity
      -> Scenario Bundle
      -> EmbodiedLab API
      -> Training Job
      -> Model artifacts + Replay Log
      -> Result Bundle
      -> EnvForge local replay and model use

## Phase 0: 基盤整備

Phase 0 は完了扱いとする。

完了済みの内容は以下である。

- docs 構成を EnvForge の運用に近づけた。
- `CLAUDE.md` を廃止し、`AGENTS.md` を作成した。
- Python 3.13 と `uv` を前提に依存関係を同期できることを確認した。
- Ruff の target version を Python 3.13 に合わせた。
- PyMarkdown を dev dependency に追加した。
- `make check` で Python lint、Markdown lint、pytest を実行できるようにした。
- `.env` がなくても local check は動くようにした。
- cloud/API 系 target では不足環境変数を明示して止めるようにした。

## Phase 1: Contract Definition

次の中心課題は、以下の契約を定義することである。

- Scenario Bundle
- Result Bundle
- Replay Log

各契約には schema version と compatibility metadata を含める。

source-of-truth の分担は以下とする。

- EmbodiedLab は外部 API contract、backend runtime representation、
  job result document を持つ。
- `EmbodiedLab.Unity` は contract の Unity DTO、serializer、compatibility check を持つ。
- EnvForge は user-facing scenario editor と replay visualization semantics を持つ。
- 各リポジトリは canonical fixture による contract 適合 test を持つ。

## Phase 2: Reward Components

continuous navigation runtime の reward を EnvForge Scenario Bundle の宣言的
reward component から構成する。

初期セットは小さく保つ。

- goal reached reward
- goal progress reward
- collision penalty
- obstacle proximity penalty
- step penalty
- stuck penalty
- zone entry reward or penalty

任意のユーザ提供 reward code は初期連携では扱わない。

## Phase 3: Replay Logs

EmbodiedLab は動画ではなく構造化 Replay Log を生成する。
EnvForge はこのログをローカルで再生し、ユーザが視点変更、
報酬イベント確認、失敗ケース分析をできるようにする。

初期 Replay Log は以下を含める。

- episode id
- step index
- robot position and rotation
- action
- total reward
- reward components
- termination reason
- collision or contact events
- compact sensor summaries

全観測画像は初期版では必須にしない。

Replay Log v0 は Unity `JsonUtility` で読み込める形を優先する。
action、reward components、sensor summaries は arbitrary dictionary ではなく
`name` / `value` を持つ配列として表す。

現在の continuous navigation runtime では、trainer evaluation の先頭 episode から
Replay Log steps を生成し、GCS replay artifact へ渡す。robot position、forward/turn
action、reward components、front_distance は EnvForge がそのまま再生・確認できる
contract field として扱う。

## Phase 4: Environment Upgrade

旧 grid-world 実装を廃止し、EnvForge-compatible scenario model へ移行する。

Unity をクラウドで実行することは必須ではない。
重要なのは、学習に必要な EnvForge のシナリオ条件を再現することである。

初期版では以下を support する。

- static walls
- static obstacles
- fixed robot type
- one forward-facing camera abstraction
- one distance sensor abstraction
- goal and episode termination conditions
- declarative reward components

dynamic obstacles、humans、curriculum learning、multiple robot types は
後続フェーズに送る。

ContinuousNavigationEnv と ContinuousNavigationSpec を主経路として使い、
Scenario Bundle から EnvForge x/z meter 座標を保った runtime spec へ
変換する。初期実装では、連続 action forward/turn、Y 回転、
goal radius、static walls、static obstacles、回転付き box collision、
距離センサ range を扱う。

行動契約は、修士論文「図 2.3 行動モデルのネットワーク構造」と
「表 3.1 報酬設計」に合わせる。EnvForge / EmbodiedLab の主経路では
`obs_0` を `3 x 84 x 112` の segmentation image、`obs_1` を
`[angle, distance]` の 2 値 numeric observation として扱う。
action は `[v, omega]` で、`v` は 0..1、`omega` は -1..1 とする。
exported ONNX もこの action 契約に揃える。

報酬は、目的地到達 +100、壁衝突 -50、action 選択ごと -0.01、
目的地に近づいた場合 +0.1、目的地方向角が ±90 度を超える場合 -0.1、
±150 度を超える場合 -5.0、`v` が 0 かつ `omega` が -0.3..0.3 の場合
-0.1 とする。`goal_progress` は距離差分への比例報酬ではなく、
近づいた step に対する固定報酬として扱う。

Stable-Baselines3 の policy は標準 `MultiInputPolicy` ではなく、
Figure 2.3 に対応する NavigationFinalPolicy を使う。画像 branch は
`Conv(3->16, kernel=8x8)`、`LeakyReLU`、`Conv(16->32, kernel=4x4)`、
`LeakyReLU`、`Flatten(3456)`、`FC(256)`、`LeakyReLU` とする。
numeric branch は標準化して image features と concat し、その後
`FC(256)`、Sigmoid gate、`FC(256)`、Sigmoid gate、`FC(2)` を通す。
ONNX export は deterministic action head として `v=sigmoid(raw_v)`、
`omega=clip(raw_omega,-3,3)/3` を公開する。

既定の trainer 経路は continuous navigation runtime へ切り替えた。
trainer は ContinuousNavigationSpec を学習し、policy.zip、通常 ONNX、
Unity Sentis 向け ONNX、Replay Log を主成果物として保存する。Sentis 向け
ONNX は `obs_0` (`batch x 3 x 84 x 112`) と `obs_1`
(`batch x 2`, angle / distance) から `[forward, turn]` の continuous action を返す。

## Phase 5: Model Compatibility

EmbodiedLab は内部で ML-Agents を使う必要はない。
ただし EnvForge に返す結果には、モデル互換性情報を含める。

- model format
- observation input layout
- action output layout
- robot version
- sensor version
- scenario schema version
- EnvForge binary compatibility

ONNX/Sentis ONNX は continuous 主経路の artifact として生成できる。
Result Bundle には EnvForge が artifact を取得し、input/output layout を
解釈するための metadata を含める。

Phase 5 の最小到達点は以下である。

- Result Bundle が通常 ONNX と Sentis ONNX の location を含む。
- Sentis ONNX が固定長 `float32[1,28226]` input layout を明示する。
- EnvForge が Result Bundle から Replay Log と model artifact を取得できる。
- EnvForge が Replay Log をローカル再生できる。

ただし最終判断は、EnvForge が安定して load / run できる形式に従う。

## Phase 6: Training Performance Without Contract Changes

学習速度改善では、EnvForge との行動・観測・報酬契約を変更しない。
特に以下は高速化のために削らない。

- `obs_0` の forward camera segmentation image
- `obs_1` の angle / distance numeric observation
- front distance sensor value
- NavigationFinalPolicy の network structure
- 現在の reward component と weight

2026-06-03 の benchmark では、1CPU/1Gi、`n_envs=1` の baseline は
10,000 steps あたり約 32 分であった。4CPU/4Gi、`n_envs=1` は
10,000 steps あたり約 31 分で、CPU を増やすだけではほぼ改善しなかった。
4CPU/4Gi、`n_envs=4` は診断ログ追加後に `sb3_first_step` と
10,000 step progress を確認できたが、10,000 steps あたり約 38 分 50 秒で
baseline より遅かった。

その後、単一 env の local profile では、`obs_0` の forward camera segmentation
image generation が step 時間の大半を占めることを確認した。Python の二重 loop
実装では `render_segmentation` が約 102 ms / call、`env.step` が約 65 ms /
call であった。同じ semantics のまま NumPy vectorization に置き換えた後は、
`render_segmentation` が約 0.16 ms / call、`env.step` が約 0.5 ms / call まで
改善した。front distance sensor は約 1 ms / call で、次の改善対象は collision
probe と sensor 周辺の geometry cache である。

geometry cache では、障害物の中心、half extents、rotation sin/cos を env
initialization 時に事前計算し、front distance sensor と movement collision
probe で再利用する。local profile では `front_distance` が約 1.05 ms / call
から約 0.04 ms / call、`env.step` が約 0.5 ms / call から約 0.24 ms / call
まで改善した。image generation は約 0.15 ms / call で維持できている。
同じ変更を trainer に deploy した後、MVP default 相当の `n_envs=1` Cloud Run
job では 10,000 step progress が約 51 から 52 秒間隔で安定した。検証用 job は
60,000 steps まで確認した後に cancel した。

次の比較では、`n_envs=4` の実行方式として `SubprocVecEnv` を使った場合の
process startup、pickle、step ごとの inter-process communication が現在の高速化済み
environment step に対して重すぎないかを確認する。これは reward、observation、
sensor / camera generation、NavigationFinalPolicy を変更せず、実行方式だけを比較する。
EnvForge ユーザーが vectorized environment の種類を選ぶ必要はないため、Scenario Bundle
には `training.vec_env` を持たせない。EmbodiedLab は `n_envs == 1` なら単一 env、
`n_envs > 1` なら `SubprocVecEnv` を自動選択する。

`n_envs=4` の local step-only benchmark では、`DummyVecEnv` が約 0.260 ms /
vector step、`SubprocVecEnv` が約 0.450 ms / vector step であり、純粋な
environment stepping だけを見ると Subproc の inter-process communication は明確な
overhead になった。一方、同じ image / sensor / reward / policy のまま 4CPU/4Gi の
Cloud Run trainer で PPO 学習込みにした benchmark では、`SubprocVecEnv` は
10,000 step progress が約 38.6 から 39.2 秒間隔で安定した。`SubprocVecEnv` の
process startup は約 62 秒かかったが、学習中の throughput は `n_envs=1` の
約 51 から 52 秒 / 10,000 steps より速い。今後の CPU / `n_envs` 比較では、
startup cost と steady-state throughput を分けて記録する。

PyTorch thread setting の軽量 benchmark では、4CPU/4Gi、`n_envs=4`、
`SubprocVecEnv` 固定で比較した。未指定時の Cloud Run 上の実効
`torch_num_threads` は 2 で、10,000 step progress は約 38 から 39 秒間隔で
安定した。`torch_num_threads=1` は約 52 から 54 秒 / 10,000 steps まで遅くなり、
採用しない。`torch_num_threads=2` を明示した場合は約 35 から 39 秒 /
10,000 steps で、未指定時と同等以上だった。したがって、MVP default の本命候補は
4CPU/4Gi、`n_envs=4`、`torch_num_threads=2` とする。`n_envs > 1` では
EmbodiedLab が `SubprocVecEnv` を自動選択する。EnvForge からは `cpu_count` を
Scenario Bundle に含めるが、Cloud Run Jobs API の run override では CPU / memory を
実行ごとに変更できないため、現時点の実CPUは deploy 済み job definition 側で決まる。
将来、EnvForge から CPU 数を実効反映する場合は、CPU別 job selection か job definition
更新の同期制御を別途設計する。
ただし、正式な default 昇格は同設定で MVP default scenario を完走させ、生成 model の
EnvForge 推論挙動を確認してから行う。

優先順位は以下とする。

1. `n_envs > 1` execution の診断ログを追加し、env 構築、reset、PPO model
   construction、`learn()` 開始、最初の progress callback のどこで止まるかを
   Cloud Run logs で確認する。
2. 単一 env の environment step profile を取り、camera image generation、
   front distance sensor、movement collision、PPO update のどこが支配的かを
   測る。
3. camera と sensor の出力を維持したまま、Python loop を NumPy vectorization や
   geometry cache に置き換える。出力 semantics は変えない。
4. obstacle / wall collision は、事前計算した rotation、half extents、
   bounding volume、必要に応じた spatial index で高速化する。ユーザが壁パーツを
   増やす将来機能でも線形劣化しすぎない形を目指す。
5. true parallel execution は `DummyVecEnv` ではなく `SubprocVecEnv` などを
   検討する。ただし Cloud Run 上の process startup、pickle、PyTorch thread count
   との相互作用を計測してから採用する。
6. PyTorch / SB3 の CPU thread setting は、policy architecture を変えずに調整する。
   `n_envs` と `n_steps` の組み合わせは、学習契約を変えない範囲で比較条件を明示して
   benchmark する。

この Phase では、学習負荷を下げるために sensor や camera を省略する最適化は行わない。
必要な観測を同じ意味で生成しつつ、実装と実行形態を高速化する。

## Phase 7: Unity SDK 分離

EnvForge 内の汎用 client 機能を、独立した `EmbodiedLab.Unity` リポジトリの
UPM package へ移す。対象は bundle DTO、HTTP API client、WebSocket result stream、
HTTP 再同期、artifact download、Replay Bundle の取得と parse、互換性検査である。

EnvForge には world editor、Scenario Bundle 構築、UI、ユーザ向け job history、
Replay の scene 表示、ONNX Runtime 推論を残す。詳細は
`docs/implementation/unity-sdk-roadmap.md` に記録する。

## Phase 8: Episode Environment Generation

既存の固定マップを `fixed` mode の既定動作として維持し、明示的に選択した場合だけ
`generated` mode を使う。`generated` mode は versioned な宣言的生成規則、seed、
制約から episode ごとの環境を構成する。

マップを複数領域へ分割し、領域ごとに候補の壁パーツをランダム選択する方式は一例であり、
特定の 4 分割方式を契約そのものにはしない。生成結果は Replay または関連 metadata に
記録し、episode を再現できるようにする。ユーザ提供の任意コード実行は対象外とする。

## Open Issues

- API contract の正本から Unity DTO を生成するか、fixture 適合 test で同期するか。
- `EmbodiedLab.Unity` の release、tag、package distribution をどう運用するか。
- `generated` mode の最初の schema と生成結果の記録先。
- Replay Bundle の巨大 chunk を全結合せずに読む streaming load。
- user-specific result を導入した後の GCS access をどうするか。
- public-read model artifact をいつまで許容するか。
- WSL2 `aarch64` 開発と Cloud Run `linux/amd64` deploy の差をどう扱うか。
