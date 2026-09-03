# 現在のアーキテクチャ

## 概要

EmbodiedLab は現在、Scenario Bundle を continuous navigation
runtime へ変換し、クラウド上で PPO 学習する最小限の学習ループを実装している。

FastAPI service が submission を受け取り、Firestore に保存し、
Cloud Run Job を起動する。trainer は Stable-Baselines3 PPO で
方策を学習し、model artifact を Google Cloud Storage にアップロードし、
結果状態を Firestore に書き戻し、Pub/Sub と WebSocket relay を通して
更新を通知する。

## Runtime Flow

    Client
      -> POST /submissions
          -> SDK は idempotency key と cancellation capability を生成して送る
          -> 同じ key、scenario、capability の再試行は同じ submission を返す
          -> client が保持する cancellation capability を response に返す
          -> submission と queued Result を batch で原子的に作成
          -> private dispatch state を pending から一度だけ claim
          -> Cloud Run Job with SUBMISSION_ID override
              -> Operation metadata の正確な Execution name を submission に保存
              -> response 消失時は trainer が Cloud Run runtime の execution ID から
                 同じ exact name をCAS保存してから学習を開始
              -> Firestore submission lookup
              -> Continuous navigation PPO training
              -> GCS artifact upload
              -> Firestore result update
              -> Pub/Sub event
                  -> notification service push endpoint
                      -> WebSocket subscribers
      -> POST /submissions/{submission_id}/cancel
          -> cancellation capability を検証
          -> 保存済みの正確な Cloud Run Execution を cancel
          -> cancelling / cancelled event を Pub/Sub へ publish
      -> GET /results/{submission_id}
      -> WebSocket /ws/results/{submission_id}

## Services

### `server/`

FastAPI API service である。
submission の受理、result document の queued 化、Cloud Run Job の起動とキャンセル、
result document の返却を担当する。キャンセルは submission ごとの capability token で
保護し、Firestore には SHA-256 hash だけを保存する。private control の期限付き
cancellation intent leaseとfencing tokenで同時RPCを一つにし、期限切れownerが
新しいleaseを変更することを防ぐ。最初のRPCが受付結果を返さず不明になった場合だけ、
lease期限後に同じexact executionへのcancelを再送し得る。Cloud RunがOperationを
返したrequestは再送せず、exact executionから調停する。公開Resultを排他lockには
使わない。
submission response が失われた場合、client は同じ `Idempotency-Key` と
`X-EmbodiedLab-Cancel-Token` で再試行する。server は同一 request を同じ submission へ
解決し、異なる scenario または capability での key 再利用を拒否する。
submission と queued Result は batch で原子的に作成し、server が private dispatch state を
一度だけ claim して Cloud Run Job を開始する。確定拒否は terminal `failed` Result にし、
RPC timeout や response loss は別 execution を自動起動せず `ambiguous` として調停する。
trainer は `CLOUD_RUN_JOB`、`CLOUD_RUN_EXECUTION`、`REGION` から
exact execution name を再構成し、dispatch が未解決の間だけCAS保存する。
既にdispatchがterminalなら学習を
開始しないため、terminal Resultは`starting`や`running`へ復活しない。

主な endpoint は以下である。

- `POST /submissions`
- `POST /submissions/{submission_id}/cancel`
- `GET /results/{submission_id}`

Cloud Run Job が timeout などで Python trainer の cleanup 前に終了した場合、
trainer 自身は Firestore result を更新できない。このため API は
`GET /results/{submission_id}` で active status
（`queued`、`starting`、`running`、`cancelling`）の result を返す前に、学習開始時に
保存した正確な Cloud Run Execution を取得する。対応 execution が失敗済み、または
正常終了したのに terminal Result がなければ `failed`、キャンセル済みなら
`cancelled` に更新して Pub/Sub へ publish する。
recent execution の走査や `SUBMISSION_ID` override による推測は行わない。

### `trainer/`

Cloud Run Job service である。
submission を読み、training spec に変換し、PPO training を実行し、
artifact をアップロードし、Firestore を更新し、status event を publish する。
trainer job の task timeout は約 1 日を想定し、Makefile の
`TRAINER_TASK_TIMEOUT` で `24h` を既定値にしている。

### `notification/`

FastAPI WebSocket relay service である。
Pub/Sub push event を受け取り、submission id に対応する WebSocket client へ
broadcast する。

### `embodiedlab/`

schemas、result models、repository protocols、continuous navigation environment、
training converter、training runner を含む shared library である。

## 現在のデータモデル

現在の API は Scenario Bundle を受け取る。主経路の環境は、
Gymnasium-compatible continuous navigation runtime である。action は PPO 内部では
raw `forward` と raw `turn` を分けて扱い、runtime 適用時は
`forward=sigmoid(raw_forward)`、`turn=clip(raw_turn,-3,3)/3` に写像される。
observation は Scenario Bundle で指定した名前、解像度、semantic mode から構成する
semantic camera と、同じく Scenario Bundle で順序を指定した
`[goal_angle_degrees, goal_distance_meters]` である。固定 tutorial の現在値は
`obs_0: 3 x 84 x 112`、`obs_1: 2` だが、runtime や policy network にこの shape を
重複して直書きしない。

`training.randomize_start` が `true` の Scenario は、submission の永続化と job 起動より
前に開始可能領域を検証する。world bounds から1.35 m内側の一様抽選矩形を32 x 32の
固定 grid で評価し、robot radius を含む障害物の衝突領域から0.65 m以上、かつ goal
領域から0.65 m以上離れた点の割合を開始可能面積率として近似する。8%未満なら
`422 Unprocessable Entity` とする。

受理済み Scenario の runtime は、同じ点ごとの安全判定を使って最大512回抽選する。
すべて外れた場合は、受理時の grid で確認した開始可能点のうち抽選矩形の中心に最も近い
決定的な点を使う。宣言された固定 `start_pose` へ暗黙に戻す旧 fallback は使わない。

## 現在の成果物

trainer job が完了すると、以下の成果物をアップロードする。

    results/<submission_id>/
      model/
        policy.onnx
      replay/
        manifest.json
        train/chunk_<index>.jsonl.gz
        eval/checkpoint_<step>.jsonl.gz

`policy.onnx` は continuous navigation の dict observation を2 input として公開する
opset 18 の ONNX artifact である。input 名、shape、layout は Scenario Bundle から導出し、
保存済み policy の observation space と一致しなければ export を失敗させる。output は
`[forward, turn]` の continuous action である。Replay Bundle は manifest と
gzip 圧縮した JSON Lines chunk からなり、各行は `scenario_id` と `job_id` を含む
`ReplayLogStep` として書き込み前に検証される。Result Bundle には ONNX の
artifact location、target、opset、全 input/output metadata を含める。

ダウンロード対象の ONNX、Replay manifest はすべて `size_bytes` と
`sha256` を持つ。Replay manifest の各 chunk も圧縮後 bytes の size と digest を持つ。
trainer は upload 前に実ファイルを検証し、GCS object は generation precondition 付きで
新規作成する。Replay gzip は同じ入力から同じ digest を得られるよう timestamp を固定する。

## 現在の強み

- API、trainer、notification、shared models が分離されている。
- Firestore-backed submissions と results がある。
- Cloud Run Job execution path が存在する。
- Artifact upload path が存在する。
- Pub/Sub と WebSocket の status update path が存在する。
- Repository protocols と fake repositories により core flow が testable である。

## 現在の連携状態と次の不足

現在の production training path は `ContinuousNavigationEnv` と
`run_continuous_navigation_training` を使う。この runtime は左手系 Y-up meter の x/z
座標、Y 回転、連続 action forward/turn、goal radius、static walls、
static obstacles、回転付き box collision、任意の距離センサ range を表現する。
Replay Bundle は continuous runtime の実座標と実 action から生成する。

現行の `DistanceSensor` は前方1本の range measurement であり、LiDAR のような angular
scan ではない。runtime はその1本を進行方向に 0.005 m 間隔でサンプルして最初の衝突までの
距離を求める。この値は角度分解能を表さない。Ray と交差する厚さ 0.005 m 未満の障害物が
サンプル点の間に完全に収まる場合、現行実装はその衝突を見逃し得る。

Scenario Bundle、Result Bundle、Replay Bundle の契約と、EnvForge からのジョブ投入、
進捗監視、artifact download、Replay 再生、ONNX Runtime 推論の主導線は実装済みである。
`EmbodiedLab.Unity` は contract DTO、server-owned job lifecycle、artifact 検証、
Replay 読み込みを所有する。EnvForge は同じ SDK revision へ移行済みであり、
移行前の直接 client、重複 DTO、重複 Replay／artifact 実装は残していない。

次に不足しているものは以下である。

- `EmbodiedLab.Unity` の contract snapshot と generated DTO は現在の v0 schema に
  同期済みである。今後の schema 更新を package release へ反映する運用は未確定である。
- package version と API contract version の compatibility 方針が未確定である。
- 現在の Scenario Bundle は固定マップを表し、episode ごとの宣言的な環境生成規則を
  表現できない。
- ONNX export と Result Bundle metadata は continuous 主経路に接続済みであり、
  SDK は実ファイルの tensor metadata も検証する。
- reward weight と発火条件は Scenario Bundle から continuous runtime へ反映する。
  `goal_progress`、wide/rear angle、inactive の判定値も JSON を正本とする。
- PPO hyperparameter、environment 数、CPU、PyTorch thread、Replay 間隔、start pose
  randomization は Scenario Bundle から受け取る。実行時に解決した library version、
  resource 数、全 PPO 値は Result Bundle の training configuration に保存する。
- robot と sensor descriptor が最小限である。
- forward camera observation は semantic 2.5D projection であり、
  Unity の material、lighting、shadow、post-processing を再現するものではない。
- artifact access が public-read 前提である。

固定マップは今後も既定動作として維持する。episode ごとの環境生成は明示的な
`generated` mode として追加し、ユーザ提供の任意コードではなく、検証可能な宣言的
schema と seed に基づいて EmbodiedLab runtime が実行する。
