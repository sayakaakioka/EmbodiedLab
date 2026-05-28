# EnvForge 連携 Phase 3 fixture 検証

## 目的

Phase 3 では、EnvForge から EmbodiedLab へ Scenario Bundle を送る導線を
作り始める。

最初の目標は、クラウドジョブ起動ではなく、EnvForge が生成する
Scenario Bundle JSON を EmbodiedLab の Pydantic model と `/submissions`
endpoint で検証できることを確認することである。

## 今回の範囲

EnvForge 側の canonical fixture と同じ JSON を EmbodiedLab 側にも置く。

    tests/fixtures/envforge/navigation_default_scenario_bundle.json

この fixture は、EnvForge の固定ナビゲーションシナリオを表す。
`ScenarioBundle.model_validate` で読み込めることをテストし、以下の前提を確認する。

- `schema_version` は `scenario-bundle.v0` である。
- 座標系は `envforge_xz_meters` である。
- robot action layout は `["forward", "turn"]` である。
- sensor は `front_camera` と `front_distance` である。
- training の `max_episode_steps` は EnvForge 側の固定値と一致する。

Phase 3B では、同じ fixture を FastAPI test client から `/submissions` に POST
する smoke test も追加する。これにより、Pydantic model 単体だけでなく API
境界でも EnvForge の JSON を受け取れることを確認する。

## この順序を採用する理由

EnvForge 側の Unity UI、HTTP client、認証、クラウドジョブ起動を実装する前に、
データ契約そのものの不一致を検出できるようにする。

当面の source of truth は EmbodiedLab 側の Pydantic model である。
そのため、EnvForge が出す JSON を EmbodiedLab のテストで検証する経路を先に
作ることが、以後の送信導線を安全に進める前提になる。

## 保留事項

- EnvForge fixture と EmbodiedLab fixture の同期方法。
- EnvForge 側で fixture を Unity batchmode から自動生成する方法。
- 実 backend に対する `/submissions` smoke test。
