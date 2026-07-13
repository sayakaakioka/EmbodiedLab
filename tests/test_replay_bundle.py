import json
from pathlib import Path

from embodiedlab.result_models import ReplayBundleManifest
from embodiedlab.training.replay_bundle import ReplayBundleWriter

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_replay_bundle_writer_matches_canonical_manifest(tmp_path):
    writer = ReplayBundleWriter(
        root_dir=tmp_path / "replay",
        job_id="submission-1",
        scenario_id="navigation_default",
        total_timesteps=5000,
        train_chunk_steps=2,
    )
    writer.record_train_step({"checkpoint_step": 1})
    writer.record_train_step({"checkpoint_step": 2})
    writer.write_eval_checkpoint(
        checkpoint_step=5000,
        steps=[
            {"episode_id": "eval_env_00_episode_000001", "step_index": 0},
            {"episode_id": "eval_env_00_episode_000001", "step_index": 1},
        ],
        success_rate=1.0,
        avg_reward=82.4,
        avg_steps=118.5,
    )

    manifest = writer.finish()
    fixture_path = FIXTURE_DIR / "envforge" / "navigation_replay_bundle_manifest.json"
    expected = json.loads(fixture_path.read_text(encoding="utf-8"))
    written = json.loads(
        (writer.root_dir / "manifest.json").read_text(encoding="utf-8"),
    )
    validated = ReplayBundleManifest.model_validate(expected)

    assert manifest == expected
    assert written == expected
    assert validated.model_dump(mode="json", exclude_none=True) == expected
