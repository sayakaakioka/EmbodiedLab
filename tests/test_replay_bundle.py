import gzip
import json
from pathlib import Path

from embodiedlab.result_models import ReplayBundleManifest, ReplayLogStep
from embodiedlab.training.replay_bundle import ReplayBundleWriter

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _replay_step(
    *,
    phase="train",
    checkpoint_step=1,
    episode_id="train_env_00_episode_000001",
    step_index=0,
):
    return {
        "phase": phase,
        "checkpoint_step": checkpoint_step,
        "env_index": 0,
        "policy_mode": "stochastic" if phase == "train" else "deterministic",
        "episode_id": episode_id,
        "step_index": step_index,
        "time_seconds": step_index * 0.1,
        "robot": {
            "position": {"x": 0.0, "z": 0.0},
            "rotation_y_degrees": 0.0,
        },
        "action": {"values": []},
        "reward": {"total": 0.0, "components": []},
        "events": [],
        "sensors": [],
        "terminated": False,
        "termination_reason": None,
    }


def test_replay_bundle_writer_matches_canonical_manifest(tmp_path):
    writer = ReplayBundleWriter(
        root_dir=tmp_path / "replay",
        job_id="submission-1",
        scenario_id="navigation_default",
        total_timesteps=5000,
        train_chunk_steps=2,
    )
    writer.record_train_step(_replay_step(checkpoint_step=1))
    writer.record_train_step(_replay_step(checkpoint_step=2, step_index=1))
    writer.write_eval_checkpoint(
        checkpoint_step=5000,
        steps=[
            _replay_step(
                phase="eval",
                checkpoint_step=5000,
                episode_id="eval_env_00_episode_000001",
            ),
            _replay_step(
                phase="eval",
                checkpoint_step=5000,
                episode_id="eval_env_00_episode_000001",
                step_index=1,
            ),
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

    for chunk in manifest["chunks"]:
        with gzip.open(writer.root_dir / chunk["path"], "rt", encoding="utf-8") as file:
            rows = [ReplayLogStep.model_validate_json(line) for line in file]
        assert len(rows) == chunk["step_count"]
        assert {row.scenario_id for row in rows} == {"navigation_default"}
        assert {row.job_id for row in rows} == {"submission-1"}
