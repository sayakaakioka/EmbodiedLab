"""Replay Bundle writing helpers for client timeline playback."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

from embodiedlab.artifact_integrity import compute_file_integrity
from embodiedlab.result_models import (
    REPLAY_BUNDLE_SCHEMA_VERSION,
    REPLAY_LOG_SCHEMA_VERSION,
    ReplayBundleManifest,
    ReplayLogStep,
)
from embodiedlab.schemas import MAX_REPLAY_CHUNK_STEPS


class ReplayBundleWriter:
    """Write chunked train/eval replay logs and a manifest."""

    def __init__(
        self,
        *,
        root_dir: str | Path,
        job_id: str,
        scenario_id: str,
        total_timesteps: int,
        train_chunk_steps: int,
    ) -> None:
        """Create a writer rooted at the future Replay Bundle directory."""
        self.root_dir = Path(root_dir)
        self.job_id = job_id
        self.scenario_id = scenario_id
        self.total_timesteps = total_timesteps
        if not 1 <= train_chunk_steps <= MAX_REPLAY_CHUNK_STEPS:
            msg = f"train_chunk_steps must be between 1 and {MAX_REPLAY_CHUNK_STEPS}"
            raise ValueError(msg)
        self.train_chunk_steps = train_chunk_steps
        self.root_dir.mkdir(parents=True, exist_ok=True)
        (self.root_dir / "train").mkdir(parents=True, exist_ok=True)
        (self.root_dir / "eval").mkdir(parents=True, exist_ok=True)
        self._chunks: list[dict[str, Any]] = []
        self._train_file: gzip.GzipFile | None = None
        self._train_chunk_index = 0
        self._train_chunk_count = 0
        self._train_chunk_first_step: int | None = None
        self._train_chunk_last_step: int | None = None
        self._train_chunk_path: str | None = None

    @property
    def chunks(self) -> list[dict[str, Any]]:
        """Return chunk metadata accumulated so far."""
        return list(self._chunks)

    def record_train_step(self, step: dict[str, Any]) -> None:
        """Append one stochastic training transition to the active chunk."""
        self._ensure_train_chunk(int(step.get("checkpoint_step", 0)))
        if self._train_file is None:
            msg = "Training replay chunk was not opened."
            raise RuntimeError(msg)
        self._train_file.write((self._serialize_step(step) + "\n").encode("utf-8"))
        self._train_chunk_count += 1
        self._train_chunk_last_step = int(step.get("checkpoint_step", 0))
        if self._train_chunk_count >= self.train_chunk_steps:
            self._close_train_chunk()

    def write_eval_checkpoint(
        self,
        *,
        checkpoint_step: int,
        steps: list[dict[str, Any]],
        success_rate: float,
        avg_reward: float,
        avg_steps: float,
    ) -> None:
        """Write one deterministic evaluation checkpoint as its own chunk."""
        if len(steps) > MAX_REPLAY_CHUNK_STEPS:
            msg = f"eval replay chunk cannot exceed {MAX_REPLAY_CHUNK_STEPS} steps"
            raise ValueError(msg)
        relative_path = f"eval/checkpoint_{checkpoint_step:08d}.jsonl.gz"
        path = self.root_dir / relative_path
        with gzip.GzipFile(filename=path, mode="wb", mtime=0) as replay_file:
            for step in steps:
                replay_file.write((self._serialize_step(step) + "\n").encode("utf-8"))
        integrity = compute_file_integrity(path)
        self._chunks.append(
            {
                "phase": "eval",
                "policy_mode": "deterministic",
                "checkpoint_step": checkpoint_step,
                "start_step": None,
                "end_step": None,
                "path": relative_path,
                "format": "jsonl.gz",
                "step_count": len(steps),
                "episode_count": _count_episodes(steps),
                "success_rate": success_rate,
                "avg_reward": avg_reward,
                "avg_steps": avg_steps,
                **integrity.as_dict(),
            },
        )

    def finish(self) -> dict[str, Any]:
        """Close pending chunks and write the Replay Bundle manifest."""
        self._close_train_chunk()
        manifest = ReplayBundleManifest(
            schema_version=REPLAY_BUNDLE_SCHEMA_VERSION,
            job_id=self.job_id,
            scenario_id=self.scenario_id,
            total_timesteps=self.total_timesteps,
            chunks=self._chunks,
        ).model_dump(mode="json")
        (self.root_dir / "manifest.json").write_text(
            ReplayBundleManifest.model_validate(manifest).model_dump_json(
                indent=2,
                exclude_none=False,
            ),
            encoding="utf-8",
        )
        return manifest

    def _serialize_step(self, step: dict[str, Any]) -> str:
        payload = {
            **step,
            "schema_version": REPLAY_LOG_SCHEMA_VERSION,
            "scenario_id": self.scenario_id,
            "job_id": self.job_id,
        }
        return ReplayLogStep.model_validate(payload).model_dump_json()

    def _ensure_train_chunk(self, checkpoint_step: int) -> None:
        if self._train_file is not None:
            return
        relative_path = f"train/chunk_{self._train_chunk_index:06d}.jsonl.gz"
        self._train_chunk_index += 1
        self._train_chunk_path = relative_path
        self._train_chunk_first_step = checkpoint_step
        self._train_chunk_last_step = checkpoint_step
        self._train_chunk_count = 0
        self._train_file = gzip.GzipFile(
            filename=self.root_dir / relative_path,
            mode="wb",
            mtime=0,
        )

    def _close_train_chunk(self) -> None:
        if self._train_file is None:
            return
        self._train_file.close()
        if self._train_chunk_count > 0:
            if self._train_chunk_path is None:
                msg = "Training replay chunk path was not recorded."
                raise RuntimeError(msg)
            integrity = compute_file_integrity(
                self.root_dir / self._train_chunk_path,
            )
            self._chunks.append(
                {
                    "phase": "train",
                    "policy_mode": "stochastic",
                    "checkpoint_step": self._train_chunk_last_step,
                    "start_step": self._train_chunk_first_step,
                    "end_step": self._train_chunk_last_step,
                    "path": self._train_chunk_path,
                    "format": "jsonl.gz",
                    "step_count": self._train_chunk_count,
                    "episode_count": None,
                    "success_rate": None,
                    "avg_reward": None,
                    "avg_steps": None,
                    **integrity.as_dict(),
                },
            )
        self._train_file = None
        self._train_chunk_path = None
        self._train_chunk_first_step = None
        self._train_chunk_last_step = None
        self._train_chunk_count = 0


def _count_episodes(steps: list[dict[str, Any]]) -> int:
    return len(
        {str(step.get("episode_id", "")) for step in steps if step.get("episode_id")},
    )
