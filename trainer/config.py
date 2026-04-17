from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainerConfig:
	db_id: str
	model_bucket: str
	submission_id: str


def _get_required_env(name: str) -> str:
	value = os.environ.get(name)
	if not value:
		raise RuntimeError(f"{name} is not set")

	return value


def load_trainer_config() -> TrainerConfig:
	return TrainerConfig(
		db_id=os.environ.get("DB_ID", "(default)"),
		model_bucket=_get_required_env("MODEL_BUCKET"),
		submission_id=_get_required_env("SUBMISSION_ID"),
	)
