from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class TrainingAlgorithm(StrEnum):
	PPO = "ppo"


class TrainingConfig(BaseModel):
	algorithm: TrainingAlgorithm = Field(default=TrainingAlgorithm.PPO)
	timesteps: int = Field(default=5_000, ge=1)
	seed: int = Field(default=10)
	max_steps: int = Field(default=50, ge=1)
	n_steps: int = Field(default=32, ge=1)
	batch_size: int = Field(default=32, ge=1)
	gamma: float = Field(default=0.99, gt=0.0, le=1.0)
	learning_rate: float = Field(default=3e-4, gt=0.0)
	ent_coef: float = Field(default=0.0, ge=0.0)
	eval_episodes: int = Field(default=20, ge=1)
