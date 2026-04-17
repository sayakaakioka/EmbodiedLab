from __future__ import annotations

from typing import Any

from embodiedlab.schemas import SubmitRequest
from embodiedlab.training.training_models import GridWorldSpec


def convert_submission_to_spec(submission: dict[str, Any] | SubmitRequest) -> GridWorldSpec:
	req = (
		submission
		if isinstance(submission, SubmitRequest)
		else SubmitRequest.model_validate(submission)
	)
	environment = req.environment
	robot = req.robot

	width, height = environment.size

	return GridWorldSpec(
		width=width,
		height=height,
		obstacles=frozenset(environment.obstacles),
		goal=environment.goal,
		robot_start=environment.robot_start,
		robot_type=robot.type,
	)
