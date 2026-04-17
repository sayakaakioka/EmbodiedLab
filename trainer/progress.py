from __future__ import annotations

from embodiedlab.result_models import Progress, ResultStatus


def build_progress(
	*,
	phase: ResultStatus,
	current_step: int,
	total_steps: int,
	message: str,
) -> Progress:
	return Progress(
		phase=phase,
		current_step=current_step,
		total_steps=total_steps,
		message=message,
	)


def failed_progress(message: str, total_steps: int = 0) -> Progress:
	return build_progress(
		phase=ResultStatus.FAILED,
		current_step=0,
		total_steps=total_steps,
		message=message,
	)


def starting_progress(total_steps: int) -> Progress:
	return build_progress(
		phase=ResultStatus.STARTING,
		current_step=0,
		total_steps=total_steps,
		message="Trainer job started",
	)


def running_progress(total_steps: int) -> Progress:
	return build_progress(
		phase=ResultStatus.RUNNING,
		current_step=0,
		total_steps=total_steps,
		message="Training",
	)


def completed_progress(total_steps: int) -> Progress:
	return build_progress(
		phase=ResultStatus.COMPLETED,
		current_step=total_steps,
		total_steps=total_steps,
		message="Training completed",
	)
