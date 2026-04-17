from __future__ import annotations

from embodiedlab.result_models import Progress, ResultStatus, build_result_update


def update_result(
	result_ref,
	*,
	status: ResultStatus,
	progress: dict | Progress,
	summary=None,
	error=None,
	artifacts=None,
) -> None:
	payload = build_result_update(
		status=status,
		progress=progress,
		summary=summary,
		error=error,
		artifacts=artifacts,
	)
	result_ref.set(payload, merge=True)
