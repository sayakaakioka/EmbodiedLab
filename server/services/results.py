from __future__ import annotations

from embodiedlab.result_models import (
	ResultStatus,
	build_queued_result_document,
	build_result_update,
)


def create_queued_result(db, submission_id: str) -> None:
	db.collection("results").document(submission_id).set(
		build_queued_result_document(submission_id)
	)


def mark_result_failed(db, submission_id: str, message: str) -> None:
	payload = build_result_update(
		status=ResultStatus.FAILED,
		progress={
			"phase": ResultStatus.FAILED,
			"current_step": 0,
			"total_steps": 0,
			"message": message,
		},
		error=message,
	)
	db.collection("results").document(submission_id).set(payload, merge=True)


def fetch_result(db, submission_id: str) -> dict | None:
	result_snap = db.collection("results").document(submission_id).get()
	if not result_snap.exists:
		return None

	return result_snap.to_dict()
