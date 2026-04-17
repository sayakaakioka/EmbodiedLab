from embodiedlab.result_models import (
	ResultStatus,
	build_queued_result_document,
	build_result_update,
)


def test_build_queued_result_document_returns_firestore_payload():
	payload = build_queued_result_document("submission-1")

	assert payload["submission_id"] == "submission-1"
	assert payload["status"] == "queued"
	assert payload["progress"] == {
		"phase": "queued",
		"current_step": 0,
		"total_steps": 0,
		"message": "Queued",
	}
	assert payload["summary"] is None
	assert payload["error"] is None
	assert payload["artifacts"] is None
	assert isinstance(payload["updated_at"], str)


def test_build_result_update_serializes_status_enum():
	payload = build_result_update(
		status=ResultStatus.RUNNING,
		progress={
			"phase": ResultStatus.RUNNING,
			"current_step": 0,
			"total_steps": 10,
			"message": "Training",
		},
	)

	assert payload["status"] == "running"
	assert payload["progress"]["phase"] == "running"
	assert payload["progress"]["total_steps"] == 10
