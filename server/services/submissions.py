from __future__ import annotations

import uuid

from embodiedlab.schemas import SubmitRequest, build_submission_document


def save_submission(db, req: SubmitRequest) -> str:
	submission_id = str(uuid.uuid4())

	db.collection("submissions").document(submission_id).set(
		build_submission_document(submission_id, req)
	)

	return submission_id


def submission_exists(db, submission_id: str) -> bool:
	submission_snap = db.collection("submissions").document(submission_id).get()
	return submission_snap.exists
