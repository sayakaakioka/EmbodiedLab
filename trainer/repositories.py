from __future__ import annotations

from google.cloud import firestore


def create_firestore_client(db_id: str):
	return firestore.Client(database=db_id)


def get_submission_ref(db, submission_id: str):
	return db.collection("submissions").document(submission_id)


def get_result_ref(db, submission_id: str):
	return db.collection("results").document(submission_id)


def fetch_submission(db, submission_id: str) -> dict | None:
	submission_snap = get_submission_ref(db, submission_id).get()
	if not submission_snap.exists:
		return None

	return submission_snap.to_dict()
