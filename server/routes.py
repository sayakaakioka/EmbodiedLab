from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from embodiedlab.schemas import SubmitRequest
from server.config import ServerConfig
from server.dependencies import get_config, get_db
from server.services.jobs import run_training_job
from server.services.results import create_queued_result, fetch_result, mark_result_failed
from server.services.submissions import save_submission, submission_exists

router = APIRouter()


@router.post("/submissions")
def create_submission(req: SubmitRequest, db=Depends(get_db)):
	submission_id = save_submission(db, req)

	return {
		"status": "accepted",
		"submission_id": submission_id,
	}


@router.post("/submissions/{submission_id}/train")
def train(
	submission_id: str,
	db=Depends(get_db),
	server_config: ServerConfig = Depends(get_config),
):
	if not submission_exists(db, submission_id):
		raise HTTPException(status_code=404, detail="Submission not found")

	create_queued_result(db, submission_id)
	try:
		run_training_job(server_config, submission_id)
	except Exception as exc:
		mark_result_failed(db, submission_id, "Failed to start trainer job")
		raise HTTPException(status_code=500, detail="Failed to start trainer job") from exc

	return {
		"status": "accepted",
		"submission_id": submission_id,
	}


@router.get("/results/{submission_id}")
def get_result(submission_id: str, db=Depends(get_db)):
	result = fetch_result(db, submission_id)
	if result is None:
		raise HTTPException(status_code=404, detail="Result not found")

	return result
