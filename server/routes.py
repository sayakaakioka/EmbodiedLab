"""FastAPI routes for submission creation, training, and result lookup."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from embodiedlab.repositories import (
    ResultFailureWriter,
    ResultQueueWriter,
    ResultReader,
    SubmissionExistenceChecker,
    SubmissionWriter,
)
from embodiedlab.schemas import SubmitRequest
from server.config import ServerConfig
from server.dependencies import (
    get_config,
    get_result_repository,
    get_submission_repository,
)
from server.services.jobs import run_training_job
from server.services.training_requests import (
    SubmissionNotFoundError,
    TrainingStartError,
    start_training_for_submission,
)

router = APIRouter()


@router.post("/submissions")
def create_submission(
    req: SubmitRequest,
    submission_repository: Annotated[
        SubmissionWriter,
        Depends(get_submission_repository),
    ],
) -> dict[str, str]:
    """Create a new submission and persist it to Firestore."""
    submission_id = submission_repository.save(req)

    return {
        "status": "accepted",
        "submission_id": submission_id,
    }


@router.post("/submissions/{submission_id}/train")
def train(
    submission_id: str,
    server_config: Annotated[ServerConfig, Depends(get_config)],
    submission_repository: Annotated[
        SubmissionExistenceChecker,
        Depends(get_submission_repository),
    ],
    result_repository: Annotated[
        ResultQueueWriter | ResultFailureWriter,
        Depends(get_result_repository),
    ],
) -> dict[str, str]:
    """Queue a result document and trigger the trainer job."""
    try:
        start_training_for_submission(
            submission_repository=submission_repository,
            result_repository=result_repository,
            config=server_config,
            submission_id=submission_id,
            trigger_job=run_training_job,
        )
    except SubmissionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Submission not found") from exc
    except TrainingStartError as exc:
        raise HTTPException(
            status_code=500,
            detail=exc.message,
        ) from exc

    return {
        "status": "accepted",
        "submission_id": submission_id,
    }


@router.get("/results/{submission_id}")
def get_result(
    submission_id: str,
    result_repository: Annotated[
        ResultReader,
        Depends(get_result_repository),
    ],
) -> dict[str, Any]:
    """Return the latest result document for the submission."""
    result = result_repository.fetch(submission_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found")

    return result
