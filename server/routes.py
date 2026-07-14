"""FastAPI routes for submission creation, training, and result lookup."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from embodiedlab.api_models import SubmissionResponse, TrainingResponse
from embodiedlab.repositories import (
    ResultQueueWriter,
    ResultReader,
    ResultUpdateWriter,
    SubmissionControlReader,
    SubmissionExecutionWriter,
    SubmissionExistenceChecker,
    SubmissionWriter,
)
from embodiedlab.result_models import ResultDocument
from embodiedlab.schemas import ScenarioBundle
from server.config import ServerConfig
from server.dependencies import (
    get_cancellation_requester,
    get_config,
    get_execution_outcome_reader,
    get_result_event_publisher,
    get_result_repository,
    get_submission_repository,
)
from server.services.cancellation_tokens import (
    hash_cancel_token,
    issue_cancel_token,
)
from server.services.cancellations import (
    CancellationNotAllowedError,
    CancellationRequester,
    CancellationRequestError,
    CancellationResultNotFoundError,
    InvalidCancelTokenError,
    ResultEventPublisher,
    cancel_training_for_submission,
)
from server.services.execution_reconciliation import (
    ExecutionOutcomeReader,
    reconcile_result_with_execution,
)
from server.services.jobs import run_training_job
from server.services.training_requests import (
    SubmissionNotFoundError,
    TrainingStartError,
    start_training_for_submission,
)

router = APIRouter()
cancel_token_scheme = HTTPBearer(auto_error=False)


@router.post("/submissions")
def create_submission(
    scenario: ScenarioBundle,
    submission_repository: Annotated[
        SubmissionWriter,
        Depends(get_submission_repository),
    ],
) -> SubmissionResponse:
    """Create a new submission and persist it to Firestore."""
    cancel_token = issue_cancel_token()
    submission_id = submission_repository.save(
        scenario,
        cancel_token_hash=hash_cancel_token(cancel_token),
    )

    return SubmissionResponse(
        status="accepted",
        submission_id=submission_id,
        cancel_token=cancel_token,
    )


@router.post("/submissions/{submission_id}/train")
def train(
    submission_id: str,
    server_config: Annotated[ServerConfig, Depends(get_config)],
    submission_repository: Annotated[
        SubmissionExistenceChecker | SubmissionExecutionWriter,
        Depends(get_submission_repository),
    ],
    result_repository: Annotated[
        ResultQueueWriter | ResultUpdateWriter,
        Depends(get_result_repository),
    ],
) -> TrainingResponse:
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

    return TrainingResponse(status="accepted", submission_id=submission_id)


@router.post(
    "/submissions/{submission_id}/cancel",
    response_model=ResultDocument,
    response_model_exclude_unset=True,
)
def cancel_training(  # noqa: PLR0913
    submission_id: str,
    response: Response,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(cancel_token_scheme),
    ],
    server_config: Annotated[ServerConfig, Depends(get_config)],
    submission_repository: Annotated[
        SubmissionControlReader,
        Depends(get_submission_repository),
    ],
    result_repository: Annotated[
        ResultReader | ResultUpdateWriter,
        Depends(get_result_repository),
    ],
    request_cancellation: Annotated[
        CancellationRequester,
        Depends(get_cancellation_requester),
    ],
    publish_event: Annotated[
        ResultEventPublisher,
        Depends(get_result_event_publisher),
    ],
) -> ResultDocument:
    """Cancel the exact Cloud Run execution controlled by the bearer token."""
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=403, detail="Invalid cancellation token")

    try:
        cancellation = cancel_training_for_submission(
            config=server_config,
            submission_id=submission_id,
            cancel_token=credentials.credentials,
            submission_repository=submission_repository,
            result_repository=result_repository,
            request_cancellation=request_cancellation,
            publish_event=publish_event,
        )
    except InvalidCancelTokenError as exc:
        raise HTTPException(
            status_code=403,
            detail="Invalid cancellation token",
        ) from exc
    except CancellationResultNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Result not found") from exc
    except CancellationNotAllowedError as exc:
        raise HTTPException(
            status_code=409,
            detail="Job cannot be cancelled in its current state",
        ) from exc
    except CancellationRequestError as exc:
        raise HTTPException(
            status_code=502,
            detail="Failed to cancel trainer job",
        ) from exc

    if cancellation.pending:
        response.status_code = 202
    return ResultDocument.model_validate(cancellation.result)


@router.get(
    "/results/{submission_id}",
    response_model=ResultDocument,
    response_model_exclude_unset=True,
)
def get_result(  # noqa: PLR0913
    submission_id: str,
    server_config: Annotated[ServerConfig, Depends(get_config)],
    result_repository: Annotated[
        ResultReader | ResultUpdateWriter,
        Depends(get_result_repository),
    ],
    submission_repository: Annotated[
        SubmissionControlReader,
        Depends(get_submission_repository),
    ],
    read_execution: Annotated[
        ExecutionOutcomeReader,
        Depends(get_execution_outcome_reader),
    ],
    publish_event: Annotated[
        ResultEventPublisher,
        Depends(get_result_event_publisher),
    ],
) -> ResultDocument:
    """Return the latest result document for the submission."""
    result = result_repository.fetch(submission_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found")

    return ResultDocument.model_validate(
        reconcile_result_with_execution(
            config=server_config,
            submission_id=submission_id,
            submission_repository=submission_repository,
            result_repository=result_repository,
            result=result,
            read_outcome=read_execution,
            publish_event=publish_event,
        ),
    )
