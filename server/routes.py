"""FastAPI routes for submission creation, cancellation, and result lookup."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from embodiedlab.api_models import SubmissionResponse
from embodiedlab.repositories import (
    ApiResultStore,
    SubmissionConflictError,
    SubmissionControlStore,
    SubmissionRecoveryError,
)
from embodiedlab.result_models import ResultDocument, ResultStatus
from embodiedlab.schemas import ScenarioBundle
from server.config import ServerConfig
from server.dependencies import (
    get_cancellation_requester,
    get_config,
    get_execution_outcome_reader,
    get_result_event_publisher,
    get_result_repository,
    get_submission_repository,
    get_submission_workflow,
)
from server.services.cancellation_tokens import (
    hash_cancel_token,
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
from server.services.submission_workflow import (
    SubmissionDispatchUnavailableError,
    SubmissionWorkflow,
)

router = APIRouter()
cancel_token_scheme = HTTPBearer(auto_error=False)
RECOVERY_HEADER_PATTERN = r"^[A-Za-z0-9_-]{32,128}$"


@router.post("/submissions")
def create_submission(
    scenario: ScenarioBundle,
    workflow: Annotated[
        SubmissionWorkflow,
        Depends(get_submission_workflow),
    ],
    idempotency_key: Annotated[
        str,
        Header(alias="Idempotency-Key", pattern=RECOVERY_HEADER_PATTERN),
    ],
    client_cancel_token: Annotated[
        str,
        Header(
            alias="X-EmbodiedLab-Cancel-Token",
            pattern=RECOVERY_HEADER_PATTERN,
        ),
    ],
) -> SubmissionResponse:
    """Persist and dispatch a training submission as one accepted operation."""
    try:
        submission_id = workflow.submit(
            scenario,
            cancel_token_hash=hash_cancel_token(client_cancel_token),
            idempotency_key=idempotency_key,
        )
    except SubmissionConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                "Idempotency key was already used with a different submission request"
            ),
        ) from exc
    except SubmissionRecoveryError as exc:
        raise HTTPException(
            status_code=409,
            detail="Existing submission cannot be recovered with the current contract",
        ) from exc
    except SubmissionDispatchUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail="Submission was saved but dispatch is temporarily unavailable",
        ) from exc

    return SubmissionResponse(
        status="accepted",
        submission_id=submission_id,
        cancel_token=client_cancel_token,
    )


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
        SubmissionControlStore,
        Depends(get_submission_repository),
    ],
    result_repository: Annotated[
        ApiResultStore,
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
        ApiResultStore,
        Depends(get_result_repository),
    ],
    submission_repository: Annotated[
        SubmissionControlStore,
        Depends(get_submission_repository),
    ],
    workflow: Annotated[
        SubmissionWorkflow,
        Depends(get_submission_workflow),
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

    progress = result.get("progress")
    total_steps = progress.get("total_steps", 0) if isinstance(progress, dict) else 0
    if not isinstance(total_steps, int) or total_steps < 0:
        total_steps = 0
    if result.get("status") == ResultStatus.QUEUED.value:
        try:
            workflow.reconcile(
                submission_id,
                total_steps=total_steps,
            )
        except SubmissionDispatchUnavailableError as exc:
            raise HTTPException(
                status_code=503,
                detail="Submission dispatch is temporarily unavailable",
            ) from exc
        result = result_repository.fetch(submission_id) or result

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
