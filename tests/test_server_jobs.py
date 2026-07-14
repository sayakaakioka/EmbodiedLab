import pytest
from google.cloud import run_v2

from server.config import ServerConfig
from server.services.execution_reconciliation import read_execution_outcome
from server.services.jobs import request_training_cancellation, run_training_job


class FakeOperation:
    def __init__(self, *, metadata=None):
        self.metadata = metadata


class FakeJobsClient:
    def __init__(self, operation):
        self.operation = operation
        self.requests = []

    def run_job(self, *, request):
        self.requests.append(request)
        return self.operation


class FakeExecutionsClient:
    def __init__(self, operation):
        self.operation = operation
        self.requests = []

    def cancel_execution(self, *, request):
        self.requests.append(request)
        return self.operation


class FakeExecutionReader:
    def __init__(self, execution):
        self.execution = execution
        self.requests = []

    def get_execution(self, *, request):
        self.requests.append(request)
        return self.execution


def build_config():
    return ServerConfig(
        db_id="test-db",
        region="asia-northeast1",
        job_path="projects/test/locations/asia-northeast1/jobs/test-trainer",
        project_id="test-project",
        pubsub_topic="test-topic",
    )


def test_run_training_job_returns_execution_name_from_operation_metadata():
    execution_name = (
        "projects/test/locations/asia-northeast1/jobs/test-trainer/"
        "executions/test-trainer-abcde"
    )
    operation = FakeOperation(metadata=run_v2.Execution(name=execution_name))
    client = FakeJobsClient(operation)

    actual_name = run_training_job(
        build_config(),
        "submission-1",
        create_jobs_client=lambda **_kwargs: client,
    )

    assert actual_name == execution_name
    request = client.requests[0]
    assert request.name.endswith("/jobs/test-trainer")
    assert request.overrides.container_overrides[0].env[0].name == "SUBMISSION_ID"
    assert request.overrides.container_overrides[0].env[0].value == "submission-1"


def test_run_training_job_rejects_missing_execution_metadata():
    client = FakeJobsClient(FakeOperation(metadata=None))

    with pytest.raises(
        RuntimeError,
        match="Cloud Run did not return an execution name",
    ):
        run_training_job(
            build_config(),
            "submission-1",
            create_jobs_client=lambda **_kwargs: client,
        )


def test_request_training_cancellation_targets_exact_execution_name():
    operation = FakeOperation()
    client = FakeExecutionsClient(operation)
    execution_name = (
        "projects/test/locations/asia-northeast1/jobs/test-trainer/"
        "executions/test-trainer-abcde"
    )

    actual_operation = request_training_cancellation(
        build_config(),
        execution_name,
        create_executions_client=lambda **_kwargs: client,
    )

    assert actual_operation is operation
    assert client.requests == [run_v2.CancelExecutionRequest(name=execution_name)]


def test_read_execution_outcome_treats_cancelled_execution_as_cancelled():
    execution = run_v2.Execution(
        name=(
            "projects/test/locations/asia-northeast1/jobs/test-trainer/"
            "executions/test-trainer-abcde"
        ),
        cancelled_count=1,
        failed_count=1,
    )
    client = FakeExecutionReader(execution)

    outcome = read_execution_outcome(
        build_config(),
        execution.name,
        create_executions_client=lambda **_kwargs: client,
    )

    assert outcome.status.value == "cancelled"
    assert client.requests == [run_v2.GetExecutionRequest(name=execution.name)]
