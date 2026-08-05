import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from google.api_core.exceptions import RetryError

import trainer.job
from server.config import ServerConfig
from server.dependencies import (
    get_cancellation_requester,
    get_config,
    get_execution_outcome_reader,
    get_result_event_publisher,
    get_result_repository,
    get_submission_repository,
    get_training_job_runner,
)
from server.main import create_app
from server.services.execution_reconciliation import ExecutionOutcome
from server.services.jobs import (
    CancellationRequestRejectedError,
    TrainingDispatchRejectedError,
)
from tests.fakes import (
    FakeResultRepository,
    FakeSubmissionRepository,
    completed_artifacts,
    resolved_training_configuration,
    resolved_training_summary,
    result_document,
    scenario_payload,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"
EXECUTION_NAME = (
    "projects/test/locations/asia-northeast1/jobs/test-trainer/"
    "executions/test-trainer-abcde"
)
PENDING_MESSAGE = "cancellation still pending"
IDEMPOTENCY_KEY = "submission-recovery-key-0000000001"
CLIENT_CANCEL_TOKEN = "cancel-capability-0000000000000001"  # noqa: S105
IDEMPOTENCY_HEADERS = {
    "Idempotency-Key": IDEMPOTENCY_KEY,
    "X-EmbodiedLab-Cancel-Token": CLIENT_CANCEL_TOKEN,
}


class CompletedCancellationOperation:
    def __init__(self):
        self.timeouts = []

    def result(self, *, timeout):
        self.timeouts.append(timeout)
        return object()


class PendingCancellationOperation:
    def result(self, *, timeout):
        raise RetryError(PENDING_MESSAGE, TimeoutError())


class AmbiguousCancellationOperation:
    def result(self, *, timeout):
        raise RuntimeError


def build_test_app(  # noqa: PLR0913
    submission_repository: FakeSubmissionRepository,
    result_repository: FakeResultRepository,
    *,
    read_execution_outcome=lambda _config, _execution_name: None,
    request_cancellation=lambda _config, _execution_name: None,
    publish_result_event=lambda **_kwargs: None,
    run_training=lambda _config, _submission_id: EXECUTION_NAME,
):
    submission_repository.bind_result_repository(result_repository)
    app = create_app()
    app.dependency_overrides[get_config] = lambda: ServerConfig(
        db_id="test-db",
        region="asia-northeast1",
        job_path="projects/test/locations/asia-northeast1/jobs/test-trainer",
        project_id="test-project",
        pubsub_topic="test-topic",
    )
    app.dependency_overrides[get_submission_repository] = lambda: submission_repository
    app.dependency_overrides[get_result_repository] = lambda: result_repository
    app.dependency_overrides[get_execution_outcome_reader] = lambda: (
        read_execution_outcome
    )
    app.dependency_overrides[get_cancellation_requester] = lambda: request_cancellation
    app.dependency_overrides[get_result_event_publisher] = lambda: publish_result_event
    app.dependency_overrides[get_training_job_runner] = lambda: run_training
    return app


def test_create_app_registers_routes():
    app = create_app()

    paths = {route.path for route in app.routes}

    assert "/submissions" in paths
    assert "/submissions/{submission_id}/train" not in paths
    assert "/submissions/{submission_id}/cancel" in paths
    assert "/results/{submission_id}" in paths


def test_create_submission_persists_default_payload():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    client = TestClient(build_test_app(submission_repository, result_repository))

    response = client.post(
        "/submissions", json=scenario_payload(), headers=IDEMPOTENCY_HEADERS
    )

    assert response.status_code == 200
    submission_id = response.json()["submission_id"]
    cancel_token = response.json()["cancel_token"]
    submission = submission_repository.fetch(submission_id)
    assert submission["submission_id"] == submission_id
    assert len(cancel_token) >= 32
    assert submission["control"]["cancel_token_hash"] != cancel_token
    assert len(submission["control"]["cancel_token_hash"]) == 64
    assert cancel_token not in json.dumps(submission)
    scenario = submission["scenario"]
    assert scenario["schema_version"] == "scenario-bundle.v0"
    assert scenario["world"]["goal"]["position"] == {"x": 8.5, "z": 8.5}
    assert scenario["robot"]["type"] == "simple_robot"
    assert scenario["robot"]["action_space"]["layout"] == ["forward", "turn"]
    assert scenario["training"]["algorithm"] == "ppo"


def test_create_submission_replays_same_response_for_same_recovery_headers():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    dispatches = []

    def run_training(config, submission_id):
        dispatches.append((config, submission_id))
        return EXECUTION_NAME

    client = TestClient(
        build_test_app(
            submission_repository,
            result_repository,
            run_training=run_training,
        ),
    )

    first = client.post(
        "/submissions", json=scenario_payload(), headers=IDEMPOTENCY_HEADERS
    )
    replay = client.post(
        "/submissions", json=scenario_payload(), headers=IDEMPOTENCY_HEADERS
    )

    assert first.status_code == 200
    assert replay.status_code == 200
    assert replay.json() == first.json()
    assert replay.json()["cancel_token"] == CLIENT_CANCEL_TOKEN
    assert len(submission_repository.submissions) == 1
    assert len(dispatches) == 1
    submission = submission_repository.fetch(replay.json()["submission_id"])
    assert CLIENT_CANCEL_TOKEN not in json.dumps(submission)


def test_create_submission_rejects_recovery_key_reuse_with_different_request():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    client = TestClient(build_test_app(submission_repository, result_repository))
    first = client.post(
        "/submissions", json=scenario_payload(), headers=IDEMPOTENCY_HEADERS
    )
    different_payload = scenario_payload()
    different_payload["scenario_id"] = "different"

    different_scenario = client.post(
        "/submissions",
        json=different_payload,
        headers=IDEMPOTENCY_HEADERS,
    )
    different_token = client.post(
        "/submissions",
        json=scenario_payload(),
        headers={
            **IDEMPOTENCY_HEADERS,
            "X-EmbodiedLab-Cancel-Token": "different-capability-000000000000001",
        },
    )

    assert first.status_code == 200
    assert different_scenario.status_code == 409
    assert different_token.status_code == 409
    assert len(submission_repository.submissions) == 1


def test_create_submission_rejects_unrecoverable_existing_submission():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    client = TestClient(build_test_app(submission_repository, result_repository))
    first = client.post(
        "/submissions", json=scenario_payload(), headers=IDEMPOTENCY_HEADERS
    )
    result_repository.results.pop(first.json()["submission_id"])

    replay = client.post(
        "/submissions", json=scenario_payload(), headers=IDEMPOTENCY_HEADERS
    )

    assert replay.status_code == 409
    assert replay.json()["detail"] == (
        "Existing submission cannot be recovered with the current contract"
    )


def test_create_submission_requires_both_recovery_headers():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    client = TestClient(build_test_app(submission_repository, result_repository))

    only_key = client.post(
        "/submissions",
        json=scenario_payload(),
        headers={"Idempotency-Key": IDEMPOTENCY_KEY},
    )
    only_token = client.post(
        "/submissions",
        json=scenario_payload(),
        headers={"X-EmbodiedLab-Cancel-Token": CLIENT_CANCEL_TOKEN},
    )

    assert only_key.status_code == 422
    assert only_token.status_code == 422


def test_create_submission_rejects_short_recovery_headers():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    client = TestClient(build_test_app(submission_repository, result_repository))

    response = client.post(
        "/submissions",
        json=scenario_payload(),
        headers={
            "Idempotency-Key": "too-short",
            "X-EmbodiedLab-Cancel-Token": "also-too-short",
        },
    )

    assert response.status_code == 422
    assert submission_repository.submissions == {}


def test_create_submission_accepts_envforge_navigation_fixture():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    client = TestClient(build_test_app(submission_repository, result_repository))
    fixture_path = FIXTURE_DIR / "envforge" / "navigation_default_scenario_bundle.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))

    response = client.post(
        "/submissions",
        json=payload,
        headers=IDEMPOTENCY_HEADERS,
    )

    assert response.status_code == 200
    submission_id = response.json()["submission_id"]
    submission = submission_repository.fetch(submission_id)
    scenario = submission["scenario"]
    assert scenario["scenario_id"] == "navigation_default"
    assert scenario["world"]["bounds"]["min"] == {"x": -8.0, "z": -6.0}
    assert scenario["robot"]["start_pose"]["position"] == {"x": -6.0, "z": -4.0}
    assert scenario["training"]["max_episode_steps"] == 1000


def test_create_submission_queues_result_and_runs_job():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    calls = []

    def run_job(config, submission_id):
        calls.append((config, submission_id))
        return EXECUTION_NAME

    client = TestClient(
        build_test_app(
            submission_repository,
            result_repository,
            run_training=run_job,
        ),
    )

    response = client.post(
        "/submissions", json=scenario_payload(), headers=IDEMPOTENCY_HEADERS
    )

    assert response.status_code == 200
    submission_id = response.json()["submission_id"]
    assert response.json()["status"] == "accepted"
    assert response.json()["cancel_token"]
    assert calls[0][1] == "submission-1"
    assert submission_id == "submission-1"
    assert submission_repository.fetch_control(submission_id).execution_name == (
        EXECUTION_NAME
    )
    result = result_repository.fetch(submission_id)
    assert result["status"] == "queued"
    assert result["progress"]["phase"] == "queued"
    assert result["progress"]["total_steps"] == 5000


def test_create_submission_returns_503_when_dispatch_claim_is_unavailable():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    submission_repository.claim_dispatch = lambda _submission_id: (_ for _ in ()).throw(
        RuntimeError,
    )
    client = TestClient(build_test_app(submission_repository, result_repository))

    response = client.post(
        "/submissions", json=scenario_payload(), headers=IDEMPOTENCY_HEADERS
    )

    assert response.status_code == 503
    assert len(submission_repository.submissions) == 1
    assert result_repository.fetch("submission-1")["status"] == "queued"


def test_create_submission_returns_job_when_dispatch_fails():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    published_events = []

    def raise_job_error(config, submission_id):
        raise TrainingDispatchRejectedError("boom")

    client = TestClient(
        build_test_app(
            submission_repository,
            result_repository,
            run_training=raise_job_error,
            publish_result_event=lambda **kwargs: published_events.append(kwargs),
        ),
    )

    response = client.post(
        "/submissions", json=scenario_payload(), headers=IDEMPOTENCY_HEADERS
    )

    assert response.status_code == 200
    submission_id = response.json()["submission_id"]
    assert response.json()["status"] == "accepted"
    result = result_repository.fetch(submission_id)
    assert result["status"] == "failed"
    assert result["progress"]["phase"] == "failed"
    assert result["progress"]["total_steps"] == 5000
    assert result["error"] == "Failed to start trainer job"
    assert [event["status"].value for event in published_events] == ["failed"]


def test_dispatch_failure_does_not_overwrite_completed_trainer_result():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    published_events = []

    def complete_then_reject(_config, submission_id):
        result_repository.results[submission_id] = result_document(
            submission_id,
            "completed",
        )
        raise TrainingDispatchRejectedError

    client = TestClient(
        build_test_app(
            submission_repository,
            result_repository,
            run_training=complete_then_reject,
            publish_result_event=lambda **kwargs: published_events.append(kwargs),
        ),
    )

    response = client.post(
        "/submissions", json=scenario_payload(), headers=IDEMPOTENCY_HEADERS
    )

    assert response.status_code == 200
    submission_id = response.json()["submission_id"]
    assert result_repository.fetch(submission_id)["status"] == "completed"
    assert published_events == []


def test_create_submission_does_not_redispatch_ambiguous_outcome():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    dispatches = []

    def lose_dispatch_response(config, submission_id):
        dispatches.append((config, submission_id))
        raise TimeoutError

    client = TestClient(
        build_test_app(
            submission_repository,
            result_repository,
            run_training=lose_dispatch_response,
        ),
    )

    first = client.post(
        "/submissions", json=scenario_payload(), headers=IDEMPOTENCY_HEADERS
    )
    replay = client.post(
        "/submissions", json=scenario_payload(), headers=IDEMPOTENCY_HEADERS
    )

    assert first.status_code == 200
    assert replay.json() == first.json()
    assert len(dispatches) == 1
    submission_id = first.json()["submission_id"]
    assert result_repository.fetch(submission_id)["status"] == "queued"
    control = submission_repository.fetch_control(submission_id)
    assert control.dispatch_state == "ambiguous"
    assert control.execution_name is None


def test_create_submission_retries_only_execution_metadata_write():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    original_mark_dispatched = submission_repository.mark_dispatched
    attempts = []

    def flaky_mark_dispatched(submission_id, execution_name):
        attempts.append((submission_id, execution_name))
        if len(attempts) < 3:
            raise RuntimeError
        return original_mark_dispatched(submission_id, execution_name)

    submission_repository.mark_dispatched = flaky_mark_dispatched
    client = TestClient(build_test_app(submission_repository, result_repository))

    response = client.post(
        "/submissions", json=scenario_payload(), headers=IDEMPOTENCY_HEADERS
    )

    assert response.status_code == 200
    assert len(attempts) == 3
    control = submission_repository.fetch_control(response.json()["submission_id"])
    assert control.dispatch_state == "dispatched"
    assert control.execution_name == EXECUTION_NAME


def test_cancel_running_job_persists_and_publishes_transitions():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    cancellation_calls = []
    published_events = []
    operation = CompletedCancellationOperation()

    def request_cancellation(config, execution_name):
        cancellation_calls.append((config, execution_name))
        return operation

    client = TestClient(
        build_test_app(
            submission_repository,
            result_repository,
            request_cancellation=request_cancellation,
            publish_result_event=lambda **kwargs: published_events.append(kwargs),
        ),
    )
    create_response = client.post(
        "/submissions",
        json=scenario_payload(),
        headers=IDEMPOTENCY_HEADERS,
    )
    submission_id = create_response.json()["submission_id"]
    cancel_token = create_response.json()["cancel_token"]
    submission_repository.mark_dispatched(submission_id, EXECUTION_NAME)
    result_repository.create_queued(submission_id)
    result_repository.write_update(
        submission_id,
        status="running",
        progress={
            "phase": "running",
            "current_step": 12,
            "total_steps": 100,
            "message": "Training",
        },
    )

    response = client.post(
        f"/submissions/{submission_id}/cancel",
        headers={"Authorization": f"Bearer {cancel_token}"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert response.json()["progress"] == {
        "phase": "cancelled",
        "current_step": 12,
        "total_steps": 100,
        "message": "Training cancelled",
    }
    assert cancellation_calls[0][1] == EXECUTION_NAME
    assert len(operation.timeouts) == 1
    assert [event["status"].value for event in published_events] == [
        "cancelling",
        "cancelled",
    ]


def test_cancel_records_cloud_acceptance_before_waiting_for_completion():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    observed_states = []

    class InspectingOperation:
        def result(self, *, timeout):
            observed_states.append(
                submission_repository.fetch_control(
                    create_response.json()["submission_id"],
                ).cancellation_state,
            )
            return object()

    client = TestClient(
        build_test_app(
            submission_repository,
            result_repository,
            request_cancellation=lambda _config, _execution_name: InspectingOperation(),
        ),
    )
    create_response = client.post(
        "/submissions",
        json=scenario_payload(),
        headers=IDEMPOTENCY_HEADERS,
    )

    response = client.post(
        f"/submissions/{create_response.json()['submission_id']}/cancel",
        headers={
            "Authorization": f"Bearer {create_response.json()['cancel_token']}",
        },
    )

    assert response.status_code == 200
    assert observed_states == ["requested"]


def test_cancel_does_not_roll_back_progress_advanced_while_waiting():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    client = TestClient(
        build_test_app(
            submission_repository,
            result_repository,
            request_cancellation=lambda _config, _execution_name: (
                CompletedCancellationOperation()
            ),
        ),
    )
    create_response = client.post(
        "/submissions",
        json=scenario_payload(),
        headers=IDEMPOTENCY_HEADERS,
    )
    submission_id = create_response.json()["submission_id"]
    result_repository.write_update(
        submission_id,
        status="running",
        progress={
            "phase": "running",
            "current_step": 12,
            "total_steps": 100,
            "message": "Training",
        },
    )
    original_transition = result_repository.transition_status_preserving_progress

    def advance_before_transition(*args, **kwargs):
        result_repository.results[submission_id]["progress"]["current_step"] = 60
        return original_transition(*args, **kwargs)

    result_repository.transition_status_preserving_progress = advance_before_transition

    response = client.post(
        f"/submissions/{submission_id}/cancel",
        headers={
            "Authorization": f"Bearer {create_response.json()['cancel_token']}",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert response.json()["progress"]["current_step"] == 60


def test_cancel_claim_loser_returns_terminal_result_without_pending_status():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    client = TestClient(build_test_app(submission_repository, result_repository))
    create_response = client.post(
        "/submissions",
        json=scenario_payload(),
        headers=IDEMPOTENCY_HEADERS,
    )
    submission_id = create_response.json()["submission_id"]

    def complete_before_claim(*_args, **_kwargs):
        result_repository.results[submission_id] = result_document(
            submission_id,
            "completed",
        )

    submission_repository.claim_cancellation = complete_before_claim

    response = client.post(
        f"/submissions/{submission_id}/cancel",
        headers={
            "Authorization": f"Bearer {create_response.json()['cancel_token']}",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "completed"


def test_cancel_pending_submission_prevents_cloud_dispatch():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    cancellation_calls = []
    submission_repository.claim_dispatch = lambda _submission_id: False
    client = TestClient(
        build_test_app(
            submission_repository,
            result_repository,
            request_cancellation=lambda *args: cancellation_calls.append(args),
        ),
    )
    create_response = client.post(
        "/submissions",
        json=scenario_payload(),
        headers=IDEMPOTENCY_HEADERS,
    )

    response = client.post(
        f"/submissions/{create_response.json()['submission_id']}/cancel",
        headers={
            "Authorization": f"Bearer {create_response.json()['cancel_token']}",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert cancellation_calls == []


def test_cancel_rejects_missing_or_invalid_capability_token():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    cancellation_calls = []
    client = TestClient(
        build_test_app(
            submission_repository,
            result_repository,
            request_cancellation=lambda *args: cancellation_calls.append(args),
        ),
    )
    create_response = client.post(
        "/submissions",
        json=scenario_payload(),
        headers=IDEMPOTENCY_HEADERS,
    )
    submission_id = create_response.json()["submission_id"]
    submission_repository.mark_dispatched(submission_id, EXECUTION_NAME)
    result_repository.create_queued(submission_id)

    missing_response = client.post(f"/submissions/{submission_id}/cancel")
    invalid_response = client.post(
        f"/submissions/{submission_id}/cancel",
        headers={"Authorization": "Bearer wrong-token"},
    )

    assert missing_response.status_code == 403
    assert invalid_response.status_code == 403
    assert cancellation_calls == []


def test_cancel_returns_accepted_while_cloud_run_cancellation_is_pending():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    cancellation_calls = []

    def request_cancellation(config, execution_name):
        cancellation_calls.append((config, execution_name))
        return PendingCancellationOperation()

    client = TestClient(
        build_test_app(
            submission_repository,
            result_repository,
            request_cancellation=request_cancellation,
        ),
    )
    create_response = client.post(
        "/submissions",
        json=scenario_payload(),
        headers=IDEMPOTENCY_HEADERS,
    )
    submission_id = create_response.json()["submission_id"]
    cancel_token = create_response.json()["cancel_token"]
    submission_repository.mark_dispatched(submission_id, EXECUTION_NAME)
    result_repository.create_queued(submission_id)

    response = client.post(
        f"/submissions/{submission_id}/cancel",
        headers={"Authorization": f"Bearer {cancel_token}"},
    )
    replay = client.post(
        f"/submissions/{submission_id}/cancel",
        headers={"Authorization": f"Bearer {cancel_token}"},
    )

    assert response.status_code == 202
    assert response.json()["status"] == "cancelling"
    assert replay.status_code == 202
    assert replay.json()["status"] == "cancelling"
    assert len(cancellation_calls) == 1


def test_cancel_reclaims_stale_intent_after_owner_stops_before_rpc():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    cancellation_calls = []

    def request_cancellation(config, execution_name):
        cancellation_calls.append((config, execution_name))
        return CompletedCancellationOperation()

    client = TestClient(
        build_test_app(
            submission_repository,
            result_repository,
            request_cancellation=request_cancellation,
        ),
    )
    create_response = client.post(
        "/submissions",
        json=scenario_payload(),
        headers=IDEMPOTENCY_HEADERS,
    )
    submission_id = create_response.json()["submission_id"]
    cancel_token = create_response.json()["cancel_token"]
    result_repository.write_update(
        submission_id,
        status="running",
        progress={
            "phase": "running",
            "current_step": 12,
            "total_steps": 100,
            "message": "Training",
        },
    )
    stopped_at = datetime.now(UTC) - timedelta(minutes=3)
    assert (
        submission_repository.claim_cancellation(
            submission_id,
            claimed_at=stopped_at,
            stale_before=stopped_at - timedelta(minutes=2),
        )
        is not None
    )

    response = client.post(
        f"/submissions/{submission_id}/cancel",
        headers={"Authorization": f"Bearer {cancel_token}"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert len(cancellation_calls) == 1


def test_cancel_retries_stale_ambiguous_request_without_restoring_result():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    cancellation_calls = []

    def request_cancellation(_config, _execution_name):
        cancellation_calls.append(object())
        if len(cancellation_calls) == 1:
            raise TimeoutError
        return CompletedCancellationOperation()

    client = TestClient(
        build_test_app(
            submission_repository,
            result_repository,
            request_cancellation=request_cancellation,
        ),
    )
    create_response = client.post(
        "/submissions",
        json=scenario_payload(),
        headers=IDEMPOTENCY_HEADERS,
    )
    submission_id = create_response.json()["submission_id"]
    cancel_token = create_response.json()["cancel_token"]
    result_repository.write_update(
        submission_id,
        status="running",
        progress={
            "phase": "running",
            "current_step": 12,
            "total_steps": 100,
            "message": "Training",
        },
    )

    first = client.post(
        f"/submissions/{submission_id}/cancel",
        headers={"Authorization": f"Bearer {cancel_token}"},
    )
    submission_repository.submissions[submission_id]["control"][
        "cancellation_started_at"
    ] = (datetime.now(UTC) - timedelta(minutes=3)).isoformat()
    recovered = client.post(
        f"/submissions/{submission_id}/cancel",
        headers={"Authorization": f"Bearer {cancel_token}"},
    )

    assert first.status_code == 202
    assert first.json()["status"] == "cancelling"
    assert recovered.status_code == 200
    assert recovered.json()["status"] == "cancelled"
    assert len(cancellation_calls) == 2


def test_cancel_does_not_redispatch_after_accepted_operation_failure():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    cancellation_calls = []

    def request_cancellation(_config, _execution_name):
        cancellation_calls.append(object())
        if len(cancellation_calls) == 1:
            return AmbiguousCancellationOperation()
        return CompletedCancellationOperation()

    client = TestClient(
        build_test_app(
            submission_repository,
            result_repository,
            request_cancellation=request_cancellation,
        ),
    )
    create_response = client.post(
        "/submissions",
        json=scenario_payload(),
        headers=IDEMPOTENCY_HEADERS,
    )
    submission_id = create_response.json()["submission_id"]
    cancel_token = create_response.json()["cancel_token"]

    first = client.post(
        f"/submissions/{submission_id}/cancel",
        headers={"Authorization": f"Bearer {cancel_token}"},
    )
    submission_repository.submissions[submission_id]["control"][
        "cancellation_started_at"
    ] = (datetime.now(UTC) - timedelta(minutes=3)).isoformat()
    recovered = client.post(
        f"/submissions/{submission_id}/cancel",
        headers={"Authorization": f"Bearer {cancel_token}"},
    )

    assert first.status_code == 202
    assert recovered.status_code == 202
    assert recovered.json()["status"] == "cancelling"
    assert len(cancellation_calls) == 1
    assert (
        submission_repository.fetch_control(submission_id).cancellation_state
        == "requested"
    )


def test_cancel_definitive_rejection_leaves_starting_result_active():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()

    def reject_cancellation(_config, _execution_name):
        raise CancellationRequestRejectedError

    client = TestClient(
        build_test_app(
            submission_repository,
            result_repository,
            request_cancellation=reject_cancellation,
        ),
    )
    create_response = client.post(
        "/submissions",
        json=scenario_payload(),
        headers=IDEMPOTENCY_HEADERS,
    )
    submission_id = create_response.json()["submission_id"]
    cancel_token = create_response.json()["cancel_token"]
    result_repository.write_update(
        submission_id,
        status="starting",
        progress={
            "phase": "starting",
            "current_step": 0,
            "total_steps": 5000,
            "message": "Preparing training",
        },
    )

    response = client.post(
        f"/submissions/{submission_id}/cancel",
        headers={"Authorization": f"Bearer {cancel_token}"},
    )

    assert response.status_code == 502
    assert result_repository.fetch(submission_id)["status"] == "starting"
    control = submission_repository.fetch_control(submission_id)
    assert control.cancellation_state == "idle"


def test_cancel_does_not_overwrite_result_completed_before_transition():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    published_events = []
    client = TestClient(
        build_test_app(
            submission_repository,
            result_repository,
            request_cancellation=lambda _config, _execution_name: (
                CompletedCancellationOperation()
            ),
            publish_result_event=lambda **kwargs: published_events.append(kwargs),
        ),
    )
    create_response = client.post(
        "/submissions",
        json=scenario_payload(),
        headers=IDEMPOTENCY_HEADERS,
    )
    submission_id = create_response.json()["submission_id"]
    cancel_token = create_response.json()["cancel_token"]
    result_repository.write_update(
        submission_id,
        status="running",
        progress={
            "phase": "running",
            "current_step": 12,
            "total_steps": 100,
            "message": "Training",
        },
    )
    original_transition = result_repository.transition_status_preserving_progress

    def complete_before_transition(*args, **kwargs):
        result_repository.results[submission_id] = result_document(
            submission_id,
            "completed",
        )
        return original_transition(*args, **kwargs)

    result_repository.transition_status_preserving_progress = complete_before_transition

    response = client.post(
        f"/submissions/{submission_id}/cancel",
        headers={"Authorization": f"Bearer {cancel_token}"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert result_repository.fetch(submission_id)["status"] == "completed"
    assert published_events == []


def test_cancel_rejects_completed_job():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    client = TestClient(build_test_app(submission_repository, result_repository))
    create_response = client.post(
        "/submissions",
        json=scenario_payload(),
        headers=IDEMPOTENCY_HEADERS,
    )
    submission_id = create_response.json()["submission_id"]
    cancel_token = create_response.json()["cancel_token"]
    submission_repository.mark_dispatched(submission_id, EXECUTION_NAME)
    result_repository.results[submission_id] = result_document(
        submission_id,
        "completed",
    )

    response = client.post(
        f"/submissions/{submission_id}/cancel",
        headers={"Authorization": f"Bearer {cancel_token}"},
    )

    assert response.status_code == 409


def test_cancel_is_idempotent_after_job_is_cancelled():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    cancellation_calls = []
    client = TestClient(
        build_test_app(
            submission_repository,
            result_repository,
            request_cancellation=lambda *args: cancellation_calls.append(args),
        ),
    )
    create_response = client.post(
        "/submissions",
        json=scenario_payload(),
        headers=IDEMPOTENCY_HEADERS,
    )
    submission_id = create_response.json()["submission_id"]
    cancel_token = create_response.json()["cancel_token"]
    submission_repository.mark_dispatched(submission_id, EXECUTION_NAME)
    result_repository.results[submission_id] = result_document(
        submission_id,
        "cancelled",
        current_step=12,
        total_steps=100,
    )

    response = client.post(
        f"/submissions/{submission_id}/cancel",
        headers={"Authorization": f"Bearer {cancel_token}"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert cancellation_calls == []


def test_get_result_returns_existing_result():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository(
        initial_results={
            "submission-1": result_document("submission-1", "completed"),
        },
    )
    client = TestClient(build_test_app(submission_repository, result_repository))

    response = client.get("/results/submission-1")

    assert response.status_code == 200
    assert response.json()["submission_id"] == "submission-1"
    assert response.json()["status"] == "completed"


def test_get_result_fails_stale_ambiguous_dispatch_without_redispatching():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    dispatches = []

    def lose_dispatch_response(config, submission_id):
        dispatches.append((config, submission_id))
        raise TimeoutError

    client = TestClient(
        build_test_app(
            submission_repository,
            result_repository,
            run_training=lose_dispatch_response,
        ),
    )
    create_response = client.post(
        "/submissions",
        json=scenario_payload(),
        headers=IDEMPOTENCY_HEADERS,
    )
    submission_id = create_response.json()["submission_id"]
    submission_repository.submissions[submission_id]["control"][
        "dispatch_started_at"
    ] = (datetime.now(UTC) - timedelta(minutes=6)).isoformat()

    response = client.get(f"/results/{submission_id}")

    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert response.json()["error"] == (
        "Training dispatch outcome could not be confirmed"
    )
    assert len(dispatches) == 1


def test_get_result_does_not_fail_stale_dispatch_after_trainer_progress():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()

    def lose_dispatch_response(_config, _submission_id):
        raise TimeoutError

    client = TestClient(
        build_test_app(
            submission_repository,
            result_repository,
            run_training=lose_dispatch_response,
        ),
    )
    create_response = client.post(
        "/submissions",
        json=scenario_payload(),
        headers=IDEMPOTENCY_HEADERS,
    )
    submission_id = create_response.json()["submission_id"]
    submission_repository.submissions[submission_id]["control"][
        "dispatch_started_at"
    ] = (datetime.now(UTC) - timedelta(minutes=6)).isoformat()
    result_repository.results[submission_id] = result_document(
        submission_id,
        "running",
    )

    response = client.get(f"/results/{submission_id}")

    assert response.status_code == 200
    assert response.json()["status"] == "running"
    assert result_repository.fetch(submission_id)["status"] == "running"


def test_get_result_marks_active_result_failed_after_exact_cloud_run_failure():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    submission_repository.submissions["submission-1"] = {
        "submission_id": "submission-1",
        "control": {
            "cancel_token_hash": "a" * 64,
            "execution_name": (
                "projects/test/locations/asia-northeast1/jobs/test-trainer/"
                "executions/test-trainer-abcde"
            ),
        },
    }
    result_repository = FakeResultRepository(
        initial_results={
            "submission-1": result_document(
                "submission-1",
                "running",
                current_step=12,
                total_steps=1500000,
            ),
        },
    )

    def read_execution_outcome(config, execution_name):
        assert execution_name.endswith("/executions/test-trainer-abcde")
        assert config.job_path.endswith("/jobs/test-trainer")
        result_repository.results["submission-1"]["progress"]["current_step"] = 60
        return ExecutionOutcome(
            status="failed",
            message="The configured timeout was reached.",
        )

    client = TestClient(
        build_test_app(
            submission_repository,
            result_repository,
            read_execution_outcome=read_execution_outcome,
        ),
    )

    response = client.get("/results/submission-1")

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "failed"
    assert result["progress"]["phase"] == "failed"
    assert result["progress"]["current_step"] == 60
    assert result["progress"]["total_steps"] == 1500000
    assert "test-trainer-abcde" in result["error"]
    assert "configured timeout" in result["error"]


def test_execution_reconciliation_does_not_overwrite_concurrent_completion():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository(
        initial_submissions={
            "submission-1": {
                "submission_id": "submission-1",
                "control": {
                    "cancel_token_hash": "a" * 64,
                    "execution_name": EXECUTION_NAME,
                },
            },
        },
    )
    result_repository = FakeResultRepository(
        initial_results={
            "submission-1": result_document("submission-1", "running"),
        },
    )
    published_events = []
    original_transition = result_repository.transition_status_preserving_progress

    def complete_before_transition(*args, **kwargs):
        result_repository.results["submission-1"] = result_document(
            "submission-1",
            "completed",
        )
        return original_transition(*args, **kwargs)

    result_repository.transition_status_preserving_progress = complete_before_transition
    client = TestClient(
        build_test_app(
            submission_repository,
            result_repository,
            read_execution_outcome=lambda _config, _execution_name: ExecutionOutcome(
                status="failed", message="late failure"
            ),
            publish_result_event=lambda **kwargs: published_events.append(kwargs),
        ),
    )

    response = client.get("/results/submission-1")

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert published_events == []


def test_get_result_marks_cancelling_result_cancelled_after_exact_execution():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository(
        initial_submissions={
            "submission-1": {
                "submission_id": "submission-1",
                "control": {
                    "cancel_token_hash": "a" * 64,
                    "execution_name": (
                        "projects/test/locations/asia-northeast1/jobs/test-trainer/"
                        "executions/test-trainer-abcde"
                    ),
                },
            },
        },
    )
    result_repository = FakeResultRepository(
        initial_results={
            "submission-1": result_document(
                "submission-1",
                "cancelling",
                current_step=12,
                total_steps=100,
            ),
        },
    )
    published_events = []
    client = TestClient(
        build_test_app(
            submission_repository,
            result_repository,
            read_execution_outcome=lambda _config, _execution_name: ExecutionOutcome(
                status="cancelled",
                message="Cloud Run execution cancelled",
            ),
            publish_result_event=lambda **kwargs: published_events.append(kwargs),
        ),
    )

    response = client.get("/results/submission-1")

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert response.json()["progress"] == {
        "phase": "cancelled",
        "current_step": 12,
        "total_steps": 100,
        "message": "Training cancelled",
    }
    assert [event["status"].value for event in published_events] == ["cancelled"]


def test_get_result_returns_404_for_missing_result():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    client = TestClient(build_test_app(submission_repository, result_repository))

    response = client.get("/results/missing")

    assert response.status_code == 404


def test_submission_and_result_flow_integrates_with_trainer():
    from fastapi.testclient import TestClient

    submission_repository = FakeSubmissionRepository()
    result_repository = FakeResultRepository()
    published_events = []

    def run_trainer(config, submission_id):
        trainer.job.run_training_job(
            trainer.job.TrainerConfig(
                db_id=config.db_id,
                model_bucket="model-bucket",
                submission_id=submission_id,
                pubsub_topic="test-topic",
                project_id="test-project",
            ),
            create_db=lambda db_id: object(),
            create_submission_repository=lambda db: submission_repository,
            create_result_repository=lambda db: result_repository,
            train_model=lambda **kwargs: {
                "summary": resolved_training_summary(),
                "replay_bundle_dir": "replay_bundle",
            },
            upload_model=lambda **kwargs: completed_artifacts(
                "model-bucket",
                submission_id,
            ),
            publish_event=lambda **kwargs: published_events.append(kwargs),
        )
        return EXECUTION_NAME

    client = TestClient(
        build_test_app(
            submission_repository,
            result_repository,
            run_training=run_trainer,
        ),
    )

    create_response = client.post(
        "/submissions",
        json=scenario_payload(),
        headers=IDEMPOTENCY_HEADERS,
    )
    submission_id = create_response.json()["submission_id"]

    result_response = client.get(f"/results/{submission_id}")

    assert create_response.status_code == 200
    assert result_response.status_code == 200
    assert result_response.json()["status"] == "completed"
    assert "summary" not in result_response.json()
    assert result_response.json()["result_bundle"]["summary"] == {
        "success_rate": None,
        "average_episode_reward": None,
        "average_episode_steps": None,
        "configuration": resolved_training_configuration(),
    }
    assert "artifacts" not in result_response.json()
    assert (
        result_response.json()["result_bundle"]["artifacts"]["onnx_model"]["bucket"]
        == "model-bucket"
    )
    assert [event["status"].value for event in published_events] == [
        "starting",
        "running",
        "completed",
    ]
