from trainer.config import load_trainer_config


def test_load_trainer_config_builds_exact_cloud_run_execution(monkeypatch):
    values = {
        "DB_ID": "test-db",
        "MODEL_BUCKET": "test-bucket",
        "SUBMISSION_ID": "submission-1",
        "PUBSUB_TOPIC": "test-topic",
        "PROJECT_ID": "test-project",
        "REGION": "asia-northeast1",
        "CLOUD_RUN_JOB": "test-trainer",
        "CLOUD_RUN_EXECUTION": "test-trainer-abcde",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    config = load_trainer_config()

    assert config.execution_name == (
        "projects/test-project/locations/asia-northeast1/jobs/test-trainer/"
        "executions/test-trainer-abcde"
    )
