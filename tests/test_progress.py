from embodiedlab.result_models import Progress, ResultStatus
from trainer.progress import (
    completed_progress,
    failed_progress,
    running_progress,
    starting_progress,
)


def test_progress_helpers_return_progress_models():
    progress = running_progress(100)

    assert isinstance(progress, Progress)
    assert progress.phase is ResultStatus.RUNNING
    assert progress.current_step == 0
    assert progress.total_steps == 100
    assert progress.message == "Training"


def test_failed_progress_defaults_to_zero_total_steps():
    progress = failed_progress("Training failed")

    assert progress.phase is ResultStatus.FAILED
    assert progress.current_step == 0
    assert progress.total_steps == 0
    assert progress.message == "Training failed"


def test_starting_and_completed_progress():
    starting = starting_progress(100)
    completed = completed_progress(100)

    assert starting.phase is ResultStatus.STARTING
    assert starting.current_step == 0
    assert completed.phase is ResultStatus.COMPLETED
    assert completed.current_step == 100
