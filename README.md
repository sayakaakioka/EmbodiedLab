# EmbodiedLab

EmbodiedLab is an experimental platform for embodied AI research. The current prototype accepts grid-world training requests through a Cloud Run API, starts a Cloud Run Job to train a reinforcement learning policy, and stores the trained model artifact in GCS.

The project is intentionally small right now: it is focused on a minimal end-to-end loop from environment definition to model training and result retrieval.

## Architecture

```text
Client
  -> Cloud Run API
    -> Firestore submissions/{submission_id}
    -> Firestore results/{submission_id}
    -> Cloud Run Job
      -> Firestore submission lookup
      -> GridWorld PPO training
      -> GCS model upload
      -> Firestore result update
```

## Packages

```text
embodiedlab/
  Shared domain models, request schemas, result models, GridWorld environment,
  and training logic.

server/
  Cloud Run API entrypoint, routes, dependencies, and service helpers.

trainer/
  Cloud Run Job entrypoint and orchestration for loading submissions, training,
  uploading artifacts, and updating results.

tests/
  Pytest coverage for schemas, conversion, API routes, result models, progress
  helpers, and trainer job flow.
```

## API

### Create Submission

```http
POST /submissions
Content-Type: application/json
```

Example payload:

```json
{
  "environment": {
    "size": [4, 4],
    "obstacles": [{"x": 1, "y": 1}],
    "goal": {"x": 3, "y": 3},
    "robot_start": {"x": 0, "y": 0}
  },
  "robot": {
    "type": "simple"
  },
  "training": {
    "algorithm": "ppo",
    "timesteps": 5000
  }
}
```

Response:

```json
{
  "status": "accepted",
  "submission_id": "..."
}
```

### Start Training

```http
POST /submissions/{submission_id}/train
```

This creates or replaces `results/{submission_id}` with `queued` status, then starts the configured Cloud Run Job with `SUBMISSION_ID`.

### Get Result

```http
GET /results/{submission_id}
```

Result documents include:

- `status`: `queued`, `starting`, `running`, `completed`, or `failed`
- `progress`: phase, current step, total steps, and message
- `summary`: training and evaluation summary when completed
- `artifacts`: GCS model location when completed
- `error`: failure detail when failed

## Configuration

The `Makefile` includes `.env` and exports its variables.

Required API variables:

- `DB_ID`
- `REGION`
- `JOB_PATH`

Required trainer variables:

- `DB_ID`
- `MODEL_BUCKET`
- `SUBMISSION_ID` supplied by the API when starting the Cloud Run Job

Deployment and utility variables:

- `PROJECT_ID`
- `NAME_PREFIX`
- `API_NAME`
- `TRAINER_NAME`
- `TRAINER_REPO`
- `API_URL`

## Development

Create the virtual environment and install API/trainer dependencies:

```bash
make setup
```

Run the API locally:

```bash
make server_local_run
```

Run tests:

```bash
make test
```

`make test` installs `requirements-dev.txt` and runs `pytest tests`.

## Deployment

Deploy the API:

```bash
make deploy_api
```

Build and deploy the trainer image/job:

```bash
make deploy_trainer
```

Useful Cloud Run utilities:

```bash
make list_trainers
make job_log
make accounts
```

## Manual Flow

Submit the sample payload:

```bash
make submit
```

Start training for the last submission:

```bash
make train
```

Fetch the last result:

```bash
make get_result
```

## Current Scope

The current implementation supports:

- Grid-world environment definitions
- A simple robot descriptor
- PPO training through Stable-Baselines3
- Firestore-backed submissions and results
- GCS model artifact upload
- Cloud Run API and Cloud Run Job deployment

## Non-Goals For Now

- High-fidelity simulation
- User-imported robot models
- Multiple training algorithms in production
- UI or visualization
- Production-grade authentication and quota handling

## License

To be decided.
