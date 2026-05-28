# AGENTS.md

## Boundaries

Prioritize AGENTS.md in the codex folder of Google Drive above everything else.

## Project Overview

EmbodiedLab is the cloud-side training and experiment execution platform for
embodied AI workflows. Its immediate role is to provide a backend that can
accept environment and training definitions, run reinforcement learning jobs in
cloud-friendly runtimes, store resulting artifacts, and expose progress or
result data to clients.

For the EnvForge collaboration, EnvForge is the user-facing Unity application
for designing environments and replaying training behavior, while EmbodiedLab is
the backend training foundation. The two repositories should remain separate and
communicate through explicit data contracts such as Scenario Bundle, Result
Bundle, and Replay Log formats.

## Branch Notice

This repository currently contains an early prototype rather than a production
backend. The current implementation accepts grid-world training requests through
a FastAPI service, stores submissions and results in Firestore, launches a Cloud
Run Job for training, uploads model artifacts to GCS, and relays result events
through Pub/Sub and a WebSocket notification service.

The current training path uses a Python/Gymnasium grid-world environment and
Stable-Baselines3 PPO. EnvForge integration does not require EmbodiedLab to use
Unity or ML-Agents directly. Prefer a cloud-suitable simulation and training
stack when it can reproduce the scenario conditions defined by EnvForge.

The shared Python package is located in `embodiedlab/`. Service entry points are
located in `server/`, `trainer/`, and `notification/`. Tests are located in
`tests/`.

Python dependencies are managed by `uv` through `pyproject.toml` dependency
groups. The project currently declares Python `>=3.13`, while Ruff is configured
with `target-version = "py312"`; treat this as an existing consistency issue to
resolve deliberately rather than changing it incidentally.

Project direction is documented under `docs/vision/`. Current architecture,
development workflow, data models, and the EnvForge integration roadmap are
documented under `docs/implementation/`.

For sub-agent orchestration, read `docs/implementation/subagent-workflow.md`.
