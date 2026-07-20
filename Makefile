-include .env
export

SHELL := /bin/bash

GCLOUD ?= $(shell command -v gcloud 2>/dev/null)
DOCKER ?= $(shell command -v docker 2>/dev/null)
UV ?= $(shell command -v uv 2>/dev/null)
UV_SYNC_ALL_GROUPS := sync --frozen --all-groups

.PHONY: setup_all

setup_all: clear_submission_id gcp_bootstrap deploy_all
	@echo "***** all setup completed *****"


##### dependencies #####
.PHONY: ensure_uv ensure_venv check_deps require_cloud_env require_api_env show_env

ensure_uv:
	@test -n "$(UV)" || { echo "uv not found"; exit 1; }
	@$(UV) --version >/dev/null 2>&1 || { echo "uv is broken"; exit 1; }

ensure_venv: ensure_uv
	@if [ -d .venv ] && [ ! -x .venv/bin/python ]; then \
		echo "Broken .venv detected, recreating"; \
		rm -rf .venv; \
	fi
	@if [ -x .venv/bin/python ]; then \
		.venv/bin/python --version >/dev/null 2>&1 || { \
			echo "Invalid .venv detected, recreating"; \
			rm -rf .venv; \
		}; \
	fi
	@test -f pyproject.toml || { echo "pyproject.toml not found"; exit 1; }
	@test -f uv.lock || { echo "uv.lock not found"; exit 1; }
	@$(UV) $(UV_SYNC_ALL_GROUPS) >/dev/null

check_deps: ensure_venv
	@test -n "$(GCLOUD)" || { echo "gcloud not found"; exit 1; }
	@test -n "$(DOCKER)" || { echo "docker not found"; exit 1; }
	@$(GCLOUD) --version >/dev/null 2>&1 || { echo "gcloud is broken"; exit 1; }
	@$(DOCKER) --version >/dev/null 2>&1 || { echo "docker is broken"; exit 1; }
	@$(UV) --version >/dev/null 2>&1 || { echo "uv is broken"; exit 1; }
	@$(UV) run python --version >/dev/null 2>&1 || { echo "project python is broken"; exit 1; }
	@echo "deps OK"
	@echo " GCLOUD=$(GCLOUD)"
	@echo " DOCKER=$(DOCKER)"
	@echo " UV=$(UV)"

CLOUD_ENV_VARS := \
	PROJECT_ID \
	REGION \
	DB_ID \
	ARTIFACT_REPO \
	MODEL_BUCKET \
	PUBSUB_TOPIC \
	PUBSUB_SUBSCRIPTION \
	RUNTIME_SA_NAME \
	API_SERVICE_NAME \
	TRAINER_JOB_NAME \
	NOTIFICATION_SERVICE_NAME \
	NOTIFICATION_PUSH_PATH

API_ENV_VARS := API_URL

require_cloud_env:
	@missing=0; \
	for name in $(CLOUD_ENV_VARS); do \
		if [ -z "$${!name}" ]; then \
			echo "$$name required"; \
			missing=1; \
		fi; \
	done; \
	exit $$missing

require_api_env:
	@missing=0; \
	for name in $(API_ENV_VARS); do \
		if [ -z "$${!name}" ]; then \
			echo "$$name required"; \
			missing=1; \
		fi; \
	done; \
	exit $$missing

RUNTIME_SA_EMAIL := $(RUNTIME_SA_NAME)@$(PROJECT_ID).iam.gserviceaccount.com
RUN_CANCEL_ROLE_ID ?= embodiedLabExecutionCanceller
RUN_CANCEL_ROLE := projects/$(PROJECT_ID)/roles/$(RUN_CANCEL_ROLE_ID)

show_env: check_deps require_cloud_env
	@echo "PROJECT_ID=$(PROJECT_ID)"
	@echo "REGION=$(REGION)"
	@echo "DB_ID=$(DB_ID)"
	@echo "ARTIFACT_REPO=$(ARTIFACT_REPO)"
	@echo "MODEL_BUCKET=$(MODEL_BUCKET)"
	@echo "PUBSUB_TOPIC=$(PUBSUB_TOPIC)"
	@echo "PUBSUB_SUBSCRIPTION=$(PUBSUB_SUBSCRIPTION)"
	@echo "RUNTIME_SA_NAME=$(RUNTIME_SA_NAME)"
	@echo "RUNTIME_SA_EMAIL=$(RUNTIME_SA_EMAIL)"
	@echo "API_SERVICE_NAME=$(API_SERVICE_NAME)"
	@echo "TRAINER_JOB_NAME=$(TRAINER_JOB_NAME)"
	@echo "NOTIFICATION_SERVICE_NAME=$(NOTIFICATION_SERVICE_NAME)"


##### Google Cloud Run (Service / Job) setup #####
.PHONY: gcp_auth gcp_bootstrap enable_services \
	create_runtime_sa create_artifact_repo create_model_bucket create_firestore_db create_pubsub_topic \
	create_run_cancel_role grant_runtime_roles

gcp_auth: check_deps require_cloud_env
	$(GCLOUD) auth login
	$(GCLOUD) config set project $(PROJECT_ID)

gcp_bootstrap: check_deps require_cloud_env \
	enable_services \
	create_runtime_sa \
	create_artifact_repo \
	create_model_bucket \
	create_firestore_db \
	create_pubsub_topic \
	create_run_cancel_role \
	grant_runtime_roles
	@echo "GCP bootstrap complete"

enable_services: check_deps require_cloud_env
	$(GCLOUD) config set project $(PROJECT_ID)
	$(GCLOUD) services enable \
		run.googleapis.com \
		cloudbuild.googleapis.com \
		artifactregistry.googleapis.com \
		firestore.googleapis.com \
		pubsub.googleapis.com \
		iam.googleapis.com \
		storage.googleapis.com \
		logging.googleapis.com

create_runtime_sa: check_deps require_cloud_env
	@$(GCLOUD) iam service-accounts describe $(RUNTIME_SA_EMAIL) >/dev/null 2>&1 || \
		$(GCLOUD) iam service-accounts create $(RUNTIME_SA_NAME) \
			--display-name="$(RUNTIME_SA_NAME) Runtime"

create_artifact_repo: check_deps require_cloud_env
	@$(GCLOUD) artifacts repositories describe $(ARTIFACT_REPO) \
		--location=$(REGION) >/dev/null 2>&1 || \
		$(GCLOUD) artifacts repositories create $(ARTIFACT_REPO) \
			--repository-format=docker \
			--location=$(REGION) \
			--description="$(ARTIFACT_REPO) container images"

create_model_bucket: check_deps require_cloud_env
	@$(GCLOUD) storage buckets describe gs://$(MODEL_BUCKET) >/dev/null 2>&1 || \
		$(GCLOUD) storage buckets create gs://$(MODEL_BUCKET) \
			--location=$(REGION) \
			--uniform-bucket-level-access
	$(GCLOUD) storage buckets add-iam-policy-binding gs://$(MODEL_BUCKET) \
		--member="allUsers" \
		--role="roles/storage.objectViewer"

create_firestore_db: check_deps require_cloud_env
	@$(GCLOUD) firestore databases describe --database=$(DB_ID) >/dev/null 2>&1 || \
		$(GCLOUD) firestore databases create \
			--database=$(DB_ID) \
			--location=$(REGION) \
			--type=firestore-native

create_pubsub_topic: check_deps require_cloud_env
	@$(GCLOUD) pubsub topics describe $(PUBSUB_TOPIC) >/dev/null 2>&1 || \
		$(GCLOUD) pubsub topics create $(PUBSUB_TOPIC)

create_run_cancel_role: check_deps require_cloud_env
	@$(GCLOUD) iam roles describe $(RUN_CANCEL_ROLE_ID) \
		--project=$(PROJECT_ID) >/dev/null 2>&1 || \
		$(GCLOUD) iam roles create $(RUN_CANCEL_ROLE_ID) \
			--project=$(PROJECT_ID) \
			--title="EmbodiedLab execution canceller" \
			--description="Cancel an EmbodiedLab Cloud Run execution" \
			--permissions=run.executions.cancel \
			--stage=GA
	$(GCLOUD) iam roles update $(RUN_CANCEL_ROLE_ID) \
		--project=$(PROJECT_ID) \
		--permissions=run.executions.cancel \
		--stage=GA

grant_runtime_roles: check_deps require_cloud_env
	$(GCLOUD) projects add-iam-policy-binding $(PROJECT_ID) \
		--member="serviceAccount:$(RUNTIME_SA_EMAIL)" \
		--role="roles/datastore.user"
	$(GCLOUD) projects add-iam-policy-binding $(PROJECT_ID) \
		--member="serviceAccount:$(RUNTIME_SA_EMAIL)" \
		--role="roles/pubsub.publisher"
	$(GCLOUD) projects add-iam-policy-binding $(PROJECT_ID) \
		--member="serviceAccount:$(RUNTIME_SA_EMAIL)" \
		--role="roles/run.jobsExecutorWithOverrides"
	$(GCLOUD) projects add-iam-policy-binding $(PROJECT_ID) \
		--member="serviceAccount:$(RUNTIME_SA_EMAIL)" \
		--role="roles/run.viewer"
	$(GCLOUD) projects add-iam-policy-binding $(PROJECT_ID) \
		--member="serviceAccount:$(RUNTIME_SA_EMAIL)" \
		--role="$(RUN_CANCEL_ROLE)"
	$(GCLOUD) storage buckets add-iam-policy-binding gs://$(MODEL_BUCKET) \
		--member="serviceAccount:$(RUNTIME_SA_EMAIL)" \
		--role="roles/storage.objectCreator"


##### build / deploy #####
.PHONY: deploy_all setup_builder \
	build_api deploy_api \
	build_trainer deploy_trainer \
	build_notification deploy_notification

deploy_all: deploy_api deploy_trainer deploy_notification
	@echo "deploy complete"

BUILDER_NAME := embodiedlab-builder
BUILD_PLATFORM := linux/amd64
BUILD_ATTESTATION_FLAGS := --provenance=false --sbom=false

setup_builder:
	@if ! docker buildx inspect $(BUILDER_NAME) >/dev/null 2>&1; then \
	docker buildx create --name $(BUILDER_NAME) --driver docker-container --use --bootstrap; \
	else \
	docker buildx use $(BUILDER_NAME); \
	docker buildx inspect --bootstrap >/dev/null; \
	fi

ARTIFACT_HOST := $(REGION)-docker.pkg.dev
API_IMAGE := $(ARTIFACT_HOST)/$(PROJECT_ID)/$(ARTIFACT_REPO)/api:latest

build_api: check_deps require_cloud_env setup_builder
	$(DOCKER) buildx build \
		--builder $(BUILDER_NAME) \
		--platform $(BUILD_PLATFORM) \
		$(BUILD_ATTESTATION_FLAGS) \
		-f server/Dockerfile \
		-t $(API_IMAGE) \
		--push .

deploy_api: build_api
	$(GCLOUD) run deploy $(API_SERVICE_NAME) \
		--image $(API_IMAGE) \
		--region $(REGION) \
		--service-account $(RUNTIME_SA_EMAIL) \
		--allow-unauthenticated \
		--memory 1Gi \
		--set-env-vars DB_ID=$(DB_ID),REGION=$(REGION),TRAINER_JOB_NAME=$(TRAINER_JOB_NAME),PROJECT_ID=$(PROJECT_ID),PUBSUB_TOPIC=$(PUBSUB_TOPIC)

TRAINER_IMAGE := $(ARTIFACT_HOST)/$(PROJECT_ID)/$(ARTIFACT_REPO)/trainer:latest
TRAINER_CPU ?= 4
TRAINER_MEMORY ?= 4Gi
TRAINER_TASK_TIMEOUT ?= 24h

build_trainer: check_deps require_cloud_env setup_builder
	$(DOCKER) buildx build \
		--builder $(BUILDER_NAME) \
		--platform $(BUILD_PLATFORM) \
		$(BUILD_ATTESTATION_FLAGS) \
		-f trainer/Dockerfile \
		-t $(TRAINER_IMAGE) \
		--push .

deploy_trainer: build_trainer
	$(GCLOUD) run jobs describe $(TRAINER_JOB_NAME) --region $(REGION) >/dev/null 2>&1 && \
	$(GCLOUD) run jobs update $(TRAINER_JOB_NAME) \
		--image $(TRAINER_IMAGE) \
		--region $(REGION) \
		--service-account $(RUNTIME_SA_EMAIL) \
		--cpu $(TRAINER_CPU) \
		--memory $(TRAINER_MEMORY) \
		--task-timeout $(TRAINER_TASK_TIMEOUT) \
		--update-env-vars DB_ID=$(DB_ID),MODEL_BUCKET=$(MODEL_BUCKET),PROJECT_ID=$(PROJECT_ID),PUBSUB_TOPIC=$(PUBSUB_TOPIC) || \
	$(GCLOUD) run jobs create $(TRAINER_JOB_NAME) \
		--image $(TRAINER_IMAGE) \
		--region $(REGION) \
		--service-account $(RUNTIME_SA_EMAIL) \
		--cpu $(TRAINER_CPU) \
		--memory $(TRAINER_MEMORY) \
		--task-timeout $(TRAINER_TASK_TIMEOUT) \
		--set-env-vars DB_ID=$(DB_ID),MODEL_BUCKET=$(MODEL_BUCKET),PROJECT_ID=$(PROJECT_ID),PUBSUB_TOPIC=$(PUBSUB_TOPIC)

NOTIFICATION_IMAGE := $(ARTIFACT_HOST)/$(PROJECT_ID)/$(ARTIFACT_REPO)/notification:latest

build_notification: check_deps require_cloud_env setup_builder
	$(DOCKER) buildx build \
		--builder $(BUILDER_NAME) \
		--platform $(BUILD_PLATFORM) \
		$(BUILD_ATTESTATION_FLAGS) \
		-f notification/Dockerfile \
		-t $(NOTIFICATION_IMAGE) \
		--push .

deploy_notification: build_notification
	$(GCLOUD) run deploy $(NOTIFICATION_SERVICE_NAME) \
		--image $(NOTIFICATION_IMAGE) \
		--region $(REGION) \
		--service-account $(RUNTIME_SA_EMAIL) \
		--allow-unauthenticated \
		--min-instances 1 \
		--max-instances 1 \
		--set-env-vars DB_ID=$(DB_ID),PROJECT_ID=$(PROJECT_ID)


##### pubsub / notification #####
.PHONY: show_notification_url recreate_pubsub_push

show_notification_url: check_deps require_cloud_env
	@$(GCLOUD) run services describe $(NOTIFICATION_SERVICE_NAME) \
		--region $(REGION) \
		--format='value(status.url)'

recreate_pubsub_push: check_deps require_cloud_env
	@NOTIFICATION_URL="$$( $(GCLOUD) run services describe $(NOTIFICATION_SERVICE_NAME) --region $(REGION) --format='value(status.url)' )"; \
	test -n "$$NOTIFICATION_URL" || { echo "notification service URL not found"; exit 1; }; \
	echo "Using push endpoint: $$NOTIFICATION_URL$(NOTIFICATION_PUSH_PATH)"; \
	$(GCLOUD) pubsub subscriptions delete $(PUBSUB_SUBSCRIPTION) --quiet >/dev/null 2>&1 || true; \
	$(GCLOUD) pubsub subscriptions create $(PUBSUB_SUBSCRIPTION) \
		--topic=$(PUBSUB_TOPIC) \
		--push-endpoint=$$NOTIFICATION_URL$(NOTIFICATION_PUSH_PATH) \
		--enable-message-ordering


##### test #####
.PHONY: submit train get_result get_result_ws

LAST_SUBMISSION_RESPONSE_FILE := .last_submit_response.json
LAST_SUBMISSION_ID_FILE := .last_submission_id
LAST_SUBMISSION_IDEMPOTENCY_KEY_FILE := .last_submission_idempotency_key
LAST_CANCEL_TOKEN_FILE := .last_cancel_token

submit: check_deps require_api_env
	@test -f payload.json
	@test -s $(LAST_SUBMISSION_IDEMPOTENCY_KEY_FILE) || \
		$(UV) run python -c 'import secrets; print(secrets.token_urlsafe(32))' \
		> $(LAST_SUBMISSION_IDEMPOTENCY_KEY_FILE)
	@test -s $(LAST_CANCEL_TOKEN_FILE) || \
		$(UV) run python -c 'import secrets; print(secrets.token_urlsafe(32))' \
		> $(LAST_CANCEL_TOKEN_FILE)
	@curl -s -X POST $(API_URL)/submissions \
		-H "Content-Type: application/json" \
		-H "Idempotency-Key: $$(cat $(LAST_SUBMISSION_IDEMPOTENCY_KEY_FILE))" \
		-H "X-EmbodiedLab-Cancel-Token: $$(cat $(LAST_CANCEL_TOKEN_FILE))" \
		-d @payload.json \
		| tee $(LAST_SUBMISSION_RESPONSE_FILE) \
		| $(UV) run python -m json.tool
	@$(UV) run python -c 'import json; print(json.load(open("$(LAST_SUBMISSION_RESPONSE_FILE)", encoding="utf-8"))["submission_id"])' > $(LAST_SUBMISSION_ID_FILE)
	@echo "Saved submission_id: $$(cat $(LAST_SUBMISSION_ID_FILE))"

train: check_deps require_api_env
	@test -f $(LAST_SUBMISSION_ID_FILE)
	curl -s -X POST $(API_URL)/submissions/$$(cat $(LAST_SUBMISSION_ID_FILE))/train \
		| $(UV) run python -m json.tool

get_result: check_deps require_api_env
	@test -f $(LAST_SUBMISSION_ID_FILE)
	curl -s $(API_URL)/results/$$(cat $(LAST_SUBMISSION_ID_FILE)) \
		| $(UV) run python -m json.tool

get_result_ws: check_deps require_api_env
	@test -f $(LAST_SUBMISSION_ID_FILE)
	@SUBMISSION_ID=$$(cat $(LAST_SUBMISSION_ID_FILE)) $(UV) run python tools/ws_client.py

.PHONY: show_submission_id clear_submission_id

show_submission_id:
	test -f $(LAST_SUBMISSION_ID_FILE)
	@cat $(LAST_SUBMISSION_ID_FILE)

clear_submission_id:
	rm -f $(LAST_SUBMISSION_ID_FILE)
	rm -f $(LAST_SUBMISSION_RESPONSE_FILE)
	rm -f $(LAST_SUBMISSION_IDEMPOTENCY_KEY_FILE)
	rm -f $(LAST_CANCEL_TOKEN_FILE)

##### local checks #####
.PHONY: local_setup lint_python lint_markdown lint test check local_test server_local

local_setup:
	@$(UV) $(UV_SYNC_ALL_GROUPS)

lint_python: local_setup
	$(UV) run ruff check embodiedlab server trainer tests notification

lint_markdown: local_setup
	$(UV) run pymarkdown scan --recurse --respect-gitignore README.md AGENTS.md docs

lint: lint_python lint_markdown

test: local_setup
	$(UV) run pytest

check: lint test

local_test: local_setup
	$(UV) run pytest tests

server_local: check_deps
	$(UV) run uvicorn server.main:app --host 0.0.0.0 --port 8000


##### dev utilities #####
.PHONY: list_trainers auth_docker clear_model_bucket \
	logs_api logs_trainer logs_trainer_exec logs_notification \
	logs_raw

list_trainers: check_deps require_cloud_env
	$(GCLOUD) run jobs executions list \
		--job $(TRAINER_JOB_NAME) \
		--region $(REGION)

auth_docker: check_deps require_cloud_env
	$(GCLOUD) auth configure-docker $(REGION)-docker.pkg.dev

clear_model_bucket: check_deps require_cloud_env
	$(GCLOUD) storage rm --recursive gs://$(MODEL_BUCKET)/**

LOG_LIMIT ?= 50

logs_api: check_deps require_cloud_env
	$(GCLOUD) logging read \
		'resource.type="cloud_run_revision" AND resource.labels.service_name="$(API_SERVICE_NAME)"' \
		--limit $(LOG_LIMIT) \
		--format="value(textPayload)"

logs_trainer: check_deps require_cloud_env
	@EXEC=$$($(GCLOUD) run jobs executions list \
		--job $(TRAINER_JOB_NAME) \
		--region $(REGION) \
		--limit 1 \
		--format="value(name)"); \
	test -n "$$EXEC" || { echo "No execution found"; exit 1; }; \
	echo "Execution: $$EXEC"; \
	$(GCLOUD) logging read \
		"resource.type=cloud_run_job AND resource.labels.job_name=$(TRAINER_JOB_NAME) AND labels.\"run.googleapis.com/execution_name\"=$$EXEC" \
		--limit $(LOG_LIMIT) \
		--format="value(textPayload)"

logs_trainer_exec: check_deps require_cloud_env
	@test -n "$(EXECUTION_NAME)" || { echo "EXECUTION_NAME required"; exit 1; }
	$(GCLOUD) logging read \
		"resource.type=cloud_run_job AND resource.labels.job_name=$(TRAINER_JOB_NAME) AND labels.\"run.googleapis.com/execution_name\"=$(EXECUTION_NAME)" \
		--limit $(LOG_LIMIT) \
		--format="value(textPayload)"

logs_notification: check_deps require_cloud_env
	$(GCLOUD) logging read \
		'resource.type="cloud_run_revision" AND resource.labels.service_name="$(NOTIFICATION_SERVICE_NAME)"' \
		--limit $(LOG_LIMIT) \
		--format="value(textPayload)"

logs_raw: check_deps require_cloud_env
	$(GCLOUD) logging read \
		'resource.type="cloud_run_revision"' \
		--limit $(LOG_LIMIT)
