include .env
export

.PHONY: setup test

PYTHON := $(shell command -v python3 2>/dev/null || command -v python)
LAST_RESPONSE_FILE := .last_submit_response.json
LAST_SUBMISSION_ID_FILE := .last_submission_id

setup: setup_api setup_trainer
	test -d .venv || $(PYTHON) -m venv .venv
	.venv/bin/python -m pip install --upgrade pip

setup_api:
	.venv/bin/python -m pip install -r requirements-api.txt

setup_trainer:
	.venv/bin/python -m pip install -r requirements-job.txt

test:
	.venv/bin/python -m pip install -r requirements-dev.txt
	.venv/bin/python -m pytest tests

deploy_api:
	gcloud run deploy $(API_NAME) \
		--source . \
		--region $(REGION) \
		--memory 1Gi \
		--set-build-env-vars GOOGLE_PYTHON_VERSION=3.13 \
		--set-env-vars DB_ID=$(DB_ID),JOB_PATH=$(JOB_PATH),REGION=$(REGION)

build_trainer:
	docker buildx build \
		--platform linux/amd64 \
		-f trainer/Dockerfile.job \
		-t $(TRAINER_REPO) \
		--push \
		.

deploy_trainer: build_trainer
	gcloud run jobs update $(TRAINER_NAME) \
		--image $(TRAINER_REPO) \
		--region $(REGION) \
		--update-env-vars DB_ID=$(DB_ID),MODEL_BUCKET=$(MODEL_BUCKET)

### dev utilities ###
server_local_run:
	.venv/bin/python -m uvicorn server.main:app --host 0.0.0.0 --port 8000

list_trainers:
	gcloud run jobs executions list \
		--job $(TRAINER_NAME) \
		--region $(REGION)

gcloud_check:
	gcloud auth list
	gcloud config get-value account
	gcloud config get-value project

auth_docker:
	gcloud auth configure-docker $(REGION)-docker.pkg.dev

job_log:
	EXECUTION_NAME=$$(gcloud run jobs executions list \
		--job $(TRAINER_NAME) \
		--region $(REGION) \
		--limit 1 \
		--format="value(name)"); \
	gcloud logging read \
		"resource.type=cloud_run_job AND resource.labels.job_name=$(TRAINER_NAME) AND labels.\"run.googleapis.com/execution_name\"=$$EXECUTION_NAME" \
		--limit 50

accounts:
	gcloud run jobs describe $(TRAINER_NAME) --region $(REGION)
	gcloud run services describe $(API_NAME) --region $(REGION)

### test utilities ###
submit:
	curl -s -X POST $(API_URL)/submissions \
		-H "Content-Type: application/json" \
		-d @payload.json \
		| tee $(LAST_RESPONSE_FILE) \
		| jq .
	jq -r '.submission_id' $(LAST_RESPONSE_FILE) > $(LAST_SUBMISSION_ID_FILE)
	@echo "Saved submission_id: $$(cat $(LAST_SUBMISSION_ID_FILE))"

show_submission_id:
	test -f $(LAST_SUBMISSION_ID_FILE)
	@cat $(LAST_SUBMISSION_ID_FILE)

clear_submission_id:
	rm -f $(LAST_SUBMISSION_ID_FILE)
	rm -f $(LAST_RESPONSE_FILE)

train:
	test -f $(LAST_SUBMISSION_ID_FILE)
	curl -s -X POST $(API_URL)/submissions/$$(cat $(LAST_SUBMISSION_ID_FILE))/train

get_result:
	test -f $(LAST_SUBMISSION_ID_FILE)
	curl -s $(API_URL)/results/$$(cat $(LAST_SUBMISSION_ID_FILE))
