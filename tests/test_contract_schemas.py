import json
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

from server.main import create_app
from tools.export_contract_schemas import (
    CONTRACT_DIR,
    render_contract_schemas,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "envforge"


def test_committed_contract_schemas_match_fresh_export():
    expected = render_contract_schemas()
    actual_names = {
        path.name for path in CONTRACT_DIR.glob("*.schema.json") if path.is_file()
    }

    assert actual_names == set(expected)
    for file_name, content in expected.items():
        assert (CONTRACT_DIR / file_name).read_text(encoding="utf-8") == content


def test_public_result_contracts_reject_unknown_top_level_fields():
    rendered = render_contract_schemas()

    result_document = json.loads(rendered["result-document.schema.json"])
    result_bundle = json.loads(rendered["result-bundle.schema.json"])

    assert result_document["additionalProperties"] is False
    assert result_document["$defs"]["ResultBundle"]["additionalProperties"] is False
    assert result_bundle["additionalProperties"] is False


def test_result_document_schema_rejects_incoherent_terminal_states():
    rendered = render_contract_schemas()
    schema = json.loads(rendered["result-document.schema.json"])
    validator = Draft202012Validator(schema)
    completed = json.loads(
        (FIXTURE_DIR / "navigation_completed_result_document.json").read_text(
            encoding="utf-8",
        ),
    )
    validator.validate(completed)

    missing_summary = deepcopy(completed)
    missing_summary["result_bundle"]["summary"] = None
    with pytest.raises(JsonSchemaValidationError):
        validator.validate(missing_summary)

    outer_status_mismatch = deepcopy(completed)
    outer_status_mismatch["status"] = "running"
    outer_status_mismatch["progress"]["phase"] = "running"
    with pytest.raises(JsonSchemaValidationError):
        validator.validate(outer_status_mismatch)

    missing_progress = deepcopy(completed)
    missing_progress["progress"] = None
    with pytest.raises(JsonSchemaValidationError):
        validator.validate(missing_progress)


def test_replay_manifest_schema_rejects_phase_contract_violations():
    rendered = render_contract_schemas()
    schema = json.loads(rendered["replay-bundle-manifest.schema.json"])
    validator = Draft202012Validator(schema)
    manifest = json.loads(
        (FIXTURE_DIR / "navigation_replay_bundle_manifest.json").read_text(
            encoding="utf-8",
        ),
    )
    validator.validate(manifest)

    unsafe_path = deepcopy(manifest)
    unsafe_path["chunks"][0]["path"] = "train/../chunk.jsonl.gz"
    with pytest.raises(JsonSchemaValidationError):
        validator.validate(unsafe_path)

    wrong_policy_mode = deepcopy(manifest)
    wrong_policy_mode["chunks"][0]["policy_mode"] = "deterministic"
    with pytest.raises(JsonSchemaValidationError):
        validator.validate(wrong_policy_mode)

    missing_train_range = deepcopy(manifest)
    missing_train_range["chunks"][0]["start_step"] = None
    with pytest.raises(JsonSchemaValidationError):
        validator.validate(missing_train_range)

    empty_manifest = deepcopy(manifest)
    empty_manifest["chunks"] = []
    with pytest.raises(JsonSchemaValidationError):
        validator.validate(empty_manifest)


def test_openapi_exposes_typed_sdk_responses():
    openapi = create_app().openapi()

    submission_response = openapi["paths"]["/submissions"]["post"]["responses"]["200"][
        "content"
    ]["application/json"]["schema"]
    cancellation_response = openapi["paths"]["/submissions/{submission_id}/cancel"][
        "post"
    ]["responses"]["200"]["content"]["application/json"]["schema"]
    result_response = openapi["paths"]["/results/{submission_id}"]["get"]["responses"][
        "200"
    ]["content"]["application/json"]["schema"]

    assert submission_response == {"$ref": "#/components/schemas/SubmissionResponse"}
    assert "/submissions/{submission_id}/train" not in openapi["paths"]
    assert cancellation_response == {"$ref": "#/components/schemas/ResultDocument"}
    assert result_response == {"$ref": "#/components/schemas/ResultDocument"}
