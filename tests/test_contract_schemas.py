from server.main import create_app
from tools.export_contract_schemas import (
    CONTRACT_DIR,
    render_contract_schemas,
)


def test_committed_contract_schemas_match_fresh_export():
    expected = render_contract_schemas()
    actual_names = {
        path.name for path in CONTRACT_DIR.glob("*.schema.json") if path.is_file()
    }

    assert actual_names == set(expected)
    for file_name, content in expected.items():
        assert (CONTRACT_DIR / file_name).read_text(encoding="utf-8") == content


def test_openapi_exposes_typed_sdk_responses():
    openapi = create_app().openapi()

    submission_response = openapi["paths"]["/submissions"]["post"]["responses"]["200"][
        "content"
    ]["application/json"]["schema"]
    training_response = openapi["paths"]["/submissions/{submission_id}/train"]["post"][
        "responses"
    ]["200"]["content"]["application/json"]["schema"]
    cancellation_response = openapi["paths"]["/submissions/{submission_id}/cancel"][
        "post"
    ]["responses"]["200"]["content"]["application/json"]["schema"]
    result_response = openapi["paths"]["/results/{submission_id}"]["get"]["responses"][
        "200"
    ]["content"]["application/json"]["schema"]

    assert submission_response == {"$ref": "#/components/schemas/SubmissionResponse"}
    assert training_response == {"$ref": "#/components/schemas/TrainingResponse"}
    assert cancellation_response == {"$ref": "#/components/schemas/ResultDocument"}
    assert result_response == {"$ref": "#/components/schemas/ResultDocument"}
