"""Export deterministic JSON Schemas for Unity SDK contract generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from embodiedlab.api_models import SubmissionResponse
from embodiedlab.result_models import (
    ReplayBundleManifest,
    ReplayLogStep,
    ResultBundle,
    ResultDocument,
)
from embodiedlab.schemas import ScenarioBundle

if TYPE_CHECKING:
    from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DIR = PROJECT_ROOT / "contracts" / "v0"
JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"

CONTRACT_MODELS: tuple[tuple[str, type[BaseModel]], ...] = (
    ("scenario-bundle.schema.json", ScenarioBundle),
    ("submission-response.schema.json", SubmissionResponse),
    ("result-document.schema.json", ResultDocument),
    ("result-bundle.schema.json", ResultBundle),
    ("replay-bundle-manifest.schema.json", ReplayBundleManifest),
    ("replay-log-step.schema.json", ReplayLogStep),
)


def render_contract_schemas() -> dict[str, str]:
    """Return canonical JSON text for every published contract model."""
    rendered = {}
    for file_name, model_type in CONTRACT_MODELS:
        schema = {
            "$schema": JSON_SCHEMA_DIALECT,
            **model_type.model_json_schema(mode="serialization"),
        }
        rendered[file_name] = (
            json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
    return rendered


def write_contract_schemas() -> None:
    """Write all generated schemas to the versioned contract directory."""
    CONTRACT_DIR.mkdir(parents=True, exist_ok=True)
    for file_name, content in render_contract_schemas().items():
        (CONTRACT_DIR / file_name).write_text(content, encoding="utf-8")


def check_contract_schemas() -> bool:
    """Return whether committed schema files match a fresh export."""
    expected = render_contract_schemas()
    actual_names = {
        path.name for path in CONTRACT_DIR.glob("*.schema.json") if path.is_file()
    }
    expected_names = set(expected)
    valid = actual_names == expected_names

    for file_name, content in expected.items():
        path = CONTRACT_DIR / file_name
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            print(f"Contract schema is out of date: {path.relative_to(PROJECT_ROOT)}")
            valid = False

    for file_name in sorted(actual_names - expected_names):
        print(f"Unexpected contract schema: contracts/v0/{file_name}")

    return valid


def main() -> int:
    """Run the contract exporter or drift check."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when committed schemas differ from a fresh export.",
    )
    args = parser.parse_args()

    if args.check:
        return 0 if check_contract_schemas() else 1

    write_contract_schemas()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
