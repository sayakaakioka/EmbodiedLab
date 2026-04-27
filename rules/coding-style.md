# Code style

## Python

- Python 3.12, `ruff` for linting and formatting (line length 88, all rules
  enabled, `D203`/`D213` ignored)
- All files use `from __future__ import annotations`
- Pydantic v2 models throughout; use `model_dump(mode="json")` when writing to
  Firestore
- `ResultStatus` is a `StrEnum` — values are lowercase strings used directly in
  Firestore documents
- Result `artifacts` payloads are plain JSON dicts. GCS artifacts use
  `{ "storage": "gcs", "bucket": "...", "path": "..." }`; completed training
  results should include `model` (`policy.zip`), `onnx_model` (`policy.onnx`),
  and `sentis_model` (`policy.sentis.onnx`) when upload succeeds.

## Markdown

- Lint and format with `npx markdownlint-cli2 --fix "**/*.md"`
- Config in `.markdownlint-cli2.yaml`: line length 120, tables/code
  blocks/headings excluded from line-length check
- First line of every `.md` file must be an H1 heading (MD041)
