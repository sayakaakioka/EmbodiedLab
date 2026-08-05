# Code style

## Python

- Python 3.13, `ruff` for linting and formatting (line length 88, all rules
  enabled, `D203`/`D213` ignored)
- All files use `from __future__ import annotations`
- Pydantic v2 models throughout; use `model_dump(mode="json")` when writing to
  Firestore
- `ResultStatus` is a `StrEnum` — values are lowercase strings used directly in
  Firestore documents
- Uploaded artifact payloads are plain JSON dicts used to build
  `ResultBundle.artifacts`. GCS artifacts use
  `{ "storage": "gcs", "bucket": "...", "path": "...", "format": "...",
  "size_bytes": 123, "sha256": "..." }`; completed results include
  `onnx_model` (`policy.onnx`) and `replay_bundle`, each with byte size and
  SHA-256.
  Result documents and events do not duplicate summary or artifact values at
  the top level.

## Markdown

- Lint with:

  ```bash
  uv run pymarkdown scan --recurse --respect-gitignore README.md AGENTS.md docs rules
  ```

- First line of every `.md` file must be an H1 heading (MD041)
