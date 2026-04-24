# Development Guide

This page is the human-facing entry point for day-to-day development. It
summarizes the common setup and workflow, while the canonical detailed rules
still live under `rules/`.

## Where To Read What

- project and service payloads: [data-models.md](/workspaces/EmbodiedLab/docs/data-models.md)
- coding conventions: [rules/coding-style.md](/workspaces/EmbodiedLab/rules/coding-style.md)
- common commands and deployment flow: [rules/command.md](/workspaces/EmbodiedLab/rules/command.md)

## Environment Setup

The project uses `uv` for Python environment and package management.

Dependency groups are split by responsibility:

- `embodiedlab`: shared library dependencies used by multiple services
  Currently this is mainly the shared modeling layer such as `pydantic`.
- `server`: API-service runtime dependencies
- `trainer`: training-job runtime dependencies
- `notification`: WebSocket relay runtime dependencies
- `dev`: test, lint, and local-development tooling

In practice, `embodiedlab` is the shared base group. You typically combine it
with one service group rather than installing it alone.

Full local environment:

```bash
uv sync --frozen --all-groups
```

Service-scoped environment:

```bash
uv sync --frozen --group embodiedlab --group server
uv sync --frozen --group embodiedlab --group trainer
uv sync --frozen --group embodiedlab --group notification
```

Or via Make:

```bash
make local_setup
```

## Local Development

Run the API locally:

```bash
make server_local
```

Run the test suite:

```bash
make local_test
uv run pytest
```

Run lint checks:

```bash
uv run ruff check embodiedlab server trainer tests notification
npx markdownlint-cli2 README.md CLAUDE.md docs/**/*.md
```

## Manual End-To-End Flow

When infrastructure is already deployed and `.env` is configured:

```bash
make submit
make train
make get_result
make get_result_ws
```

## Notes On `rules/`

`rules/` remains the source of truth for concise operational and coding rules.
`docs/` is intended for explanatory project documentation and overview pages.
