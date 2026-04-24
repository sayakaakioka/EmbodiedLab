# syntax=docker/dockerfile:1

FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups --group embodiedlab --group notification --no-install-project

COPY notification ./notification
COPY embodiedlab ./embodiedlab
RUN uv sync --frozen --no-default-groups --group embodiedlab --group notification


FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    PORT=8080

WORKDIR /app
COPY --from=builder /app /app

EXPOSE 8080

CMD ["uvicorn", "notification.main:app", "--host", "0.0.0.0", "--port", "8080"]
