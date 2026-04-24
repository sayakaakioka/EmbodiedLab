#!/usr/bin/env bash
set -euo pipefail

cd /workspaces/EmbodiedLab

if [ -d .venv ] && [ ! -x .venv/bin/python ]; then
  echo "Broken .venv detected. Recreating..."
  rm -rf .venv
fi

if [ -x .venv/bin/python ]; then
  .venv/bin/python --version >/dev/null 2>&1 || {
    echo "Invalid .venv detected. Recreating..."
    rm -rf .venv
  }
fi

uv sync --frozen --all-groups
node --version
npx --version
gcloud --version
docker --version
docker buildx version
uv --version
