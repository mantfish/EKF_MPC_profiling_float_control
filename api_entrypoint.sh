#!/bin/sh
set -e

PORT=${PORT:-8080}
exec uv run uvicorn api:app --host 0.0.0.0 --port "$PORT"