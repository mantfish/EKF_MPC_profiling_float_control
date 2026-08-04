#!/bin/sh
set -e

PORT=${PORT:-8080}
# Unbuffered stdout/stderr: without this, Python fully-buffers output when
# it's not attached to a terminal (i.e. piped to Render's log collector), so
# log lines can sit unflushed and get silently lost when the process is
# SIGKILLed on OOM instead of exiting gracefully.
export PYTHONUNBUFFERED=1
exec uv run uvicorn api:app --host 0.0.0.0 --port "$PORT"