FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN chmod +x /app/api_entrypoint.sh

ENV PORT=8080

ENTRYPOINT ["./api_entrypoint.sh"]