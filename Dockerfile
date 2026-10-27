# syntax=docker/dockerfile:1
FROM python:3.15-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_EDITABLE=1

# Dependencies first, so they cache separately from source changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:3.15-slim

RUN useradd --create-home --shell /bin/bash app
WORKDIR /app

# Non-editable install: the package lives inside .venv, so src/ isn't needed at runtime.
COPY --from=builder --chown=app:app /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"
USER app

CMD ["feelings"]
