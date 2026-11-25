# Build stage: resolve dependencies from the lock file into a virtualenv.
FROM python:3.12-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN pip install --no-cache-dir "uv==0.12.6"

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY schemas ./schemas
RUN uv sync --frozen --no-dev --no-editable

# Runtime stage: only the virtualenv, running as an unprivileged user.
FROM python:3.12-slim

RUN useradd --create-home --uid 10001 app
COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    CTGOV_CACHE_DIR=/home/app/.cache/ctgov-extract

USER app
WORKDIR /home/app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"

CMD ["ctgov-extract", "serve", "--host", "0.0.0.0", "--port", "8000"]
