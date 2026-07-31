# Multi-stage: lightgbm and shap build native extensions, so a single-stage image
# would ship gcc into production.

FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# A virtualenv rather than --user: one directory to copy into the runtime stage.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copied alone so the dependency layer survives every source edit.
COPY requirements.txt .
RUN pip install --requirement requirements.txt


FROM python:3.12-slim AS runtime

# lightgbm and xgboost link against OpenMP. Without libgomp the import fails, not the build.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser configs/ ./configs/
COPY --chown=appuser:appuser prompts/ ./prompts/

# WORKDIR creates /app as root, so the ledger's own mkdir would raise PermissionError on
# the first /explain — a 500 visible only in production.
RUN mkdir -p /app/artifacts/models /app/artifacts/reports \
    && chown -R appuser:appuser /app/artifacts

# Mounted, not baked in: baking would tie the image tag to a model version.
VOLUME ["/app/artifacts/models"]

USER appuser
EXPOSE 8000

# Reports 503 until the model loads — a container that cannot decide a claim must not
# receive traffic.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail --silent http://localhost:8000/api/v1/health || exit 1

# Exec form, so uvicorn is PID 1 and receives SIGTERM directly.
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
