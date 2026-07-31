# Multi-stage: the build toolchain compiles the wheels, the runtime image never carries it.
# lightgbm and shap both build native extensions, so a single-stage image would ship gcc.

FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# A virtualenv rather than --user: one directory to copy, and the runtime stage inherits
# nothing else from this layer.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copied alone so the dependency layer survives every source edit. Rebuilding scikit-learn
# and lightgbm because a route changed costs minutes on each push.
COPY requirements.txt .
RUN pip install --requirement requirements.txt


FROM python:3.12-slim AS runtime

# libgomp is the one build artefact still needed at run time: lightgbm and xgboost link
# against OpenMP. Without it the import fails, not the build.
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

# Created here and owned by appuser. WORKDIR creates /app as root, so the LLM ledger's own
# mkdir would raise PermissionError on the first /explain — a 500 in production only.
RUN mkdir -p /app/artifacts/models /app/artifacts/reports \
    && chown -R appuser:appuser /app/artifacts

# Model and fitted transformers are mounted, not baked in. Baking them would tie the image
# tag to a model version, so promoting a model would require rebuilding the application.
VOLUME ["/app/artifacts/models"]

USER appuser
EXPOSE 8000

# Reports 503 until the model loads, which is the intended answer: a container that cannot
# decide a claim must not receive traffic.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail --silent http://localhost:8000/api/v1/health || exit 1

# Exec form, so uvicorn is PID 1 and receives SIGTERM directly. Under the shell form it
# would be a child of sh and the orchestrator's stop signal would never reach it.
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
