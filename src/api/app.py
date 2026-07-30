"""FastAPI application: loads the model once at startup, then serves."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.middleware import ApiKeyMiddleware, RateLimitMiddleware, RequestContextMiddleware
from src.api.routes import router
from src.common.logger import configure_logging, get_logger
from src.common.settings import get_api_config, get_settings

logger = get_logger(__name__)

DESCRIPTION = """
Claim decisioning with audience-specific explanations.

**The ML model decides. The LLM explains. No code path lets the LLM alter a decision.**

- `POST /predict` — score and routing. No LLM, no external dependency.
- `POST /explain` — the same decision rendered for one of three audiences.

The system never issues a decline on its own. At any score, an adverse outcome requires a
human reviewer.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the pipeline before the first request and hold it for the process lifetime."""
    configure_logging()
    # Loaded once: the model, the fitted transformers and the SHAP explainer together cost
    # ~0.6s, which would otherwise be paid on the first claim of every worker.
    from src.genai.explanation_pipeline import ExplanationPipeline

    try:
        app.state.pipeline = ExplanationPipeline()
        logger.info("api ready", extra=app.state.pipeline.health())
    except Exception as exc:
        # Startup fails loudly rather than serving 503s: a container that cannot decide
        # claims should never pass a health check.
        logger.exception("pipeline failed to load")
        raise RuntimeError(
            f"Cannot start without a registered model and fitted transformers: {exc}"
        ) from exc

    yield
    app.state.pipeline = None


def create_app() -> FastAPI:
    """Build the application with its middleware stack and routes."""
    cfg = get_api_config()
    service = cfg["service"]

    app = FastAPI(
        title=service["title"],
        version=service["version"],
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Starlette runs these in reverse order of registration, so the request id is bound
    # first and every 401 or 429 below it carries a correlation id.
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(ApiKeyMiddleware)
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=service["cors_origins"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Return the 422 with the request id, and log which field was rejected."""
        # Pydantic attaches the whole submitted body to every error under "input", which
        # includes the customer narrative. Echoing that back would publish PII in an HTTP
        # response and in any client that logs responses.
        detail = [
            {k: v for k, v in error.items() if k not in ("input", "ctx")} for error in exc.errors()
        ]
        fields = [".".join(str(p) for p in e["loc"][1:]) for e in exc.errors()]
        logger.warning("request rejected", extra={"path": request.url.path, "fields": fields})
        return JSONResponse(
            status_code=422,
            content={
                "detail": detail,
                "rejected_fields": fields,
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        """Return a generic 500 without leaking internals to the caller."""
        # The message stays in the logs, correlated by request id. A stack trace in an API
        # response tells an attacker about the stack and a customer nothing.
        logger.exception("unhandled error", extra={"path": request.url.path})
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal error. Quote the request id when reporting this.",
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, Any]:
        return {
            "service": service["title"],
            "version": service["version"],
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    settings = get_settings()
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
