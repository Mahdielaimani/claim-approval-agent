"""Request correlation, optional API-key auth, and per-endpoint rate limiting."""

from __future__ import annotations

import secrets
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from src.common.logger import bind_request_id, get_logger
from src.common.settings import get_api_config, get_settings

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"

Handler = Callable[[Request], Awaitable[Response]]


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Binds a request id, times the request, and writes one access log line."""

    async def dispatch(self, request: Request, call_next: Handler) -> Response:
        # Honoured from the caller when present: a claim traced across services keeps one id.
        incoming = request.headers.get(REQUEST_ID_HEADER)
        request_id = bind_request_id(incoming)
        request.state.request_id = request_id

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "request failed",
                extra={"method": request.method, "path": request.url.path},
            )
            raise

        duration_ms = int((time.perf_counter() - started) * 1000)
        response.headers[REQUEST_ID_HEADER] = request_id
        logger.info(
            "request completed",
            extra={
                "method": request.method,
                # Path only, never the query string: a claim id in a URL would be logged.
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Rejects requests without a valid API key, when one is configured."""

    def __init__(
        self,
        app: Any,
        config: dict[str, Any] | None = None,
        api_key: str | None = None,
    ) -> None:
        super().__init__(app)
        cfg = (config or get_api_config())["middleware"]["auth"]
        self.header = cfg["header"]
        self.exempt = tuple(cfg["exempt_paths"])
        # Injectable so both auth paths are testable; production reads API_KEY from env.
        self.api_key = get_settings().api_key if api_key is None else api_key

    async def dispatch(self, request: Request, call_next: Handler) -> Response:
        # No key means auth off, keeping the demo zero-setup; production sets API_KEY.
        if not self.api_key or request.url.path.startswith(self.exempt):
            return await call_next(request)

        presented = request.headers.get(self.header, "")
        # Constant-time: a plain == leaks key length and prefix through response timing.
        if not secrets.compare_digest(presented, self.api_key):
            logger.warning("unauthorised request", extra={"path": request.url.path})
            return JSONResponse(
                status_code=401,
                content={
                    "detail": f"Missing or invalid {self.header} header.",
                    "request_id": getattr(request.state, "request_id", None),
                },
            )
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Caps requests per client per minute, with a separate budget per endpoint.

    In-memory and therefore per-process: N replicas allow N times the limit. Adequate for a
    single container, and the reason the AWS design puts throttling at API Gateway.
    """

    def __init__(self, app: Any, config: dict[str, Any] | None = None) -> None:
        super().__init__(app)
        limits = (config or get_api_config())["middleware"]["rate_limit"]
        # Separate budgets: /predict is in-process, /explain spends money at a third party.
        self.limits = {
            "/api/v1/predict": int(limits["predict_per_minute"]),
            "/api/v1/explain": int(limits["explain_per_minute"]),
        }
        self._hits: dict[tuple[str, str], deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next: Handler) -> Response:
        limit = self.limits.get(request.url.path)
        if limit is None:
            return await call_next(request)

        client = request.client.host if request.client else "unknown"
        window = self._hits[(client, request.url.path)]
        now = time.monotonic()
        while window and now - window[0] > 60.0:
            window.popleft()

        if len(window) >= limit:
            retry_after = int(61.0 - (now - window[0]))
            logger.warning(
                "rate limit exceeded",
                extra={"path": request.url.path, "limit": limit, "window_seconds": 60},
            )
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(max(retry_after, 1))},
                content={
                    "detail": f"Rate limit of {limit} requests per minute exceeded.",
                    "request_id": getattr(request.state, "request_id", None),
                },
            )

        window.append(now)
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(max(limit - len(window), 0))
        return response
