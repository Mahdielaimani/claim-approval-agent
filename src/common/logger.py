"""Structured JSON logging with PII redaction and per-request correlation."""

from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
import uuid
from datetime import UTC, datetime
from typing import Any

# Redacted centrally because a per-call-site rule is one someone eventually forgets.
_PII_FIELDS = frozenset(
    {"issuedesc", "issue_desc", "other", "customer_name", "email", "phone", "imei"}
)

_STANDARD_RECORD_ATTRS = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
        "asctime",
        "message",
    }
)

_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)


def bind_request_id(request_id: str | None = None) -> str:
    """Attach a request id to the current context and return it."""
    rid = request_id or uuid.uuid4().hex[:16]
    _request_id.set(rid)
    return rid


def get_request_id() -> str | None:
    """Request id bound to the current context, if any."""
    return _request_id.get()


def _redact(value: Any, key: str | None = None) -> Any:
    if key is not None and key.lower() in _PII_FIELDS:
        return "[REDACTED]"
    if isinstance(value, dict):
        return {k: _redact(v, k) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_redact(v) for v in value]
    return value


class JsonFormatter(logging.Formatter):
    """Renders a LogRecord as one line of JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        rid = _request_id.get()
        if rid:
            payload["request_id"] = rid

        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS and not key.startswith("_"):
                payload[key] = _redact(value, key)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str | None = None) -> None:
    """Install the JSON handler on the root logger."""
    resolved = (level or os.getenv("LOG_LEVEL", "INFO")).upper()

    root = logging.getLogger()
    root.setLevel(resolved)

    # Replaced, not appended: uvicorn reloads would otherwise duplicate every line.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)

    for noisy in ("httpx", "httpcore", "LiteLLM", "urllib3", "matplotlib", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Module logger, configuring the root handler on first use."""
    if not logging.getLogger().handlers:
        configure_logging()
    return logging.getLogger(name)
