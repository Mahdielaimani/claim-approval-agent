"""Operational health of the LLM layer: spend, provider fallback, latency, degradation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from src.common.logger import get_logger
from src.common.settings import get_monitoring_config, get_settings

logger = get_logger(__name__)

LEDGER_FILE = "llm_calls.jsonl"

OK = "ok"
WARN = "warn"
BREACH = "breach"

# A percentile over a handful of calls is one sample wearing a statistic's name. Below this
# the snapshot still reports latency, flagged, rather than presenting noise as a measurement.
MIN_CALLS_FOR_PERCENTILE = 20


@dataclass
class CallRecord:
    """One LLM call, reduced to the facts needed to bill and diagnose it."""

    # Deliberately no prompt and no response: this file is read by anyone debugging cost,
    # and a claim narrative must not be recoverable from an operations artefact.
    timestamp: str
    persona: str
    provider: str
    model: str
    latency_ms: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    used_fallback_provider: bool = False
    used_template_fallback: bool = False
    retried: bool = False
    validation_passed: bool = True
    is_mock: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class Threshold:
    """One monitored quantity against its configured limit."""

    name: str
    value: float
    limit: float
    status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": round(self.value, 4),
            "limit": self.limit,
            "status": self.status,
        }


@dataclass
class LLMHealth:
    """Aggregate over a window of calls, with each threshold resolved."""

    n_calls: int
    window_hours: int | None
    total_cost_usd: float
    total_tokens: int
    latency_p50_ms: float
    latency_p95_ms: float
    latency_confident: bool
    by_provider: dict[str, int] = field(default_factory=dict)
    thresholds: list[Threshold] = field(default_factory=list)

    @property
    def breaches(self) -> list[Threshold]:
        return [t for t in self.thresholds if t.status == BREACH]

    @property
    def healthy(self) -> bool:
        return not self.breaches

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_calls": self.n_calls,
            "window_hours": self.window_hours,
            "healthy": self.healthy,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "total_tokens": self.total_tokens,
            "latency_p50_ms": round(self.latency_p50_ms, 1),
            "latency_p95_ms": round(self.latency_p95_ms, 1),
            "latency_confident": self.latency_confident,
            "by_provider": self.by_provider,
            "thresholds": [t.as_dict() for t in self.thresholds],
        }


class LLMMonitor:
    """Appends call records to a ledger and scores them against configured limits."""

    def __init__(self, ledger: Path | None = None, config: dict[str, Any] | None = None) -> None:
        cfg = (config or get_monitoring_config())["llm"]
        self.budget_usd = float(cfg["monthly_budget_usd"])
        self.alert_fraction = float(cfg["alert_at_budget_fraction"])
        self.fallback_alarm = float(cfg["fallback_rate_alarm"])
        self.template_alarm = float(cfg["template_fallback_alarm"])
        self.latency_p95_limit = float(cfg["latency_p95_ms"])
        self.ledger = Path(ledger) if ledger else Path("artifacts/reports") / LEDGER_FILE

    def record(self, call: CallRecord) -> None:
        """Append one call to the ledger."""
        # A file, not memory: the API runs multiple workers and restarts, and a counter that
        # resets on deploy cannot answer "what did we spend this month".
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(call)) + "\n")

    def record_explanation(self, explanation: Any, persona: str | None = None) -> CallRecord:
        """Record one pipeline result without importing the pipeline."""
        call = CallRecord(
            timestamp=datetime.now(UTC).isoformat(),
            persona=str(persona or getattr(explanation, "persona", "") or "unknown"),
            provider=getattr(explanation, "provider", "") or "none",
            model=getattr(explanation, "model", ""),
            latency_ms=int(getattr(explanation, "llm_latency_ms", 0) or 0),
            prompt_tokens=int(getattr(explanation, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(explanation, "completion_tokens", 0) or 0),
            cost_usd=float(getattr(explanation, "cost_usd", 0.0) or 0.0),
            used_fallback_provider=bool(getattr(explanation, "used_fallback_provider", False)),
            used_template_fallback=bool(getattr(explanation, "used_template_fallback", False)),
            retried=bool(getattr(explanation, "retried", False)),
            validation_passed=bool((getattr(explanation, "validation", {}) or {}).get("passed")),
            is_mock=get_settings().llm_mock_mode,
        )
        self.record(call)
        return call

    def load(self, window_hours: int | None = None) -> list[CallRecord]:
        """Read the ledger, optionally restricted to a recent window."""
        if not self.ledger.exists():
            return []
        cutoff = datetime.now(UTC) - timedelta(hours=window_hours) if window_hours else None

        records = []
        for line in self.ledger.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                record = CallRecord(**data)
            except (json.JSONDecodeError, TypeError) as exc:
                # One malformed line must not blind the whole dashboard.
                logger.warning("skipping unreadable ledger line", extra={"error": str(exc)[:120]})
                continue
            if cutoff and datetime.fromisoformat(record.timestamp) < cutoff:
                continue
            records.append(record)
        return records

    def snapshot(self, window_hours: int | None = 720) -> LLMHealth:
        """Aggregate the window and resolve every threshold. Defaults to 30 days."""
        records = self.load(window_hours)
        if not records:
            return LLMHealth(0, window_hours, 0.0, 0, 0.0, 0.0, False)

        # Mock calls cost nothing and answer instantly, so mixing them in would flatter
        # every metric. Counted, then excluded from the ones they would distort.
        live = [r for r in records if not r.is_mock]
        latencies = sorted(r.latency_ms for r in live) or [0]
        cost = sum(r.cost_usd for r in live)

        health = LLMHealth(
            n_calls=len(records),
            window_hours=window_hours,
            total_cost_usd=cost,
            total_tokens=sum(r.total_tokens for r in live),
            latency_p50_ms=_percentile(latencies, 0.50),
            latency_p95_ms=_percentile(latencies, 0.95),
            latency_confident=len(live) >= MIN_CALLS_FOR_PERCENTILE,
            by_provider=_counts(r.provider for r in records),
            thresholds=self._thresholds(records, live, cost, _percentile(latencies, 0.95)),
        )
        logger.info("llm health snapshot", extra=health.as_dict())
        return health

    def _thresholds(
        self, records: list[CallRecord], live: list[CallRecord], cost: float, p95: float
    ) -> list[Threshold]:
        n = max(len(records), 1)
        fallback_rate = sum(r.used_fallback_provider for r in records) / n
        template_rate = sum(r.used_template_fallback for r in records) / n
        alert_at = self.budget_usd * self.alert_fraction

        return [
            # Alerting at 80% of budget rather than 100%: an alarm that fires when the money
            # is already spent is a report, not an alert.
            Threshold("monthly_spend_usd", cost, alert_at, _status(cost, alert_at)),
            Threshold(
                "fallback_rate",
                fallback_rate,
                self.fallback_alarm,
                _status(fallback_rate, self.fallback_alarm),
            ),
            # No LLM answered at all. One occurrence is an incident, hence a 1% limit.
            Threshold(
                "template_fallback_rate",
                template_rate,
                self.template_alarm,
                _status(template_rate, self.template_alarm),
            ),
            Threshold(
                "latency_p95_ms",
                p95,
                self.latency_p95_limit,
                _status(p95, self.latency_p95_limit)
                if len(live) >= MIN_CALLS_FOR_PERCENTILE
                else OK,
            ),
        ]


def _status(value: float, limit: float) -> str:
    if value >= limit:
        return BREACH
    return WARN if value >= limit * 0.8 else OK


def _percentile(ordered: list[int], q: float) -> float:
    """Nearest-rank percentile over an already sorted list."""
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return float(ordered[index])


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts
