"""Tests for drift scoring and LLM operational monitoring."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import ALARM, STABLE, DriftMonitor
from src.monitoring.llm_monitor import BREACH, OK, CallRecord, LLMMonitor

RNG = np.random.default_rng(20260730)


@pytest.fixture
def reference_frame() -> pd.DataFrame:
    """Eight hundred rows shaped like the feature matrix: continuous, encoded, binary."""
    n = 800
    return pd.DataFrame(
        {
            "rrp": RNG.normal(1400, 250, n),
            "fee_to_rrp_ratio": RNG.beta(2, 20, n),
            "country": RNG.choice([0, 1, 2], n, p=[0.5, 0.3, 0.2]),
            "touchScreen": RNG.choice([0.0, 1.0], n, p=[0.4, 0.6]),
            "issue_desc_length": RNG.integers(20, 400, n),
        }
    )


@pytest.fixture
def monitor(reference_frame) -> DriftMonitor:
    return DriftMonitor.from_training_data(reference_frame)


class TestDriftBinning:
    def test_low_cardinality_columns_become_categorical(self, monitor):
        """Quantile edges on an encoded categorical would name nothing."""
        assert monitor.reference["features"]["country"]["kind"] == "categorical"
        assert monitor.reference["features"]["touchScreen"]["kind"] == "categorical"

    def test_continuous_columns_become_quantile_bins(self, monitor):
        assert monitor.reference["features"]["rrp"]["kind"] == "numeric"
        assert len(monitor.reference["features"]["rrp"]["shares"]) <= 10

    def test_reference_shares_sum_to_one(self, monitor):
        for spec in monitor.reference["features"].values():
            assert sum(spec["shares"]) == pytest.approx(1.0, abs=1e-9)

    def test_outer_edges_are_open(self, monitor):
        """A live value beyond anything seen in training must still land in a bin."""
        edges = monitor.reference["features"]["rrp"]["edges"]
        assert edges[0] == -np.inf and edges[-1] == np.inf


class TestDriftScoring:
    def test_reference_against_itself_is_zero(self, monitor, reference_frame):
        report = monitor.compare(reference_frame)
        assert all(f.psi == pytest.approx(0.0, abs=1e-9) for f in report.features)
        assert not report.requires_investigation

    def test_a_resampled_batch_stays_stable(self, monitor, reference_frame):
        """Sampling noise alone must not raise an alarm, or nobody will read the alarms."""
        batch = reference_frame.sample(400, random_state=7)
        report = monitor.compare(batch)
        assert not report.alarms

    def test_a_shifted_continuous_feature_alarms(self, monitor, reference_frame):
        batch = reference_frame.copy()
        batch["rrp"] = batch["rrp"] * 1.6
        report = monitor.compare(batch)
        assert report.requires_investigation
        assert next(f for f in report.features if f.feature == "rrp").status == ALARM

    def test_an_untouched_feature_stays_stable_beside_a_drifting_one(
        self, monitor, reference_frame
    ):
        batch = reference_frame.copy()
        batch["rrp"] = batch["rrp"] * 1.6
        report = monitor.compare(batch)
        assert next(f for f in report.features if f.feature == "country").status == STABLE

    def test_a_dropped_category_alarms(self, monitor, reference_frame):
        batch = reference_frame[reference_frame["country"] == 0]
        report = monitor.compare(batch)
        assert next(f for f in report.features if f.feature == "country").status == ALARM

    def test_an_unseen_category_inflates_psi(self, monitor, reference_frame):
        """A level absent from training falls in no bin, and that must show up as drift."""
        batch = reference_frame.copy()
        batch.loc[batch.index[:400], "country"] = 9
        assert next(f for f in monitor.compare(batch).features if f.feature == "country").psi > 0.2

    def test_psi_is_always_finite(self, monitor, reference_frame):
        """An empty bin makes the log term infinite unless the floor is applied."""
        # Above the median, so the lower bins are empty while the batch stays above the
        # minimum row count — filtering harder would skip the comparison and assert nothing.
        batch = reference_frame[reference_frame["rrp"] > reference_frame["rrp"].median()]
        report = monitor.compare(batch)
        assert report.skipped is None
        assert report.features
        for feature in report.features:
            assert np.isfinite(feature.psi)

    def test_the_largest_shift_names_a_real_bin(self, monitor, reference_frame):
        batch = reference_frame.copy()
        batch["rrp"] = batch["rrp"] * 1.6
        drift = next(f for f in monitor.compare(batch).features if f.feature == "rrp")
        assert drift.largest_shift_bin in monitor.reference["features"]["rrp"]["labels"]


class TestDriftGuards:
    def test_a_small_batch_is_refused_not_scored(self, monitor, reference_frame):
        """PSI on a tiny batch produces large values from nothing."""
        report = monitor.compare(reference_frame.head(50))
        assert report.skipped is not None
        assert report.features == []
        assert not report.requires_investigation

    def test_ignored_features_are_not_scored(self, reference_frame):
        config = {
            "drift": {
                "thresholds": {"watch": 0.1, "alarm": 0.2},
                "n_bins": 10,
                "min_rows": 200,
                "ignore": ["issue_desc_length"],
                "monitor_prediction_distribution": True,
            }
        }
        monitor = DriftMonitor.from_training_data(reference_frame, config=config)
        scored = {f.feature for f in monitor.compare(reference_frame).features}
        assert "issue_desc_length" not in scored
        assert "rrp" in scored

    def test_a_missing_column_is_skipped_rather_than_crashing(self, monitor, reference_frame):
        batch = reference_frame.drop(columns=["touchScreen"])
        scored = {f.feature for f in monitor.compare(batch).features}
        assert "touchScreen" not in scored and "rrp" in scored


class TestPredictionDrift:
    def test_scores_are_monitored_separately_from_features(self, reference_frame):
        scores = RNG.beta(2, 5, len(reference_frame))
        monitor = DriftMonitor.from_training_data(reference_frame, scores)
        report = monitor.compare(reference_frame, scores)
        assert report.prediction_drift is not None
        assert report.prediction_drift.feature == "decline_risk_score"
        assert report.prediction_drift.psi == pytest.approx(0.0, abs=1e-9)

    def test_a_shifted_score_distribution_forces_investigation(self, reference_frame):
        """Every feature can stay in range while the model's output moves."""
        scores = RNG.beta(2, 5, len(reference_frame))
        monitor = DriftMonitor.from_training_data(reference_frame, scores)
        report = monitor.compare(reference_frame, np.clip(scores + 0.35, 0, 1))
        assert report.prediction_drift.status == ALARM
        assert not report.alarms
        assert report.requires_investigation


class TestReferencePersistence:
    def test_round_trip_preserves_scores(self, monitor, reference_frame, tmp_path):
        batch = reference_frame.copy()
        batch["rrp"] = batch["rrp"] * 1.4
        before = {f.feature: f.psi for f in monitor.compare(batch).features}

        monitor.save(tmp_path)
        after = {f.feature: f.psi for f in DriftMonitor.load(tmp_path).compare(batch).features}
        assert before == after

    def test_the_reference_is_readable_json(self, monitor, tmp_path):
        """Diffable in review: a binary reference cannot be inspected when it disagrees."""
        path = monitor.save(tmp_path)
        assert json.loads(path.read_text(encoding="utf-8"))["n_rows"] == 800


def call(**overrides) -> CallRecord:
    base = {
        "timestamp": datetime.now(UTC).isoformat(),
        "persona": "customer",
        "provider": "groq",
        "model": "groq/llama-3.3-70b-versatile",
        "latency_ms": 1400,
        "prompt_tokens": 900,
        "completion_tokens": 180,
        "cost_usd": 0.0004,
    }
    return CallRecord(**{**base, **overrides})


@pytest.fixture
def ledger(tmp_path) -> LLMMonitor:
    return LLMMonitor(ledger=tmp_path / "llm_calls.jsonl")


def fill(monitor: LLMMonitor, n: int, **overrides) -> None:
    for _ in range(n):
        monitor.record(call(**overrides))


class TestLedgerPrivacy:
    def test_the_ledger_records_no_free_text(self, ledger):
        """Anyone debugging a cost reads this file; a claim narrative must not be in it."""
        ledger.record(call())
        row = json.loads(Path(ledger.ledger).read_text(encoding="utf-8").strip())
        assert not [v for v in row.values() if isinstance(v, str) and len(v) > 40]

    def test_the_ledger_has_no_content_field(self, ledger):
        ledger.record(call())
        row = json.loads(Path(ledger.ledger).read_text(encoding="utf-8").strip())
        for forbidden in ("prompt", "response", "content", "explanation", "issue_desc", "summary"):
            assert forbidden not in row


class TestLedgerReading:
    def test_an_empty_ledger_is_healthy_but_not_confident(self, ledger):
        health = ledger.snapshot()
        assert health.n_calls == 0 and health.healthy
        assert not health.latency_confident

    def test_a_malformed_line_does_not_blind_the_dashboard(self, ledger):
        fill(ledger, 3)
        with Path(ledger.ledger).open("a", encoding="utf-8") as fh:
            fh.write("{not json\n")
        assert len(ledger.load()) == 3

    def test_records_outside_the_window_are_excluded(self, ledger):
        fill(ledger, 2)
        ledger.record(call(timestamp=(datetime.now(UTC) - timedelta(days=45)).isoformat()))
        assert len(ledger.load(window_hours=720)) == 2
        assert len(ledger.load(window_hours=None)) == 3


class TestThresholds:
    def test_healthy_traffic_breaches_nothing(self, ledger):
        fill(ledger, 25)
        assert ledger.snapshot().healthy

    def test_a_rising_fallback_rate_breaches(self, ledger):
        fill(ledger, 8)
        fill(ledger, 2, provider="openai", used_fallback_provider=True)
        breached = {t.name for t in ledger.snapshot().breaches}
        assert "fallback_rate" in breached

    def test_one_template_fallback_in_fifty_breaches(self, ledger):
        """No LLM answered at all: one occurrence is an incident, not a rate to tolerate."""
        fill(ledger, 49)
        fill(ledger, 1, used_template_fallback=True, provider="none")
        breached = {t.name for t in ledger.snapshot().breaches}
        assert "template_fallback_rate" in breached

    def test_spend_alerts_before_the_budget_is_gone(self, ledger):
        """An alarm that fires once the money is spent is a report, not an alert."""
        fill(ledger, 60, cost_usd=0.30)
        spend = next(t for t in ledger.snapshot().thresholds if t.name == "monthly_spend_usd")
        assert spend.status == BREACH
        assert spend.limit < ledger.budget_usd

    def test_latency_cannot_breach_on_too_few_calls(self, ledger):
        """One slow sample is not a p95, and must not page anyone."""
        fill(ledger, 5, latency_ms=30_000)
        health = ledger.snapshot()
        assert not health.latency_confident
        assert next(t for t in health.thresholds if t.name == "latency_p95_ms").status == OK

    def test_latency_breaches_once_there_are_enough_calls(self, ledger):
        fill(ledger, 25, latency_ms=30_000)
        health = ledger.snapshot()
        assert health.latency_confident
        assert next(t for t in health.thresholds if t.name == "latency_p95_ms").status == BREACH


class TestMockExclusion:
    def test_mock_calls_are_counted_but_not_billed(self, ledger):
        fill(ledger, 10, is_mock=True, cost_usd=99.0, latency_ms=1)
        health = ledger.snapshot()
        assert health.n_calls == 10
        assert health.total_cost_usd == 0.0
        assert health.total_tokens == 0


class TestRecordFromPipeline:
    def test_an_explanation_is_reduced_to_operational_facts(self, ledger):
        class FakeExplanation:
            persona = "adjuster"
            provider = "groq_reserve_2"
            model = "groq/llama-3.3-70b-versatile"
            llm_latency_ms = 2100
            prompt_tokens = 1143
            completion_tokens = 196
            cost_usd = 0.000829
            used_fallback_provider = True
            used_template_fallback = False
            retried = True
            validation = {"passed": True}

        record = ledger.record_explanation(FakeExplanation())
        assert record.persona == "adjuster"
        assert record.total_tokens == 1339
        assert record.used_fallback_provider and record.retried
        assert len(ledger.load()) == 1

    def test_a_template_fallback_records_no_provider(self, ledger):
        class Degraded:
            persona = "customer"
            used_template_fallback = True

        record = ledger.record_explanation(Degraded())
        assert record.provider == "none"
        assert record.used_template_fallback
