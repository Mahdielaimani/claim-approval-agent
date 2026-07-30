"""Tests for the HTTP contract."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.middleware import REQUEST_ID_HEADER
from src.monitoring.llm_monitor import LLMMonitor

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "models"

needs_model = pytest.mark.skipif(
    not (ARTIFACTS / "model.joblib").exists() or not (ARTIFACTS / "transformers.joblib").exists(),
    reason="no registered model; train and register one first",
)


def example(name: str) -> dict[str, Any]:
    """Load one of the committed demo claims."""
    return json.loads((EXAMPLES / f"{name}.json").read_text(encoding="utf-8"))


class StubPrediction:
    def __init__(self, score: float, review: bool, decision: str, flags: list[str]) -> None:
        self._d = {
            "claim_id": "CLM-TEST",
            "request_id": "req-test",
            "decline_risk_score": score,
            "decision": decision,
            "requires_human_review": review,
            "human_review_reason": "borderline_confidence" if review else None,
            "risk_flags": flags,
            "decision_threshold": 0.344,
            "model_version": "1",
            "model_algorithm": "random_forest",
            "latency_ms": 42,
        }

    def as_dict(self) -> dict[str, Any]:
        return dict(self._d)


class StubExplanation:
    def __init__(self, persona: str, prediction: StubPrediction) -> None:
        self.persona = persona
        self.prediction = prediction

    def as_dict(self) -> dict[str, Any]:
        if self.persona == "customer":
            return {
                "claim_id": "CLM-TEST",
                "request_id": "req-test",
                "persona": "customer",
                "status": self.prediction.as_dict()["decision"],
                "under_review": self.prediction.as_dict()["requires_human_review"],
                "explanation": {
                    "summary": "Your claim is being reviewed.",
                    "key_factors": ["The condition of your screen."],
                    "next_steps": ["No action needed."],
                },
                "factors": [
                    {
                        "factor": "the condition of your screen",
                        "direction": "increases decline risk",
                    }
                ],
                "total_latency_ms": 1200,
            }
        return {
            **self.prediction.as_dict(),
            "persona": self.persona,
            "explanation": {
                "summary": "Routed for review.",
                "key_factors": ["fee_to_rrp_ratio -0.04"],
                "risk_assessment": "No triggers.",
                "review_notes": "Verify the serial.",
            },
            "factors": [
                {
                    "feature": "fee_to_rrp_ratio",
                    "label": "Excess share",
                    "value": 0.04,
                    "shap_value": -0.04,
                    "direction": "reduces decline risk",
                    "unit": "",
                }
            ],
            "generation": {
                "provider": "groq",
                "model": "groq/llama-3.3-70b-versatile",
                "prompt_version": "v1.0.0",
                "used_fallback_provider": False,
                "used_template_fallback": False,
                "retried": False,
                "llm_latency_ms": 1100,
            },
            "validation": {"passed": True, "n_errors": 0, "n_warnings": 0, "failed_checks": []},
            "total_latency_ms": 1200,
        }


class StubPipeline:
    """Stands in for the real pipeline so the HTTP contract is testable without a model."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], str | None]] = []

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model_version": "1",
            "model_algorithm": "random_forest",
            "decision_threshold": 0.344,
            "n_features": 37,
            "prompt_version": "v1.0.0",
            "llm_mode": "mock",
            "llm_chain": ["groq/llama-3.3-70b-versatile"],
            "trained_at": "2026-07-29T17:47:14+00:00",
            "git_commit": "abc1234",
        }

    def predict(self, claim: dict[str, Any], *, claim_id: str | None = None) -> StubPrediction:
        self.calls.append(("predict", claim, claim_id))
        return StubPrediction(0.3015, True, "under review", [])

    def explain(
        self, claim: dict[str, Any], persona: str = "customer", *, claim_id: str | None = None
    ) -> StubExplanation:
        self.calls.append(("explain", claim, claim_id))
        return StubExplanation(persona, self.predict(claim, claim_id=claim_id))


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """App with a stubbed pipeline. Lifespan is skipped, so no model is loaded."""
    app = create_app()
    app.state.pipeline = StubPipeline()
    # Its own ledger per test: without this the suite appends to the real operations
    # artefact, and the monthly spend figure would include fabricated calls.
    app.state.llm_monitor = LLMMonitor(ledger=tmp_path / "llm_calls.jsonl")
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def claim() -> dict[str, Any]:
    return example("claim_borderline")


class TestHealth:
    def test_reports_what_is_loaded(self, client):
        body = client.get("/api/v1/health").json()
        assert body["status"] == "ok"
        assert body["model_algorithm"] == "random_forest"
        assert body["n_features"] == 37

    def test_needs_no_auth(self, client):
        """A health check that requires a key is unusable by a load balancer."""
        assert client.get("/api/v1/health").status_code == 200

    def test_returns_503_without_a_model(self):
        """A missing model is a 503 with an actionable message, not a stack trace."""
        app = create_app()
        response = TestClient(app, raise_server_exceptions=False).get("/api/v1/health")
        assert response.status_code == 503
        assert "Train and register" in response.json()["detail"]


class TestPredict:
    def test_returns_score_and_routing(self, client, claim):
        body = client.post("/api/v1/predict", json=claim).json()
        assert 0.0 <= body["decline_risk_score"] <= 1.0
        assert isinstance(body["requires_human_review"], bool)
        assert body["decision_threshold"] == 0.344

    def test_rejects_a_missing_required_field(self, client):
        response = client.post("/api/v1/predict", json=example("claim_invalid_missing_rrp"))
        assert response.status_code == 422
        assert response.json()["rejected_fields"] == ["rrp"]

    def test_rejects_a_claim_type_never_trained_on(self, client):
        """Scoring an unseen claim type would produce a number with no basis."""
        response = client.post("/api/v1/predict", json=example("claim_invalid_claimtype"))
        assert response.status_code == 422
        assert response.json()["rejected_fields"] == ["claimType"]

    def test_rejects_an_unknown_field(self, client, claim):
        """A misspelled field must be reported, not silently ignored."""
        response = client.post("/api/v1/predict", json={**claim, "fraudScore": 0.9})
        assert response.status_code == 422

    @pytest.mark.parametrize(
        "override,field",
        [
            ({"rrp": -1.0}, "rrp"),
            ({"rrp": 0.0}, "rrp"),
            ({"country": "DE"}, "country"),
            ({"policyStatus": "Suspended"}, "policyStatus"),
            ({"touchScreen": 9.0}, "touchScreen"),
            ({"issueDesc": ""}, "issueDesc"),
        ],
    )
    def test_field_constraints(self, client, claim, override, field):
        response = client.post("/api/v1/predict", json={**claim, **override})
        assert response.status_code == 422
        assert field in response.json()["rejected_fields"]

    def test_422_does_not_echo_the_customer_narrative(self, client):
        """Pydantic attaches the whole body to each error; that would publish PII."""
        response = client.post("/api/v1/predict", json=example("claim_invalid_missing_rrp"))
        assert "toestel" not in response.text.lower()
        assert "input" not in response.json()["detail"][0]

    def test_422_carries_the_request_id(self, client):
        body = client.post("/api/v1/predict", json=example("claim_invalid_missing_rrp")).json()
        assert body["request_id"]


class TestExplain:
    def test_customer_response_omits_every_internal(self, client, claim):
        """The persona filter must hold at the envelope, not only in the prompt."""
        body = client.post("/api/v1/explain", json={**claim, "persona": "customer"}).json()
        for forbidden in (
            "decline_risk_score",
            "decision_threshold",
            "model_version",
            "model_algorithm",
            "risk_flags",
            "generation",
            "validation",
        ):
            assert forbidden not in body, f"customer response leaked {forbidden}"

    def test_customer_schema(self, client, claim):
        body = client.post("/api/v1/explain", json={**claim, "persona": "customer"}).json()
        assert set(body["explanation"]) == {"summary", "key_factors", "next_steps"}

    @pytest.mark.parametrize("persona", ["adjuster", "auditor"])
    def test_technical_personas_receive_attribution_and_provenance(self, client, claim, persona):
        body = client.post("/api/v1/explain", json={**claim, "persona": persona}).json()
        assert body["decline_risk_score"] > 0
        assert body["generation"]["prompt_version"]
        assert body["validation"]["passed"] is True

    def test_persona_defaults_to_customer(self, client, claim):
        body = client.post("/api/v1/explain", json=claim).json()
        assert body["persona"] == "customer"

    def test_rejects_an_unknown_persona(self, client, claim):
        response = client.post("/api/v1/explain", json={**claim, "persona": "manager"})
        assert response.status_code == 422

    def test_persona_never_reaches_the_feature_matrix(self, client, claim):
        """persona is routing metadata; in the claim dict it would become a stray column."""
        pipeline = client.app.state.pipeline
        client.post("/api/v1/explain", json={**claim, "persona": "adjuster"})
        _, submitted, claim_id = pipeline.calls[0]
        assert "persona" not in submitted
        assert claim_id == claim["claim_id"]


class TestMiddleware:
    def test_request_id_is_returned(self, client):
        assert client.get("/api/v1/health").headers[REQUEST_ID_HEADER]

    def test_request_id_is_honoured_when_supplied(self, client):
        response = client.get("/api/v1/health", headers={REQUEST_ID_HEADER: "upstream-trace"})
        assert response.headers[REQUEST_ID_HEADER] == "upstream-trace"

    def test_rate_limit_headers_present(self, client, claim):
        headers = client.post("/api/v1/predict", json=claim).headers
        assert headers["X-RateLimit-Limit"] == "100"
        assert int(headers["X-RateLimit-Remaining"]) < 100

    def test_explain_has_a_tighter_budget_than_predict(self, client, claim):
        """A paid third-party call cannot share a budget with an in-process one."""
        predict = client.post("/api/v1/predict", json=claim).headers["X-RateLimit-Limit"]
        explain = client.post("/api/v1/explain", json=claim).headers["X-RateLimit-Limit"]
        assert int(explain) < int(predict)


class TestDocumentation:
    def test_openapi_documents_both_explanation_shapes(self, client):
        """The persona filter should be visible in the schema, not only in the code."""
        schemas = client.get("/openapi.json").json()["components"]["schemas"]
        assert "CustomerExplanationResponse" in schemas
        assert "TechnicalExplanationResponse" in schemas
        customer = schemas["CustomerExplanationResponse"]["properties"]
        assert "decline_risk_score" not in customer

    def test_claim_example_is_published(self, client):
        schema = client.get("/openapi.json").json()["components"]["schemas"]["ClaimRequest"]
        assert "example" in schema

    def test_docs_are_reachable(self, client):
        assert client.get("/docs").status_code == 200


@needs_model
class TestAgainstTheRealPipeline:
    """Integration checks. Skipped when no model is registered, as in CI."""

    @pytest.fixture(scope="class")
    def live(self, tmp_path_factory: pytest.TempPathFactory) -> Any:
        app = create_app()
        with TestClient(app) as client:
            # Lifespan already built a monitor pointing at the real ledger; replace it so
            # integration runs do not bill themselves to the operations record.
            app.state.llm_monitor = LLMMonitor(
                ledger=tmp_path_factory.mktemp("ledger") / "llm_calls.jsonl"
            )
            yield client

    def test_routing_bands(self, live):
        expected = {
            "claim_straight_through": False,
            "claim_borderline": True,
            "claim_high_risk": True,
        }
        for name, review in expected.items():
            body = live.post("/api/v1/predict", json=example(name)).json()
            assert body["requires_human_review"] is review, name

    def test_forced_review_fires_below_the_threshold(self, live):
        """A theft claim leaves the fast queue on a rule, not on its score."""
        body = live.post("/api/v1/predict", json=example("claim_theft")).json()
        assert body["decline_risk_score"] < 0.30
        assert body["requires_human_review"] is True
        assert "theft_claim" in body["risk_flags"]

    def test_never_auto_declines(self, live):
        body = live.post("/api/v1/predict", json=example("claim_high_risk")).json()
        assert body["requires_human_review"] is True
        assert "declin" not in body["decision"].lower()

    def test_predict_stays_within_its_latency_budget(self, live):
        claim = example("claim_borderline")
        # Median of warmed requests: the first call pays for lazily-initialised BLAS thread
        # pools, which is a start-up cost the container absorbs once, not a serving cost.
        live.post("/api/v1/predict", json=claim)
        samples = [live.post("/api/v1/predict", json=claim).json()["latency_ms"] for _ in range(5)]
        assert statistics.median(samples) < 200, f"latencies {samples}"

    def test_customer_response_leaks_nothing_end_to_end(self, live):
        raw = live.post(
            "/api/v1/explain", json={**example("claim_borderline"), "persona": "customer"}
        ).text.lower()
        for term in ("decline_risk_score", "shap", "random_forest", "llama", "0.344"):
            assert term not in raw, f"leaked {term}"
