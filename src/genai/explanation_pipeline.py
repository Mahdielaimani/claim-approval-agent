"""Facade over prediction, the HITL gate, SHAP, and the persona explanation."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from src.common.logger import bind_request_id, get_logger
from src.common.settings import get_api_config, get_llm_config, get_settings
from src.data.preprocessor import get_text_for_llm
from src.explainability.shap_service import ShapService
from src.genai.llm_client import LLMClient, LLMError
from src.genai.output_validator import OutputValidator, fallback_explanation
from src.genai.personas import DecisionContext, build_context, get_persona, risk_flags
from src.models.registry import load_production_model, load_transformers

logger = get_logger(__name__)

STRAIGHT_THROUGH = "straight-through processing candidate"
UNDER_REVIEW = "under review"
EXPERT_REVIEW = "mandatory expert review"


@dataclass
class Prediction:
    """Model output plus the routing decision derived from it."""

    claim_id: str
    request_id: str
    score: float
    decision: str
    requires_human_review: bool
    human_review_reason: str | None
    risk_flags: list[str] = field(default_factory=list)
    threshold: float = 0.5
    model_version: str = ""
    model_algorithm: str = ""
    latency_ms: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form for the /predict response."""
        return {
            "claim_id": self.claim_id,
            "request_id": self.request_id,
            "decline_risk_score": round(self.score, 4),
            "decision": self.decision,
            "requires_human_review": self.requires_human_review,
            "human_review_reason": self.human_review_reason,
            "risk_flags": self.risk_flags,
            "decision_threshold": self.threshold,
            "model_version": self.model_version,
            "model_algorithm": self.model_algorithm,
            "latency_ms": self.latency_ms,
        }


@dataclass
class Explanation:
    """A rendered explanation for one persona, with its provenance."""

    prediction: Prediction
    persona: str
    content: dict[str, Any]
    factors: list[dict[str, Any]] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    prompt_version: str = ""
    used_fallback_provider: bool = False
    used_template_fallback: bool = False
    validation: dict[str, Any] = field(default_factory=dict)
    retried: bool = False
    llm_latency_ms: int = 0
    total_latency_ms: int = 0
    # Carried through for the cost ledger, not for the response: a budget monitor that
    # cannot see tokens reports zero spend forever and never fires.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form for the /explain response, filtered for this persona."""
        # A client renders what it receives, so the envelope is filtered like the prompt.
        if self.persona == "customer":
            return {
                "claim_id": self.prediction.claim_id,
                "request_id": self.prediction.request_id,
                "persona": self.persona,
                "status": self.prediction.decision,
                "under_review": self.prediction.requires_human_review,
                "explanation": self.content,
                "factors": self.factors,
                "total_latency_ms": self.total_latency_ms,
            }

        return {
            **self.prediction.as_dict(),
            "persona": self.persona,
            "explanation": self.content,
            "factors": self.factors,
            "generation": {
                "provider": self.provider,
                "model": self.model,
                "prompt_version": self.prompt_version,
                "used_fallback_provider": self.used_fallback_provider,
                "used_template_fallback": self.used_template_fallback,
                "retried": self.retried,
                "llm_latency_ms": self.llm_latency_ms,
            },
            "validation": self.validation,
            "total_latency_ms": self.total_latency_ms,
        }


class ExplanationPipeline:
    """Loads every dependency once, then serves prediction and explanation per claim."""

    def __init__(self) -> None:
        settings = get_settings()
        self.llm_config = get_llm_config()
        hitl = get_api_config()["decision_layer"]["hitl"]["thresholds"]

        self.straight_through_below = float(hitl["straight_through_below"])
        self.mandatory_review_above = float(hitl["mandatory_review_above"])

        self.model, self.card = load_production_model()
        self.preprocessor, self.feature_engineer = load_transformers()
        self.shap = ShapService(self.model, self.card.feature_names)
        self.llm = LLMClient(self.llm_config)
        self.validator = OutputValidator(self.llm_config)
        self.prompt_version = self.llm_config["prompts"]["version"]

        self.env = Environment(
            loader=FileSystemLoader(settings.resolve(settings.prompt_path)),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.system_prompt = self.env.get_template(self.llm_config["prompts"]["system"]).render()

        logger.info(
            "pipeline ready",
            extra={
                "model_version": self.card.version,
                "algorithm": self.card.algorithm,
                "threshold": self.card.threshold,
                "prompt_version": self.prompt_version,
            },
        )

    def predict(self, claim: dict[str, Any], *, claim_id: str | None = None) -> Prediction:
        """Score one claim and route it. No LLM call on this path."""
        started = time.perf_counter()
        request_id = bind_request_id()
        cid = claim_id or claim.get("claim_id") or f"CLM-{uuid.uuid4().hex[:8].upper()}"

        cleaned, features = self._transform(claim)
        score = float(self.model.predict_proba(features)[0, 1])
        explanation = self.shap.explain(features, score=score)
        flags = risk_flags(cleaned, explanation)

        review, reason, decision = self._route(score, flags)
        prediction = Prediction(
            claim_id=cid,
            request_id=request_id,
            score=score,
            decision=decision,
            requires_human_review=review,
            human_review_reason=reason,
            risk_flags=flags,
            threshold=self.card.threshold,
            model_version=self.card.version,
            model_algorithm=self.card.algorithm,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        logger.info("claim scored", extra=prediction.as_dict())
        # Cached so explain() does not repeat the SHAP computation for the same claim.
        self._last = (cid, cleaned, features, explanation, prediction)
        return prediction

    def explain(
        self, claim: dict[str, Any], persona: str = "customer", *, claim_id: str | None = None
    ) -> Explanation:
        """Score one claim and render the explanation for one audience."""
        started = time.perf_counter()
        persona_config = get_persona(persona, self.llm_config)
        prediction = self.predict(claim, claim_id=claim_id)
        _, cleaned, features, shap_result, _ = self._last

        context = build_context(
            persona_config,
            DecisionContext(
                claim_id=prediction.claim_id,
                score=prediction.score,
                decision=prediction.decision,
                requires_human_review=prediction.requires_human_review,
                human_review_reason=prediction.human_review_reason,
                threshold=prediction.threshold,
                explanation=shap_result,
                claim=cleaned,
                issue_desc=get_text_for_llm(pd.Series(claim))["issue_desc"],
                model_version=self.card.version,
                model_algorithm=self.card.algorithm,
                prompt_version=self.prompt_version,
                trained_at=self.card.trained_at,
                git_commit=self.card.git_commit,
            ),
        )
        user_prompt = self.env.get_template(
            self.llm_config["prompts"]["templates"][persona]
        ).render(**context)

        result = self._generate(user_prompt, persona, prediction, context)
        result.factors = context["factors"]
        result.prompt_version = self.prompt_version
        result.total_latency_ms = int((time.perf_counter() - started) * 1000)

        logger.info(
            "explanation generated",
            extra={
                "persona": persona,
                "claim_id": prediction.claim_id,
                "template_fallback": result.used_template_fallback,
                "retried": result.retried,
                "total_latency_ms": result.total_latency_ms,
            },
        )
        return result

    def _route(self, score: float, flags: list[str]) -> tuple[bool, str | None, str]:
        """Apply the HITL gate to a decline-risk score."""
        if score >= self.mandatory_review_above:
            return True, "high_decline_risk", EXPERT_REVIEW
        if score >= self.straight_through_below:
            return True, "borderline_confidence", UNDER_REVIEW
        # Eligible for the fast queue, NOT approved — the distinction a regulator cares about.
        if flags:
            return True, f"forced_review:{flags[0]}", UNDER_REVIEW
        return False, None, STRAIGHT_THROUGH

    def _transform(self, claim: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
        # claim_id and status are request metadata, not features; passing them through would
        # log a dropped-column warning on every single prediction.
        frame = pd.DataFrame([{k: v for k, v in claim.items() if k not in ("claim_id", "status")}])
        cleaned = self.preprocessor.transform(frame)
        features = self.feature_engineer.transform(cleaned)
        features = features.drop(columns=["status"], errors="ignore")
        # Reordered against the card: a column mismatch scores the wrong features, silently.
        return cleaned.iloc[0].to_dict(), features[self.card.feature_names]

    def _generate(
        self,
        user_prompt: str,
        persona: str,
        prediction: Prediction,
        context: dict[str, Any],
    ) -> Explanation:
        """Call the LLM, validate, retry once, then fall back to a template."""
        is_declined = prediction.score >= self.mandatory_review_above
        attempts = self.validator.max_retries + 1
        last_validation: dict[str, Any] = {}

        for attempt in range(1, attempts + 1):
            try:
                response = self.llm.complete(
                    self.system_prompt,
                    user_prompt,
                    persona=persona,
                    claim_id=prediction.claim_id,
                )
            except LLMError as exc:
                logger.warning(
                    "llm unavailable, using template",
                    extra={"persona": persona, "error": str(exc)[:200]},
                )
                break

            validation = self.validator.validate(
                response.content,
                persona=persona,
                context=context,
                requires_human_review=prediction.requires_human_review,
                is_declined=is_declined,
            )
            last_validation = validation.summary()

            if validation.passed:
                return Explanation(
                    prediction=prediction,
                    persona=persona,
                    content=validation.payload,
                    provider=response.provider,
                    model=response.model,
                    used_fallback_provider=response.used_fallback,
                    validation=last_validation,
                    retried=attempt > 1,
                    llm_latency_ms=response.latency_ms,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    cost_usd=response.cost_usd,
                )

            logger.warning(
                "output rejected",
                extra={
                    "persona": persona,
                    "attempt": attempt,
                    "issues": [i.as_dict() for i in validation.errors],
                },
            )

        # A degraded explanation is acceptable; withholding the decision is not.
        return Explanation(
            prediction=prediction,
            persona=persona,
            content=fallback_explanation(
                persona,
                requires_human_review=prediction.requires_human_review,
                claim_id=prediction.claim_id,
            ),
            provider="template",
            model="template",
            used_template_fallback=True,
            validation=last_validation,
            retried=attempts > 1,
        )

    def health(self) -> dict[str, Any]:
        """What /health reports about the loaded components."""
        return {
            "status": "ok",
            "model_version": self.card.version,
            "model_algorithm": self.card.algorithm,
            "decision_threshold": self.card.threshold,
            "n_features": len(self.card.feature_names),
            "prompt_version": self.prompt_version,
            "llm_mode": "mock" if self.llm.mock_mode else "live",
            "llm_chain": [p.model for p in self.llm.providers],
            "trained_at": self.card.trained_at,
            "git_commit": self.card.git_commit,
        }


def build_transformer_artifacts() -> Path:
    """Fit and persist the preprocessing state from the training dataset."""
    # Not done at startup: that would need the proprietary dataset inside the container.
    from src.data.feature_engineer import FeatureEngineer
    from src.data.loader import load_raw
    from src.data.preprocessor import Preprocessor
    from src.models.registry import save_transformers

    raw = load_raw()
    pre = Preprocessor().fit(raw)
    cleaned = pre.transform(raw)
    fe = FeatureEngineer().fit(cleaned)
    fe.transform(cleaned)
    return save_transformers(pre, fe)
