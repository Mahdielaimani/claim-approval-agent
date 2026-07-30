"""HTTP endpoints. Orchestration lives in the pipeline, not here."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from src.api.schemas import (
    ClaimRequest,
    CustomerExplanationResponse,
    ErrorResponse,
    ExplainRequest,
    HealthResponse,
    Persona,
    PredictionResponse,
    TechnicalExplanationResponse,
)
from src.common.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1")

ExplanationResponse = CustomerExplanationResponse | TechnicalExplanationResponse


def _pipeline(request: Request) -> Any:
    """The pipeline loaded once at application startup."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Train and register a model, then restart.",
        )
    return pipeline


def _monitor(request: Request) -> Any:
    """The LLM ledger, attached at startup."""
    monitor = getattr(request.app.state, "llm_monitor", None)
    if monitor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Monitoring not initialised.",
        )
    return monitor


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="What is loaded and serving",
    tags=["Health"],
)
async def health(request: Request) -> dict[str, Any]:
    """Report the model version, prompt version, and provider chain."""
    return _pipeline(request).health()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Score a claim and route it",
    tags=["Prediction"],
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def predict(request: Request, claim: ClaimRequest) -> dict[str, Any]:
    """Return the decline-risk score and the routing decision. No LLM on this path."""
    # Separate from /explain: no external dependency, so it is the only endpoint with an SLA.
    prediction = _pipeline(request).predict(claim.to_claim(), claim_id=claim.claim_id)
    return prediction.as_dict()


@router.post(
    "/explain",
    response_model=ExplanationResponse,
    response_model_exclude_none=False,
    summary="Score a claim and explain it for one audience",
    tags=["Explanation"],
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def explain(request: Request, payload: ExplainRequest) -> dict[str, Any]:
    """Return a persona-appropriate explanation alongside the routing decision."""
    # Response shape depends on the persona, and that filtering belongs to the pipeline.
    explanation = _pipeline(request).explain(
        payload.to_claim(),
        persona=payload.persona.value,
        claim_id=payload.claim_id,
    )
    _monitor(request).record_explanation(explanation)
    return explanation.as_dict()


@router.get(
    "/metrics",
    summary="LLM spend, provider health and latency over the last 30 days",
    tags=["Monitoring"],
)
async def metrics(request: Request, window_hours: int = 720) -> dict[str, Any]:
    """Aggregate the recorded LLM calls against the configured thresholds."""
    return _monitor(request).snapshot(window_hours).as_dict()


@router.get(
    "/personas",
    summary="Audiences an explanation can be written for",
    tags=["Explanation"],
)
async def personas() -> dict[str, Any]:
    """List the available personas and what each one is shown."""
    from src.common.settings import get_llm_config

    config = get_llm_config()["personas"]
    return {
        "personas": [
            {
                "name": persona.value,
                "top_n_factors": config[persona.value]["top_n_shap"],
                "sees_score": config[persona.value].get("show_score", False),
                "sees_attribution": config[persona.value].get("show_shap_values", False),
                "sees_provenance": config[persona.value].get("require_versions", False),
                "tone": config[persona.value]["tone"],
            }
            for persona in Persona
        ]
    }
