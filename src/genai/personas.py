"""Per-audience view of a decision: what each persona is allowed to be told."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.common.logger import get_logger
from src.common.settings import get_llm_config

logger = get_logger(__name__)

CUSTOMER = "customer"
ADJUSTER = "adjuster"
AUDITOR = "auditor"
PERSONAS: tuple[str, ...] = (CUSTOMER, ADJUSTER, AUDITOR)


@dataclass(frozen=True)
class PersonaConfig:
    """What one audience sees, and in what register."""

    name: str
    top_n_shap: int = 5
    show_score: bool = False
    show_shap_values: bool = False
    show_next_steps: bool = False
    show_risk_flags: bool = False
    include_issue_desc: bool = False
    require_versions: bool = False
    tone: str = "professional"
    reading_level: str = "standard"

    @property
    def is_technical(self) -> bool:
        """True when this audience receives model internals."""
        return self.show_shap_values or self.show_score


def get_persona(name: str, config: dict[str, Any] | None = None) -> PersonaConfig:
    """Load one persona's configuration by name."""
    if name not in PERSONAS:
        raise ValueError(f"Unknown persona '{name}'. Available: {list(PERSONAS)}")

    cfg = (config or get_llm_config())["personas"].get(name, {})
    return PersonaConfig(name=name, **cfg)


def risk_flags(claim: dict[str, Any], explanation: Any) -> list[str]:
    """Conditions that force human review regardless of the score."""
    # Each threshold traces to a measured subgroup, not to intuition — see DESIGN §3.
    flags: list[str] = []
    if claim.get("claimType") == "Theft":
        flags.append("theft_claim")
    if claim.get("policyStatus") == "Grace Period":
        flags.append("grace_period_policy")
    if float(claim.get("rrp") or 0) > 2500:
        flags.append("high_value_claim")

    diag = next(
        (c.value for c in explanation.contributions if c.feature == "diag_completeness"), None
    )
    if diag is not None and diag < 3:
        flags.append("insufficient_diagnostics")
    return flags


@dataclass
class DecisionContext:
    """Everything known about one decision, before persona filtering."""

    claim_id: str
    score: float
    decision: str
    requires_human_review: bool
    human_review_reason: str | None
    threshold: float
    explanation: Any
    # The cleaned row: raw SEK amounts would contradict the EUR values in the SHAP factors.
    claim: dict[str, Any] = field(default_factory=dict)
    issue_desc: str = ""
    model_version: str = ""
    model_algorithm: str = ""
    prompt_version: str = ""
    trained_at: str = ""
    git_commit: str = ""


def build_context(persona: PersonaConfig, decision: DecisionContext) -> dict[str, Any]:
    """Assemble the template variables this persona is permitted to receive."""
    # Filtered here, not in the prompt: a field that is absent cannot be leaked by one.
    context: dict[str, Any] = {
        "persona": persona.name,
        "tone": persona.tone,
        "reading_level": persona.reading_level,
        "claim_id": decision.claim_id,
        "decision": decision.decision,
        "requires_human_review": decision.requires_human_review,
        "outcome_statement": _outcome_statement(decision),
        "claim": _claim_summary(decision.claim, technical=persona.is_technical),
        "factors": _factors(persona, decision.explanation),
    }

    if persona.show_score:
        context["score"] = round(decision.score, 4)
        context["threshold"] = decision.threshold
        context["base_value"] = round(decision.explanation.base_value, 4)

    if persona.show_risk_flags:
        context["risk_flags"] = risk_flags(decision.claim, decision.explanation)
        context["human_review_reason"] = decision.human_review_reason

    if persona.include_issue_desc and decision.issue_desc:
        # Untrusted customer text: the template isolates it, the system prompt distrusts it.
        context["issue_desc"] = decision.issue_desc

    if persona.show_next_steps:
        context["next_steps_expected"] = True

    if persona.require_versions:
        context["provenance"] = {
            "model_version": decision.model_version,
            "model_algorithm": decision.model_algorithm,
            "prompt_version": decision.prompt_version,
            "trained_at": decision.trained_at,
            "git_commit": decision.git_commit,
            "decision_threshold": decision.threshold,
        }

    logger.info(
        "persona context built",
        extra={
            "persona": persona.name,
            "keys": sorted(context),
            "n_factors": len(context["factors"]),
        },
    )
    return context


def _outcome_statement(decision: DecisionContext) -> str:
    """Spell out what has and has not been decided, for the template to restate."""
    # Measured failure: given only "score 0.474, threshold 0.344", Llama 3.3 inferred
    # "the claim was declined" twice in a row. The inference is reasonable and wrong, so
    # the conclusion is supplied rather than left to be derived.
    if not decision.requires_human_review:
        return (
            "NO ADVERSE DECISION HAS BEEN ISSUED. This claim is eligible for the "
            "straight-through queue. That is a routing outcome, not an approval."
        )
    return (
        "NO DECISION HAS BEEN ISSUED — the claim is NOT declined and NOT approved. "
        "The score exceeding the threshold routes this claim to a human reviewer; "
        "it is not itself a decision. Any adverse outcome requires a human."
    )


def _factors(persona: PersonaConfig, explanation: Any) -> list[dict[str, Any]]:
    """Top contributions, rendered at the level of detail this persona may receive."""
    top = explanation.top(persona.top_n_shap)
    if persona.show_shap_values:
        return [c.as_dict() for c in top]

    # Balanced list: showing only what counted against a customer reads as accusation.
    balanced = explanation.increasing(3) + explanation.decreasing(2)
    return [c.as_dict(plain=True) for c in balanced]


def _claim_summary(claim: dict[str, Any], *, technical: bool) -> dict[str, Any]:
    """Claim fields worth restating, trimmed for a non-technical reader."""
    plain_fields = ("claimType", "deviceType", "coverage", "policyStatus")
    technical_fields = plain_fields + (
        "country",
        "channel",
        "rrp",
        "excessFee",
        "balanceRRP",
        "productCode",
        "policy_class",
    )
    fields = technical_fields if technical else plain_fields
    # NaN is dropped, not passed through: it is invalid JSON and meaningless to a reader.
    return {k: v for k in fields if k in claim and _present(v := claim[k])}


def _present(value: Any) -> bool:
    """False for None, NaN, empty strings, and the preprocessor's UNKNOWN sentinel."""
    if value is None:
        return False
    if isinstance(value, float) and value != value:
        return False
    if isinstance(value, str):
        # UNKNOWN is an internal marker for a missing category. Telling a customer their
        # device type is "UNKNOWN" restates a gap in our records as a fact about them.
        return bool(value.strip()) and value.strip().upper() != "UNKNOWN"
    return True
