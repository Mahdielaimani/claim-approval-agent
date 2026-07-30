"""Runtime checks on LLM output: schema, information leakage, grounding, consistency."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from src.common.logger import get_logger
from src.common.settings import get_llm_config

logger = get_logger(__name__)

ERROR = "error"
WARNING = "warning"

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)

# Lookbehind excludes identifiers: without it "CLM-000042" reads as the number -42.
_NUMBER = re.compile(r"(?<![A-Za-z0-9\-])-?\d+(?:[.,]\d+)?")

# Matched as assertions, not bare words: "reduces decline risk" is legitimate phrasing.
_DECLINED_ASSERTIONS = re.compile(
    r"\b(?:was|is|has been|been|being)\s+(?:declined|rejected|refused|denied|turned down)\b"
    r"|\bwe\s+(?:have\s+)?(?:declined|rejected|refused|denied)\b"
    r"|\byour claim (?:was|is|has been)\s+(?:declined|rejected|refused|denied)\b",
    re.IGNORECASE,
)

_FINALITY_ASSERTIONS = re.compile(
    r"\b(?:final decision (?:has been|was) made|decision is final|claim (?:is|has been) "
    r"(?:settled|closed|finalised|finalized)|has been approved|was approved)\b",
    re.IGNORECASE,
)

_REVIEWED_ASSERTIONS = re.compile(
    r"\b(?:has been|was) reviewed by (?:a|an|our)\s+\w+"
    r"|\ban? (?:adjuster|expert|human) (?:has|already) reviewed\b",
    re.IGNORECASE,
)


@dataclass
class ValidationIssue:
    """One failed check."""

    check: str
    severity: str
    message: str
    evidence: str = ""

    def as_dict(self) -> dict[str, str]:
        """Serialisable form for logs and the audit trail."""
        return {
            "check": self.check,
            "severity": self.severity,
            "message": self.message,
            "evidence": self.evidence[:200],
        }


@dataclass
class ValidationResult:
    """Outcome of validating one response."""

    payload: dict[str, Any] = field(default_factory=dict)
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        """Issues severe enough to reject the response."""
        return [i for i in self.issues if i.severity == ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Issues worth recording that do not reject the response."""
        return [i for i in self.issues if i.severity == WARNING]

    @property
    def passed(self) -> bool:
        """True when nothing blocking was found."""
        return not self.errors

    def summary(self) -> dict[str, Any]:
        """Compact result for logging."""
        return {
            "passed": self.passed,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "failed_checks": [i.check for i in self.errors],
        }


class OutputValidator:
    """Applies increasingly specific checks to one LLM response."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = (config or get_llm_config())["validation"]
        self.required_fields = cfg["required_fields"]
        self.blocklist = [t.lower() for t in cfg["customer_blocklist"]]
        self.forbidden_everywhere = [t.lower() for t in cfg["forbidden_everywhere"]]
        self.check_sentiment = cfg["check_sentiment_consistency"]
        self.check_grounding = cfg["check_numeric_grounding"]
        self.grounding_min = float(cfg["grounding_min_magnitude"])
        self.grounding_tolerance = float(cfg["grounding_tolerance"])
        self.max_retries = int(cfg["max_retries_on_failure"])

    def validate(
        self,
        raw_content: str,
        *,
        persona: str,
        context: dict[str, Any],
        requires_human_review: bool,
        is_declined: bool = False,
    ) -> ValidationResult:
        """Run every check against one response and collect what failed."""
        result = ValidationResult()

        payload = self._parse(raw_content, result)
        # Nothing downstream is meaningful without a parsed object, so the chain stops.
        if payload is None:
            return result
        result.payload = payload

        text = self._all_text(payload).lower()
        self._check_schema(payload, persona, result)
        self._check_forbidden(text, result)
        if persona == "customer":
            self._check_leakage(text, result)
        if self.check_sentiment:
            self._check_consistency(text, requires_human_review, is_declined, result)
        if self.check_grounding:
            self._check_grounding(payload, context, result)

        logger.info("output validated", extra={"persona": persona, **result.summary()})
        return result

    def _parse(self, raw_content: str, result: ValidationResult) -> dict[str, Any] | None:
        try:
            payload = json.loads(_FENCE.sub("", raw_content).strip())
        except json.JSONDecodeError as exc:
            result.issues.append(
                ValidationIssue(
                    "json_parse", ERROR, f"Response is not valid JSON: {exc}", raw_content[:200]
                )
            )
            return None
        if not isinstance(payload, dict):
            result.issues.append(
                ValidationIssue(
                    "json_parse", ERROR, f"Expected a JSON object, got {type(payload).__name__}."
                )
            )
            return None
        return payload

    def _check_schema(
        self, payload: dict[str, Any], persona: str, result: ValidationResult
    ) -> None:
        for field_name in self.required_fields.get(persona, []):
            if field_name not in payload:
                result.issues.append(
                    ValidationIssue("schema", ERROR, f"Missing required field '{field_name}'.")
                )
            elif not payload[field_name]:
                result.issues.append(
                    ValidationIssue("schema", ERROR, f"Field '{field_name}' is empty.")
                )

    def _check_forbidden(self, text: str, result: ValidationResult) -> None:
        # "auto-approved" is a regulatory problem in any persona, not a style issue.
        for term in self.forbidden_everywhere:
            if term in text:
                result.issues.append(
                    ValidationIssue(
                        "forbidden_wording", ERROR, f"Response contains '{term}'.", term
                    )
                )

    def _check_leakage(self, text: str, result: ValidationResult) -> None:
        for term in self.blocklist:
            if term in text:
                result.issues.append(
                    ValidationIssue(
                        "customer_leakage",
                        ERROR,
                        f"Customer response exposes internal term '{term}'.",
                        term,
                    )
                )

    def _check_consistency(
        self, text: str, requires_review: bool, is_declined: bool, result: ValidationResult
    ) -> None:
        """Reject an outcome the decision record does not support."""
        # Earned its place: Llama 3.3 wrote "was declined" about a straight-through claim.
        if not is_declined and (match := _DECLINED_ASSERTIONS.search(text)):
            result.issues.append(
                ValidationIssue(
                    "sentiment_consistency",
                    ERROR,
                    "Response states the claim was declined when it was not.",
                    match.group(0),
                )
            )
        if requires_review:
            if match := _FINALITY_ASSERTIONS.search(text):
                result.issues.append(
                    ValidationIssue(
                        "sentiment_consistency",
                        ERROR,
                        "Response presents a decision as final while review is " "outstanding.",
                        match.group(0),
                    )
                )
            if match := _REVIEWED_ASSERTIONS.search(text):
                result.issues.append(
                    ValidationIssue(
                        "sentiment_consistency",
                        ERROR,
                        "Response claims a human already reviewed the claim.",
                        match.group(0),
                    )
                )

    def _check_grounding(
        self, payload: dict[str, Any], context: dict[str, Any], result: ValidationResult
    ) -> None:
        """Flag figures that appear nowhere in the input the model was given."""
        allowed = self._allowed_numbers(context)
        for value in self._numbers(self._all_text(payload)):
            if abs(value) < self.grounding_min:
                continue
            if not any(
                abs(value - a) <= self.grounding_tolerance * max(abs(a), 1.0) for a in allowed
            ):
                # Warning, not error: false positives here cost more than they prevent.
                result.issues.append(
                    ValidationIssue(
                        "numeric_grounding",
                        WARNING,
                        f"Figure {value} does not appear in the input.",
                        str(value),
                    )
                )

    @staticmethod
    def _all_text(payload: Any) -> str:
        """Every string in the payload, flattened, so checks see nested lists too."""
        if isinstance(payload, str):
            return payload
        if isinstance(payload, dict):
            return " ".join(OutputValidator._all_text(v) for v in payload.values())
        if isinstance(payload, list | tuple):
            return " ".join(OutputValidator._all_text(v) for v in payload)
        return str(payload)

    @staticmethod
    def _numbers(text: str) -> list[float]:
        out: list[float] = []
        for token in _NUMBER.findall(text):
            try:
                out.append(float(token.replace(",", ".")))
            except ValueError:
                continue
        return out

    @classmethod
    def _allowed_numbers(cls, context: dict[str, Any]) -> set[float]:
        """Every figure the model was shown, including rounded forms."""
        allowed: set[float] = set()
        for value in cls._walk(context):
            allowed.add(value)
            # The prompt shows 1503.8835; writing 1503.88 or 1504 invents nothing.
            allowed.update({round(value, 2), round(value, 1), float(round(value))})
        return allowed

    @classmethod
    def _walk(cls, node: Any) -> list[float]:
        if isinstance(node, bool):
            return []
        if isinstance(node, int | float):
            return [float(node)]
        if isinstance(node, dict):
            return [v for item in node.values() for v in cls._walk(item)]
        if isinstance(node, list | tuple):
            return [v for item in node for v in cls._walk(item)]
        if isinstance(node, str):
            return cls._numbers(node)
        return []


def fallback_explanation(
    persona: str, *, requires_human_review: bool, claim_id: str
) -> dict[str, Any]:
    """Deterministic template used when the LLM fails or its output is rejected."""
    # A plain true explanation beats an error page, and beats one that failed validation.
    status = (
        "is currently under review by our team"
        if requires_human_review
        else "has been processed and is moving through our standard handling"
    )
    base = {
        "summary": f"Claim {claim_id} {status}.",
        "key_factors": ["A detailed explanation is temporarily unavailable."],
        "generated": "template",
    }
    if persona == "customer":
        base["next_steps"] = ["Contact support if you need an update on this claim."]
    elif persona == "adjuster":
        base["risk_assessment"] = "Narrative explanation unavailable; review the factor list."
        base["review_notes"] = "Assess this claim from the attribution data directly."
    else:
        base["provenance_record"] = "See the structured decision record accompanying this claim."
        base["human_oversight"] = (
            f"Human review {'required' if requires_human_review else 'not required'}."
        )
        base["limitations"] = "Generated explanation unavailable for this decision."
    return base
