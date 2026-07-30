"""Semantic evaluation of the generated explanations, judged by a second LLM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.common.settings import get_llm_config
from src.genai.explanation_pipeline import STRAIGHT_THROUGH, Explanation, ExplanationPipeline
from tests.llm_judge import GroqJudge, judge_available

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "models"

pytestmark = [
    pytest.mark.llm_eval,
    pytest.mark.skipif(
        not (ARTIFACTS / "model.joblib").exists(),
        reason="no registered model; train and register one first",
    ),
    pytest.mark.skipif(not judge_available(), reason="no GROQ_API_KEY, cannot run a judge"),
]

# Read from configs/llm.yaml so the declared quality bar and the enforced one cannot drift.
THRESHOLDS = get_llm_config()["evaluation"]["deepeval"]["thresholds"]
FAITHFULNESS_MIN = THRESHOLDS["faithfulness"]
RELEVANCY_MIN = THRESHOLDS["answer_relevancy"]
HALLUCINATION_MAX = THRESHOLDS["hallucination"]
PERSONA_MIN = THRESHOLDS["persona_adherence"]

# Vocabulary a customer must never meet. Kept here rather than imported from the validator
# so a regression in the validator cannot silently relax the test that guards it.
INTERNAL_VOCABULARY = (
    "shap",
    "xgboost",
    "random forest",
    "feature",
    "model",
    "score",
    "threshold",
    "probability",
    "classifier",
    "inference",
)


@pytest.fixture(scope="module")
def pipeline() -> ExplanationPipeline:
    return ExplanationPipeline()


@pytest.fixture(scope="module")
def explanations(pipeline: ExplanationPipeline) -> dict[str, Explanation]:
    """One explanation per persona for the same claim, generated once for the module."""
    claim = json.loads((EXAMPLES / "claim_straight_through.json").read_text(encoding="utf-8"))
    claim.pop("persona", None)

    out: dict[str, Explanation] = {}
    for persona in ("customer", "adjuster", "auditor"):
        try:
            result = pipeline.explain(claim, persona=persona, claim_id="CLM-EVAL-001")
        except Exception as exc:
            pytest.skip(f"generation unavailable for {persona}: {exc}")
        # The pipeline degrades to a canned template when every provider is down. Judging
        # that template would score the fallback, not the prompt, and quietly pass.
        if result.used_template_fallback:
            pytest.skip(f"{persona} fell back to the static template; nothing to judge")
        out[persona] = result
    return out


@pytest.fixture(scope="module")
def judge() -> GroqJudge:
    return GroqJudge()


def rendered(explanation: Explanation) -> str:
    """The explanation as the recipient reads it, sections joined."""
    return "\n".join(_flatten(explanation.content))


def _flatten(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [line for v in value.values() for line in _flatten(v)]
    if isinstance(value, list):
        return [line for v in value for line in _flatten(v)]
    return [str(value)] if value is not None else []


def grounding(explanation: Explanation) -> list[str]:
    """The facts the LLM was given, restated as sentences a judge can check against."""
    prediction = explanation.prediction
    facts = [
        f"The claim identifier is {prediction.claim_id}.",
        f"The routing outcome is: {prediction.decision}.",
        f"Human review required: {prediction.requires_human_review}.",
    ]
    for factor in explanation.factors:
        name = factor.get("label") or factor.get("factor")
        direction = factor["direction"]
        if "shap_value" in factor:
            facts.append(
                f"{name} has value {factor['value']} and contributes "
                f"{factor['shap_value']} to the decline risk ({direction})."
            )
        else:
            facts.append(f"{name} {direction}.")
    if explanation.persona != "customer":
        facts.append(
            f"The decline risk score is {prediction.score:.4f} against a decision "
            f"threshold of {prediction.threshold}."
        )
    return facts


def _skip_if_metered(exc: Exception) -> None:
    if "rate_limit" in str(exc).lower() or "429" in str(exc):
        pytest.skip(f"judge quota exhausted: {exc}")
    raise exc


def measure(metric: Any, test_case: Any) -> float:
    """Run one metric, treating an exhausted judge quota as a skip rather than a failure."""
    try:
        metric.measure(test_case)
    except Exception as exc:
        _skip_if_metered(exc)
    return float(metric.score or 0.0)


def test_customer_explanation_is_faithful_to_the_context(
    explanations: dict[str, Explanation], judge: GroqJudge
) -> None:
    from deepeval.metrics import FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    explanation = explanations["customer"]
    case = LLMTestCase(
        input="Explain the outcome of this insurance claim to the customer.",
        actual_output=rendered(explanation),
        retrieval_context=grounding(explanation),
    )
    score = measure(FaithfulnessMetric(threshold=FAITHFULNESS_MIN, model=judge), case)
    assert score >= FAITHFULNESS_MIN, f"faithfulness {score:.2f}"


def test_customer_explanation_answers_the_customer_question(
    explanations: dict[str, Explanation], judge: GroqJudge
) -> None:
    from deepeval.metrics import AnswerRelevancyMetric
    from deepeval.test_case import LLMTestCase

    case = LLMTestCase(
        input="What happened to my claim, and what do I need to do next?",
        actual_output=rendered(explanations["customer"]),
    )
    score = measure(AnswerRelevancyMetric(threshold=RELEVANCY_MIN, model=judge), case)
    assert score >= RELEVANCY_MIN, f"relevancy {score:.2f}"


def test_adjuster_explanation_does_not_hallucinate(
    explanations: dict[str, Explanation], judge: GroqJudge
) -> None:
    from deepeval.metrics import HallucinationMetric
    from deepeval.test_case import LLMTestCase

    explanation = explanations["adjuster"]
    case = LLMTestCase(
        input="Summarise the model's assessment of this claim for the handling adjuster.",
        actual_output=rendered(explanation),
        context=grounding(explanation),
    )
    # Inverted metric: it reports the share of output contradicted by the context.
    score = measure(HallucinationMetric(threshold=HALLUCINATION_MAX, model=judge), case)
    assert score <= HALLUCINATION_MAX, f"hallucination {score:.2f}"


def test_auditor_explanation_is_faithful_to_its_provenance(
    explanations: dict[str, Explanation], judge: GroqJudge
) -> None:
    from deepeval.metrics import FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    explanation = explanations["auditor"]
    card = explanation.prediction
    provenance = grounding(explanation) + [
        f"The model version is {card.model_version} and the algorithm is {card.model_algorithm}.",
        f"The prompt version is {explanation.prompt_version}.",
    ]
    case = LLMTestCase(
        input="Record this decision for the audit trail, including its provenance.",
        actual_output=rendered(explanation),
        retrieval_context=provenance,
    )
    score = measure(FaithfulnessMetric(threshold=FAITHFULNESS_MIN, model=judge), case)
    assert score >= FAITHFULNESS_MIN, f"auditor faithfulness {score:.2f}"


def test_customer_explanation_reads_at_a_customer_level(
    explanations: dict[str, Explanation], judge: GroqJudge
) -> None:
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, SingleTurnParams

    # The one requirement no assertion can express: is this actually written for a layperson?
    metric = GEval(
        name="Customer Readability",
        criteria=(
            "Judge whether the output is written for an insurance customer with no "
            "technical background. It should use everyday language, explain the outcome "
            "plainly, stay respectful, and avoid statistics, machine-learning vocabulary "
            "and internal system detail."
        ),
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
        threshold=PERSONA_MIN,
        model=judge,
    )
    case = LLMTestCase(
        input="Explain this claim outcome to the customer.",
        actual_output=rendered(explanations["customer"]),
    )
    score = measure(metric, case)
    assert score >= PERSONA_MIN, f"readability {score:.2f} — {metric.reason}"


def test_customer_explanation_names_no_internal(explanations: dict[str, Explanation]) -> None:
    text = rendered(explanations["customer"]).lower()
    leaked = [word for word in INTERNAL_VOCABULARY if word in text]
    assert not leaked, f"internal vocabulary reached the customer: {leaked}"


def test_no_persona_contradicts_the_routing_outcome(
    explanations: dict[str, Explanation],
) -> None:
    # This exact failure happened twice in Phase 3: the LLM wrote "was declined" about a
    # claim the gate had routed straight through, despite the system prompt forbidding it.
    assert explanations["customer"].prediction.decision == STRAIGHT_THROUGH
    for persona, explanation in explanations.items():
        text = rendered(explanation).lower()
        assert "was declined" not in text, f"{persona} contradicts the outcome"
        assert "has been rejected" not in text, f"{persona} contradicts the outcome"


def test_every_persona_passes_its_runtime_validator(
    explanations: dict[str, Explanation],
) -> None:
    for persona, explanation in explanations.items():
        assert (
            explanation.validation.get("passed") is True
        ), f"{persona} failed runtime validation: {explanation.validation.get('failed_checks')}"
