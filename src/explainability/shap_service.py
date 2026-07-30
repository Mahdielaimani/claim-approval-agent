"""Per-claim SHAP attribution, the factual evidence every explanation is built from."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.common.logger import get_logger
from src.explainability.feature_labels import label_for, unit_for

logger = get_logger(__name__)

# The Declined class. Binary tree ensembles return an attribution per class.
POSITIVE_CLASS = 1


@dataclass
class FeatureContribution:
    """One feature's signed contribution to this claim's decline risk."""

    feature: str
    label: str
    plain_label: str
    value: float
    shap_value: float
    direction: str
    unit: str = ""

    @property
    def magnitude(self) -> float:
        """Absolute contribution, for ranking."""
        return abs(self.shap_value)

    def as_dict(self, *, plain: bool = False) -> dict[str, Any]:
        """Serialisable form; plain=True omits the numbers a customer must not see."""
        if plain:
            return {"factor": self.plain_label, "direction": self.direction}
        return {
            "feature": self.feature,
            "label": self.label,
            "value": round(self.value, 4),
            "shap_value": round(self.shap_value, 4),
            "direction": self.direction,
            "unit": self.unit,
        }


@dataclass
class ClaimExplanation:
    """SHAP output for a single claim."""

    score: float
    base_value: float
    contributions: list[FeatureContribution] = field(default_factory=list)

    def top(self, n: int = 5) -> list[FeatureContribution]:
        """The n features that moved this claim's score the most."""
        return sorted(self.contributions, key=lambda c: c.magnitude, reverse=True)[:n]

    def increasing(self, n: int = 5) -> list[FeatureContribution]:
        """Features that pushed the score towards decline."""
        raised = [c for c in self.contributions if c.shap_value > 0]
        return sorted(raised, key=lambda c: c.shap_value, reverse=True)[:n]

    def decreasing(self, n: int = 5) -> list[FeatureContribution]:
        """Features that pushed the score away from decline."""
        lowered = [c for c in self.contributions if c.shap_value < 0]
        return sorted(lowered, key=lambda c: c.shap_value)[:n]

    def check_sum(self) -> float:
        """Base value plus all contributions, which must reconstruct the raw output."""
        # Shapley values are additive: a drifting sum means the attribution is wrong.
        return self.base_value + sum(c.shap_value for c in self.contributions)


class ShapService:
    """Builds a TreeExplainer once and attributes individual claims against it."""

    def __init__(self, model: Any, feature_names: list[str]) -> None:
        self.feature_names = list(feature_names)
        self._preprocessor, self._estimator = self._split(model)
        # Built once at startup: constructing it walks every tree, too slow per request.
        self._explainer = shap.TreeExplainer(self._estimator)
        logger.info(
            "shap explainer ready",
            extra={
                "estimator": type(self._estimator).__name__,
                "features": len(self.feature_names),
            },
        )

    @staticmethod
    def _split(model: Any) -> tuple[Any, Any]:
        """Separate the fitted preprocessing steps from the estimator TreeExplainer needs."""
        if isinstance(model, Pipeline):
            return Pipeline(model.steps[:-1]), model.steps[-1][1]
        return None, model

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        # TreeExplainer must see what the trees see, so imputation is applied first.
        ordered = X[self.feature_names]
        if self._preprocessor is None:
            return ordered
        return pd.DataFrame(
            self._preprocessor.transform(ordered), columns=self.feature_names, index=X.index
        )

    def _base_value(self) -> float:
        expected = self._explainer.expected_value
        if isinstance(expected, np.ndarray | list):
            return float(expected[POSITIVE_CLASS] if len(expected) > 1 else expected[0])
        return float(expected)

    def explain(self, X: pd.DataFrame, score: float | None = None) -> ClaimExplanation:
        """Attribute a single claim's decline risk across its features."""
        if len(X) != 1:
            raise ValueError(f"explain() takes exactly one claim, got {len(X)} rows.")

        prepared = self._prepare(X)
        values = self._explainer.shap_values(prepared)
        row = self._select_class(values)[0]

        contributions = [
            FeatureContribution(
                feature=name,
                label=label_for(name),
                plain_label=label_for(name, plain=True),
                value=float(prepared.iloc[0][name]),
                shap_value=float(row[i]),
                # Relative to decline risk, never "good" or "bad" — audiences differ.
                direction="increases decline risk" if row[i] > 0 else "reduces decline risk",
                unit=unit_for(name),
            )
            for i, name in enumerate(self.feature_names)
        ]

        return ClaimExplanation(
            score=float(score) if score is not None else self._score(X),
            base_value=self._base_value(),
            contributions=contributions,
        )

    def explain_batch(self, X: pd.DataFrame) -> np.ndarray:
        """Raw SHAP matrix for a set of claims, for summary plots and drift checks."""
        return self._select_class(self._explainer.shap_values(self._prepare(X)))

    def _score(self, X: pd.DataFrame) -> float:
        estimator = self._estimator
        prepared = self._prepare(X)
        return float(estimator.predict_proba(prepared)[0, POSITIVE_CLASS])

    @staticmethod
    def _select_class(values: Any) -> np.ndarray:
        """Pull the Declined-class attributions out of SHAP's per-class output."""
        # shap returns (n, features, classes) or a list per class, depending on version.
        if isinstance(values, list):
            return np.asarray(values[POSITIVE_CLASS])
        array = np.asarray(values)
        return array[:, :, POSITIVE_CLASS] if array.ndim == 3 else array
