"""Cross-validation and metrics for the Declined class."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.common.logger import get_logger
from src.common.settings import get_training_config

logger = get_logger(__name__)


@dataclass
class FoldMetrics:
    """Metrics for one validation fold at its selected threshold."""

    threshold: float
    f1_declined: float
    recall_declined: float
    precision_declined: float
    roc_auc: float
    pr_auc: float
    brier: float
    n_val: int
    n_declined: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int

    def as_dict(self) -> dict[str, float]:
        """Flat mapping for tabular reporting."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class CVResult:
    """Aggregated cross-validation outcome for one model."""

    model_name: str
    folds: list[FoldMetrics] = field(default_factory=list)
    oof_probabilities: np.ndarray | None = None
    oof_targets: np.ndarray | None = None

    def mean(self, metric: str) -> float:
        """Mean of a metric across folds."""
        return float(np.mean([getattr(f, metric) for f in self.folds]))

    def std(self, metric: str) -> float:
        """Standard deviation of a metric across folds."""
        return float(np.std([getattr(f, metric) for f in self.folds]))

    def summary(self) -> dict[str, Any]:
        """Row-shaped summary for the model comparison table."""
        return {
            "model": self.model_name,
            "f1_declined": round(self.mean("f1_declined"), 4),
            "f1_std": round(self.std("f1_declined"), 4),
            "recall_declined": round(self.mean("recall_declined"), 4),
            "precision_declined": round(self.mean("precision_declined"), 4),
            "roc_auc": round(self.mean("roc_auc"), 4),
            "pr_auc": round(self.mean("pr_auc"), 4),
            "brier": round(self.mean("brier"), 4),
            "threshold": round(self.mean("threshold"), 3),
            "threshold_std": round(self.std("threshold"), 3),
        }

    def fold_frame(self) -> pd.DataFrame:
        """Per-fold metrics, for inspecting stability rather than averages."""
        return pd.DataFrame([f.as_dict() for f in self.folds])


def evaluate_at_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict[str, float]:
    """Compute Declined-class metrics at a given decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "f1_declined": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_declined": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision_declined": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        # Threshold-free, so they measure ranking quality independently of the operating point.
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        # The HITL gate reads the score as a probability, so calibration is a requirement.
        "brier": brier_score_loss(y_true, y_prob),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        # The expensive error: a claim that should have been declined and was paid.
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    metric: str = "f1_declined",
    search_range: tuple[float, float] = (0.05, 0.95),
    step: float = 0.01,
) -> float:
    """Scan for the threshold maximising the given metric."""
    scorer = {"f1_declined": f1_score, "recall_declined": recall_score}[metric]
    grid = np.arange(search_range[0], search_range[1] + step / 2, step)
    scores = [scorer(y_true, (y_prob >= t).astype(int), pos_label=1, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def cross_validate(
    estimator_factory: Callable[[], Any],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    model_name: str = "model",
    config: dict[str, Any] | None = None,
) -> CVResult:
    """Stratified k-fold evaluation with the threshold chosen inside each fold."""
    cfg = config or get_training_config()
    n_folds = cfg["split"]["n_folds"]
    random_state = cfg["data"]["random_state"]
    ev = cfg["evaluation"]

    X_arr, y_arr = X.reset_index(drop=True), y.reset_index(drop=True).to_numpy()
    outer = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    result = CVResult(model_name=model_name)
    oof = np.zeros(len(y_arr), dtype=float)

    for train_idx, val_idx in outer.split(X_arr, y_arr):
        X_train, X_val = X_arr.iloc[train_idx], X_arr.iloc[val_idx]
        y_train, y_val = y_arr[train_idx], y_arr[val_idx]

        estimator = estimator_factory()
        estimator.fit(X_train, y_train)
        val_prob = estimator.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_prob

        # Threshold from an inner split: tuning it on the validation fold would report an
        # operating point that cannot exist in production.
        if ev.get("threshold_search", True):
            inner_prob = cross_val_predict(
                estimator_factory(),
                X_train,
                y_train,
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state),
                method="predict_proba",
            )[:, 1]
            threshold = find_best_threshold(
                y_train,
                inner_prob,
                search_range=tuple(ev["threshold_range"]),
                step=ev["threshold_step"],
            )
        else:
            threshold = 0.5

        metrics = evaluate_at_threshold(y_val, val_prob, threshold)
        result.folds.append(
            FoldMetrics(
                **metrics,
                n_val=len(y_val),
                n_declined=int(y_val.sum()),
            )
        )

    result.oof_probabilities = oof
    result.oof_targets = y_arr

    logger.info(
        "cross-validation complete",
        extra={
            "model": model_name,
            "folds": n_folds,
            "f1_declined": round(result.mean("f1_declined"), 4),
            "f1_std": round(result.std("f1_declined"), 4),
            "recall_declined": round(result.mean("recall_declined"), 4),
            "threshold": round(result.mean("threshold"), 3),
        },
    )
    return result


def comparison_frame(results: list[CVResult]) -> pd.DataFrame:
    """Model comparison table, ranked by F1 on the Declined class."""
    return (
        pd.DataFrame([r.summary() for r in results])
        .sort_values("f1_declined", ascending=False)
        .reset_index(drop=True)
    )


def baseline_metrics(y: pd.Series) -> dict[str, float]:
    """Metrics for the constant 'Completed' classifier, as the floor to beat."""
    # 84.3% accuracy with zero recall is the trap the whole metric choice exists to avoid.
    y_arr = y.to_numpy()
    majority = np.zeros_like(y_arr)
    return {
        "accuracy": float((majority == y_arr).mean()),
        "f1_declined": f1_score(y_arr, majority, pos_label=1, zero_division=0),
        "recall_declined": recall_score(y_arr, majority, pos_label=1, zero_division=0),
    }
