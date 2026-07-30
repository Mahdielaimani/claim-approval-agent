"""Population Stability Index between a frozen training reference and live traffic."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.logger import get_logger
from src.common.settings import get_monitoring_config, get_settings

logger = get_logger(__name__)

REFERENCE_FILE = "drift_reference.json"

WATCH = "watch"
ALARM = "alarm"
STABLE = "stable"

# Below this many distinct values a feature is treated as categorical and each value gets
# its own bin. Quantile edges on an encoded categorical are meaningless: the integers are
# labels, so the midpoint between two of them does not name anything.
CATEGORICAL_MAX_CARDINALITY = 12


@dataclass
class FeatureDrift:
    """PSI for one feature, with the bin that moved most."""

    feature: str
    psi: float
    status: str
    largest_shift_bin: str
    reference_share: float
    current_share: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "psi": round(self.psi, 4),
            "status": self.status,
            "largest_shift_bin": self.largest_shift_bin,
            "reference_share": round(self.reference_share, 4),
            "current_share": round(self.current_share, 4),
        }


@dataclass
class DriftReport:
    """Result of comparing one batch against the training reference."""

    n_rows: int
    features: list[FeatureDrift] = field(default_factory=list)
    prediction_drift: FeatureDrift | None = None
    skipped: str | None = None

    @property
    def alarms(self) -> list[FeatureDrift]:
        return [f for f in self.features if f.status == ALARM]

    @property
    def watches(self) -> list[FeatureDrift]:
        return [f for f in self.features if f.status == WATCH]

    @property
    def requires_investigation(self) -> bool:
        """True when any monitored distribution breached the alarm threshold."""
        predicted = self.prediction_drift is not None and self.prediction_drift.status == ALARM
        return bool(self.alarms) or predicted

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "skipped": self.skipped,
            "requires_investigation": self.requires_investigation,
            "n_alarms": len(self.alarms),
            "n_watches": len(self.watches),
            "prediction_drift": self.prediction_drift.as_dict() if self.prediction_drift else None,
            # Ranked, because a 37-row table nobody reads is the same as no monitoring.
            "features": [f.as_dict() for f in sorted(self.features, key=lambda f: -f.psi)],
        }


class DriftMonitor:
    """Scores live batches against binned training distributions."""

    def __init__(self, reference: dict[str, Any], config: dict[str, Any] | None = None) -> None:
        cfg = (config or get_monitoring_config())["drift"]
        self.reference = reference
        self.watch_threshold = float(cfg["thresholds"]["watch"])
        self.alarm_threshold = float(cfg["thresholds"]["alarm"])
        self.min_rows = int(cfg["min_rows"])
        self.ignored = set(cfg.get("ignore") or [])
        self.monitor_predictions = bool(cfg.get("monitor_prediction_distribution", True))

    @classmethod
    def from_training_data(
        cls,
        features: pd.DataFrame,
        scores: np.ndarray | None = None,
        config: dict[str, Any] | None = None,
    ) -> DriftMonitor:
        """Freeze the reference distribution from the data the model was trained on."""
        cfg = (config or get_monitoring_config())["drift"]
        n_bins = int(cfg["n_bins"])
        reference: dict[str, Any] = {"n_rows": int(len(features)), "features": {}}

        for column in features.columns:
            reference["features"][column] = _bin_reference(features[column], n_bins)
        if scores is not None:
            reference["prediction"] = _bin_reference(pd.Series(scores, name="score"), n_bins)

        logger.info(
            "drift reference built",
            extra={"n_rows": reference["n_rows"], "n_features": len(reference["features"])},
        )
        return cls(reference, config)

    @classmethod
    def load(cls, directory: Path, config: dict[str, Any] | None = None) -> DriftMonitor:
        """Read a reference previously written by save()."""
        path = Path(directory) / REFERENCE_FILE
        return cls(json.loads(path.read_text(encoding="utf-8")), config)

    def save(self, directory: Path) -> Path:
        """Persist the reference beside the model it belongs to."""
        # Versioned with the model, not separately: a reference from a different training
        # run measures the wrong baseline and would report drift that never happened.
        path = Path(directory) / REFERENCE_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.reference, indent=2), encoding="utf-8")
        return path

    def compare(self, current: pd.DataFrame, scores: np.ndarray | None = None) -> DriftReport:
        """Score one batch of live traffic against the reference."""
        n_rows = len(current)
        if n_rows < self.min_rows:
            reason = f"batch of {n_rows} rows is below the {self.min_rows}-row minimum"
            logger.warning("drift comparison skipped", extra={"reason": reason})
            return DriftReport(n_rows=n_rows, skipped=reason)

        drifts = []
        for name, spec in self.reference["features"].items():
            if name in self.ignored or name not in current.columns:
                continue
            drifts.append(self._score(name, spec, current[name]))

        prediction = None
        if self.monitor_predictions and scores is not None and "prediction" in self.reference:
            prediction = self._score(
                "decline_risk_score", self.reference["prediction"], pd.Series(scores)
            )

        report = DriftReport(n_rows=n_rows, features=drifts, prediction_drift=prediction)
        logger.info(
            "drift comparison complete",
            extra={
                "n_rows": n_rows,
                "n_alarms": len(report.alarms),
                "n_watches": len(report.watches),
                "requires_investigation": report.requires_investigation,
            },
        )
        return report

    def _score(self, name: str, spec: dict[str, Any], values: pd.Series) -> FeatureDrift:
        current_share = _shares(values, spec)
        reference_share = np.asarray(spec["shares"], dtype=float)

        floor = 1.0 / (2 * max(len(values), 1))
        # A bin empty in one distribution makes the log term infinite. Flooring at half an
        # observation keeps the value finite and ties it to the sample size, rather than to
        # an arbitrary epsilon that would make PSI depend on a constant nobody chose.
        ref = np.maximum(reference_share, floor)
        cur = np.maximum(current_share, floor)
        contributions = (cur - ref) * np.log(cur / ref)
        psi = float(contributions.sum())

        worst = int(np.argmax(np.abs(contributions)))
        return FeatureDrift(
            feature=name,
            psi=psi,
            status=self._status(psi),
            largest_shift_bin=spec["labels"][worst],
            reference_share=float(reference_share[worst]),
            current_share=float(current_share[worst]),
        )

    def _status(self, psi: float) -> str:
        if psi >= self.alarm_threshold:
            return ALARM
        return WATCH if psi >= self.watch_threshold else STABLE


def build_reference(directory: Path | None = None) -> Path:
    """Rebuild the drift reference from the training data and the production model."""
    from src.data.loader import load_raw
    from src.models.registry import load_production_model, load_transformers

    preprocessor, engineer = load_transformers()
    features = engineer.transform(preprocessor.transform(load_raw()))
    features = features.drop(columns=["status"], errors="ignore")

    model, card = load_production_model()
    ordered = features[card.feature_names]
    scores = model.predict_proba(ordered.astype(float))[:, 1]

    target = Path(directory) if directory else get_settings().model_path
    monitor = DriftMonitor.from_training_data(ordered, scores)
    monitor.reference["model_version"] = card.version
    return monitor.save(target)


def _bin_reference(values: pd.Series, n_bins: int) -> dict[str, Any]:
    """Freeze bin edges and reference shares for one column."""
    clean = pd.to_numeric(values, errors="coerce").dropna()
    distinct = np.unique(clean)

    if len(distinct) <= CATEGORICAL_MAX_CARDINALITY:
        counts = pd.Series(clean).value_counts().reindex(distinct, fill_value=0)
        return {
            "kind": "categorical",
            "values": [float(v) for v in distinct],
            "labels": [f"={v:g}" for v in distinct],
            "shares": (counts.to_numpy(dtype=float) / max(len(clean), 1)).tolist(),
        }

    quantiles = np.linspace(0, 1, n_bins + 1)
    # Duplicate edges collapse when a value dominates, which is common here: several ratios
    # are zero for most rows. Keeping the deduplicated edges yields fewer, honest bins.
    edges = np.unique(np.quantile(clean, quantiles))
    edges[0], edges[-1] = -np.inf, np.inf
    counts, _ = np.histogram(clean, bins=edges)
    return {
        "kind": "numeric",
        "edges": edges.tolist(),
        "labels": [f"({edges[i]:g}, {edges[i + 1]:g}]" for i in range(len(edges) - 1)],
        "shares": (counts / max(counts.sum(), 1)).tolist(),
    }


def _shares(values: pd.Series, spec: dict[str, Any]) -> np.ndarray:
    """Proportion of a live batch falling in each frozen reference bin."""
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) == 0:
        return np.zeros(len(spec["shares"]))

    if spec["kind"] == "categorical":
        counts = pd.Series(clean).value_counts()
        # A level unseen in training lands in no bin, so the shares sum below one. That is
        # deliberate: it inflates PSI, which is the correct reaction to an unknown category.
        return np.array([float(counts.get(v, 0)) for v in spec["values"]]) / len(clean)

    counts, _ = np.histogram(clean, bins=np.array(spec["edges"], dtype=float))
    return counts / max(counts.sum(), 1)
