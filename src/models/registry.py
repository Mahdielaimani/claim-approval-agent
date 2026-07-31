"""Model registry: register, promote, and load the production model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

from src.common.logger import get_logger
from src.common.settings import get_settings, get_training_config
from src.models.tracking import configure_mlflow

logger = get_logger(__name__)

# Aliases rather than stages: MLflow deprecated stages in 2.9 and removes them in 3.x.
STAGING = "staging"
PRODUCTION = "production"


@dataclass
class ModelCard:
    """What the API needs to know about the model it is serving."""

    model_name: str
    algorithm: str
    version: str
    run_id: str
    threshold: float
    feature_names: list[str]
    f1_declined: float
    recall_declined: float
    precision_declined: float
    roc_auc: float
    brier: float
    n_training_rows: int
    trained_at: str
    git_commit: str

    def to_json(self) -> str:
        """Serialised card, for the audit trail and the auditor persona."""
        return json.dumps(asdict(self), indent=2)


def _local_dir() -> Path:
    settings = get_settings()
    path = settings.resolve(settings.model_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def register(
    model: Any,
    *,
    algorithm: str,
    cv_result: Any,
    feature_names: list[str],
    n_training_rows: int,
    run_id: str | None = None,
    model_name: str | None = None,
) -> ModelCard:
    """Register a fitted model as a new version and write its card."""
    cfg = get_training_config()
    name = model_name or cfg["mlflow"]["model_name"]
    configure_mlflow()

    client = MlflowClient()
    active = mlflow.active_run()
    source_run = run_id or (active.info.run_id if active else None)
    if source_run is None:
        raise RuntimeError("register() needs an active MLflow run or an explicit run_id.")

    result = mlflow.register_model(f"runs:/{source_run}/{algorithm}", name)
    version = result.version

    card = ModelCard(
        model_name=name,
        algorithm=algorithm,
        version=str(version),
        run_id=source_run,
        # Averaged across folds: a threshold chosen on the full dataset does not transfer.
        threshold=round(cv_result.mean("threshold"), 4),
        feature_names=feature_names,
        f1_declined=round(cv_result.mean("f1_declined"), 4),
        recall_declined=round(cv_result.mean("recall_declined"), 4),
        precision_declined=round(cv_result.mean("precision_declined"), 4),
        roc_auc=round(cv_result.mean("roc_auc"), 4),
        brier=round(cv_result.mean("brier"), 4),
        n_training_rows=n_training_rows,
        trained_at=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        git_commit=str(client.get_run(source_run).data.tags.get("git_commit", "unknown")),
    )

    # Written to disk too: the container must serve without reaching a tracking server.
    (_local_dir() / "model_card.json").write_text(card.to_json(), encoding="utf-8")
    joblib.dump(model, _local_dir() / "model.joblib")
    client.set_model_version_tag(name, version, "algorithm", algorithm)
    client.set_model_version_tag(name, version, "threshold", str(card.threshold))

    logger.info(
        "model registered",
        extra={
            "model": name,
            "version": version,
            "algorithm": algorithm,
            "f1_declined": card.f1_declined,
            "threshold": card.threshold,
        },
    )
    return card


def save_transformers(preprocessor: Any, feature_engineer: Any) -> Path:
    """Persist the fitted preprocessing state beside the model."""
    # Without these the model is unusable: a serving row needs the medians and maps from fit.
    path = _local_dir() / "transformers.joblib"
    joblib.dump({"preprocessor": preprocessor, "feature_engineer": feature_engineer}, path)
    logger.info("transformers saved", extra={"path": str(path)})
    return path


def load_transformers() -> tuple[Any, Any]:
    """Load the fitted preprocessor and feature engineer."""
    path = _local_dir() / "transformers.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No transformers at {path}. Train and register a model first.")
    bundle = joblib.load(path)
    return bundle["preprocessor"], bundle["feature_engineer"]


def promote(version: str, alias: str = PRODUCTION, *, model_name: str | None = None) -> None:
    """Point an alias at a version, moving it off whatever held it before."""
    cfg = get_training_config()
    name = model_name or cfg["mlflow"]["model_name"]
    configure_mlflow()

    # An alias resolves to one version and reassigns atomically — no archive step to forget.
    MlflowClient().set_registered_model_alias(name, alias, version)
    logger.info("model promoted", extra={"model": name, "version": version, "alias": alias})


def load_production_model(model_name: str | None = None) -> tuple[Any, ModelCard]:
    """Load the Production model and its card, falling back to local artifacts."""
    cfg = get_training_config()
    name = model_name or cfg["mlflow"]["model_name"]

    try:
        configure_mlflow()
        model = mlflow.sklearn.load_model(f"models:/{name}@{PRODUCTION}")
        logger.info("model loaded from registry", extra={"model": name, "alias": PRODUCTION})
    except Exception as exc:
        # A registry outage must not take down claim decisioning.
        logger.warning("registry unavailable, using local artifact", extra={"error": str(exc)})
        model = joblib.load(_local_dir() / "model.joblib")

    return model, load_model_card()


def load_model_card() -> ModelCard:
    """Read the card written alongside the local model artifact."""
    path = _local_dir() / "model_card.json"
    if not path.exists():
        raise FileNotFoundError(f"No model card at {path}. Train and register a model first.")
    return ModelCard(**json.loads(path.read_text(encoding="utf-8")))


def list_versions(model_name: str | None = None) -> pd.DataFrame:
    """Registered versions with their aliases and tags."""
    cfg = get_training_config()
    name = model_name or cfg["mlflow"]["model_name"]
    configure_mlflow()

    client = MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{name}'")
    except Exception:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "version": v.version,
                "aliases": ",".join(v.aliases) if v.aliases else "",
                "algorithm": v.tags.get("algorithm", ""),
                "threshold": v.tags.get("threshold", ""),
                "run_id": v.run_id,
            }
            for v in versions
        ]
    ).sort_values("version", key=lambda s: s.astype(int), ascending=False)
