"""MLflow experiment tracking for training runs."""

from __future__ import annotations

import subprocess
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature

from src.common.logger import get_logger
from src.common.settings import get_settings, get_training_config

logger = get_logger(__name__)


def _git_commit() -> str:
    """Short hash of the current commit, or 'unknown' outside a repository."""
    # Pinned per run so a metric can always be traced back to the code that produced it.
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def configure_mlflow(experiment_name: str | None = None) -> str:
    """Point MLflow at the configured tracking store and experiment."""
    settings = get_settings()
    cfg = get_training_config()

    uri = settings.resolve(settings.mlflow_tracking_uri)
    uri.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri.as_uri())

    name = experiment_name or cfg["mlflow"]["experiment_name"]
    mlflow.set_experiment(name)
    return name


@contextmanager
def training_run(
    run_name: str,
    *,
    tags: dict[str, Any] | None = None,
    nested: bool = False,
) -> Generator[Any]:
    """Open an MLflow run tagged with the code version and dataset fingerprint."""
    configure_mlflow()
    base_tags = {"git_commit": _git_commit(), **(tags or {})}

    with mlflow.start_run(run_name=run_name, nested=nested, tags=base_tags) as run:
        logger.info("mlflow run started", extra={"run_name": run_name, "run_id": run.info.run_id})
        yield run


def log_dataset(X: pd.DataFrame, y: pd.Series) -> None:
    """Record the shape and class balance the run was trained on."""
    # Without this, a metric two months from now is unattributable to a data version.
    mlflow.log_params(
        {
            "n_rows": len(X),
            "n_features": X.shape[1],
            "n_declined": int(y.sum()),
            "imbalance_ratio": round(float((len(y) - y.sum()) / max(y.sum(), 1)), 3),
        }
    )
    mlflow.log_dict({"features": list(X.columns)}, "features.json")


def log_cv_result(result: Any, *, prefix: str = "") -> None:
    """Log cross-validation means, dispersion, and per-fold detail."""
    p = f"{prefix}_" if prefix else ""
    for metric in (
        "f1_declined",
        "recall_declined",
        "precision_declined",
        "roc_auc",
        "pr_auc",
        "brier",
        "threshold",
    ):
        mlflow.log_metric(f"{p}{metric}", result.mean(metric))
        # Spread beside every mean: on 453 positives, an average alone hides overfitting.
        mlflow.log_metric(f"{p}{metric}_std", result.std(metric))

    mlflow.log_table(result.fold_frame(), artifact_file=f"{p}folds.json")


def log_params(params: dict[str, Any]) -> None:
    """Record hyperparameters, coercing unsupported types to strings."""
    mlflow.log_params(
        {k: (v if isinstance(v, int | float | str | bool) else str(v)) for k, v in params.items()}
    )


def log_model(model: Any, name: str, *, signature_input: pd.DataFrame | None = None) -> str:
    """Store a fitted model as a run artifact and return its URI."""
    kwargs: dict[str, Any] = {"artifact_path": name}
    if signature_input is not None:
        # float64 throughout: MLflow rejects a serving row whose int column arrives as float.
        as_float = signature_input.astype("float64")
        kwargs["input_example"] = as_float.head(5)
        kwargs["signature"] = infer_signature(as_float)

    mlflow.sklearn.log_model(model, **kwargs)

    run = mlflow.active_run()
    if run is None:
        # Was an AttributeError on None: logging a model outside a run is a caller mistake,
        # and it should say so rather than fail on attribute access two frames deeper.
        raise RuntimeError("log_model() must be called inside an active MLflow run.")
    return f"runs:/{run.info.run_id}/{name}"


def log_figure(figure: Any, filename: str) -> None:
    """Attach a matplotlib figure to the active run."""
    mlflow.log_figure(figure, filename)


def best_run(experiment_name: str | None = None, metric: str = "f1_declined") -> pd.DataFrame:
    """Runs in the experiment, ranked by the given metric."""
    name = configure_mlflow(experiment_name)
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        return pd.DataFrame()

    return mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
    )
