"""Optuna hyperparameter search for the strongest baseline candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold

from src.common.logger import get_logger
from src.common.settings import get_training_config
from src.models.evaluator import find_best_threshold
from src.models.tracking import log_params, training_run
from src.models.trainer import build_model

logger = get_logger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _space_xgboost(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        # Ranges start above the library defaults: 453 positives overfit at looser settings.
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 20.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
    }


def _space_random_forest(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 16),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
    }


def _space_catboost(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "iterations": trial.suggest_int("iterations", 200, 800, step=100),
        "depth": trial.suggest_int("depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.5, 5.0),
    }


def _space_lightgbm(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, 63),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 20.0, log=True),
    }


def _space_logistic_regression(trial: optuna.Trial) -> dict[str, Any]:
    return {"C": trial.suggest_float("C", 1e-3, 10.0, log=True)}


# Optuna only guarantees persistent storage for None, bool, int, float and str choices, and
# warns on anything else. Tuples are suggested by name and mapped back here.
_MLP_LAYERS: dict[str, tuple[int, ...]] = {
    "32": (32,),
    "64": (64,),
    "64-32": (64, 32),
    "128-64": (128, 64),
}


def _space_mlp(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "hidden_layer_sizes": _MLP_LAYERS[
            trial.suggest_categorical("hidden_layer_sizes", list(_MLP_LAYERS))
        ],
        "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
    }


SEARCH_SPACES = {
    "xgboost": _space_xgboost,
    "random_forest": _space_random_forest,
    "catboost": _space_catboost,
    "lightgbm": _space_lightgbm,
    "logistic_regression": _space_logistic_regression,
    "mlp": _space_mlp,
}


@dataclass
class TuningResult:
    """Outcome of one model's search."""

    model_name: str
    best_params: dict[str, Any]
    best_value: float
    metric: str
    n_trials: int
    trials: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> dict[str, Any]:
        """Row-shaped summary for reporting."""
        return {
            "model": self.model_name,
            f"tuned_{self.metric}": round(self.best_value, 4),
            "n_trials": self.n_trials,
            "best_params": self.best_params,
        }


def _score_fold(
    y_true: np.ndarray, y_prob: np.ndarray, y_train: np.ndarray, train_prob: np.ndarray, metric: str
) -> float:
    if metric == "pr_auc":
        return float(average_precision_score(y_true, y_prob))

    # Measurably worse than pr_auc: an in-sample threshold does not transfer. See training.yaml.
    threshold = find_best_threshold(y_train, train_prob)
    return float(f1_score(y_true, (y_prob >= threshold).astype(int), pos_label=1, zero_division=0))


def _objective(
    trial: optuna.Trial,
    name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    cfg: dict[str, Any],
    metric: str,
) -> float:
    params = SEARCH_SPACES[name](trial)
    folds = StratifiedKFold(
        n_splits=cfg["split"]["n_folds"], shuffle=True, random_state=cfg["data"]["random_state"]
    )

    scores: list[float] = []
    for step, (train_idx, val_idx) in enumerate(folds.split(X, y)):
        model = build_model(name, params=params, config=cfg)
        model.fit(X.iloc[train_idx], y[train_idx])

        val_prob = model.predict_proba(X.iloc[val_idx])[:, 1]
        train_prob = model.predict_proba(X.iloc[train_idx])[:, 1]
        scores.append(_score_fold(y[val_idx], val_prob, y[train_idx], train_prob, metric))

        # Per-fold reporting lets the pruner drop a losing config mid-way, not after five.
        trial.report(float(np.mean(scores)), step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(scores))


def tune(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_trials: int | None = None,
    metric: str | None = None,
    config: dict[str, Any] | None = None,
    track: bool = True,
) -> TuningResult:
    """Search one model's hyperparameters and return the best configuration."""
    cfg = config or get_training_config()
    tuning = cfg["tuning"]
    trials = n_trials if n_trials is not None else tuning["n_trials"]
    objective_metric = metric or tuning["optimize_metric"]

    if name not in SEARCH_SPACES:
        raise ValueError(f"No search space for '{name}'. Available: {sorted(SEARCH_SPACES)}")

    X_arr = X.reset_index(drop=True)
    y_arr = y.reset_index(drop=True).to_numpy()

    study = optuna.create_study(
        study_name=f"{name}-{objective_metric}",
        direction=tuning["direction"],
        sampler=TPESampler(seed=cfg["data"]["random_state"]),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )
    study.optimize(
        lambda t: _objective(t, name, X_arr, y_arr, cfg, objective_metric),
        n_trials=trials,
        show_progress_bar=False,
    )

    result = TuningResult(
        model_name=name,
        best_params=study.best_params,
        best_value=study.best_value,
        metric=objective_metric,
        n_trials=len(study.trials),
        trials=study.trials_dataframe(attrs=("number", "value", "state", "params", "duration")),
    )

    if track:
        _log_study(result)

    logger.info(
        "tuning complete",
        extra={
            "model": name,
            "metric": objective_metric,
            "best_value": round(result.best_value, 4),
            "trials": result.n_trials,
            "pruned": int((result.trials["state"] == "PRUNED").sum()),
        },
    )
    return result


def _log_study(result: TuningResult) -> None:
    import mlflow

    with training_run(
        f"tune-{result.model_name}", tags={"model": result.model_name, "stage": "tuning"}
    ):
        log_params({"model": result.model_name, "n_trials": result.n_trials, **result.best_params})
        mlflow.log_metric(f"best_{result.metric}", result.best_value)
        # One table, not 50 runs per model: that would bury the six baselines in the UI.
        mlflow.log_table(result.trials, artifact_file="trials.json")


def top_candidates(comparison: pd.DataFrame, *, config: dict[str, Any] | None = None) -> list[str]:
    """Names of the strongest baselines, as ranked by the comparison table."""
    cfg = config or get_training_config()
    return comparison["model"].head(cfg["tuning"]["tune_top_n"]).tolist()
