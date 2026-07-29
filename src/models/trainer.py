"""Factory building the benchmark candidates from a name."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.common.logger import get_logger
from src.common.settings import get_training_config

logger = get_logger(__name__)

# Six candidates on identical folds, so each rejection rests on a number, not an opinion.
MODEL_NAMES: tuple[str, ...] = (
    "logistic_regression",
    "random_forest",
    "xgboost",
    "catboost",
    "lightgbm",
    "mlp",
)

# These two split on NaN directly, so "not assessed" survives intact into the model.
NATIVE_MISSING_SUPPORT: frozenset[str] = frozenset({"xgboost", "lightgbm"})


def _imbalance(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg["models"]["imbalance"]


def _logistic_regression(cfg: dict[str, Any], params: dict[str, Any]) -> Pipeline:
    # A model requirement, not a data one, so it stays out of feature engineering.
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight=_imbalance(cfg)["sklearn_class_weight"],
                    random_state=cfg["data"]["random_state"],
                    **params,
                ),
            ),
        ]
    )


def _random_forest(cfg: dict[str, Any], params: dict[str, Any]) -> Pipeline:
    defaults = {"n_estimators": 400, "min_samples_leaf": 2, "n_jobs": -1}
    return Pipeline(
        [
            # sklearn forests cannot split on NaN; a median would blur "not assessed" away.
            ("impute", SimpleImputer(strategy="constant", fill_value=-1)),
            (
                "clf",
                RandomForestClassifier(
                    class_weight=_imbalance(cfg)["sklearn_class_weight"],
                    random_state=cfg["data"]["random_state"],
                    **{**defaults, **params},
                ),
            ),
        ]
    )


def _xgboost(cfg: dict[str, Any], params: dict[str, Any]) -> XGBClassifier:
    defaults = {
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        # 2880 rows with 453 positives overfits readily; regularisation is not optional here.
        "reg_lambda": 2.0,
        "min_child_weight": 3,
    }
    return XGBClassifier(
        scale_pos_weight=_imbalance(cfg)["scale_pos_weight"],
        random_state=cfg["data"]["random_state"],
        eval_metric="logloss",
        n_jobs=-1,
        **{**defaults, **params},
    )


def _catboost(cfg: dict[str, Any], params: dict[str, Any]) -> CatBoostClassifier:
    defaults = {"iterations": 400, "depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 3.0}
    return CatBoostClassifier(
        auto_class_weights=_imbalance(cfg)["catboost"],
        random_seed=cfg["data"]["random_state"],
        verbose=False,
        allow_writing_files=False,
        **{**defaults, **params},
    )


def _lightgbm(cfg: dict[str, Any], params: dict[str, Any]) -> LGBMClassifier:
    defaults = {
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 2.0,
        # LightGBM grows leaf-wise and will happily isolate single claims at this sample size.
        "min_child_samples": 20,
        "num_leaves": 15,
    }
    return LGBMClassifier(
        scale_pos_weight=_imbalance(cfg)["scale_pos_weight"],
        random_state=cfg["data"]["random_state"],
        verbose=-1,
        n_jobs=-1,
        **{**defaults, **params},
    )


def _mlp(cfg: dict[str, Any], params: dict[str, Any]) -> ImbPipeline:
    defaults = {"hidden_layer_sizes": (64, 32), "alpha": 1e-3, "max_iter": 600}
    # MLPClassifier has no class_weight, so SMOTE resamples inside the pipeline, per fold.
    return ImbPipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("smote", SMOTE(random_state=cfg["data"]["random_state"], k_neighbors=5)),
            (
                "clf",
                MLPClassifier(
                    random_state=cfg["data"]["random_state"],
                    early_stopping=True,
                    n_iter_no_change=15,
                    **{**defaults, **params},
                ),
            ),
        ]
    )


_BUILDERS: dict[str, Callable[[dict[str, Any], dict[str, Any]], Any]] = {
    "logistic_regression": _logistic_regression,
    "random_forest": _random_forest,
    "xgboost": _xgboost,
    "catboost": _catboost,
    "lightgbm": _lightgbm,
    "mlp": _mlp,
}


def build_model(
    name: str,
    *,
    params: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> Any:
    """Construct an unfitted estimator by name."""
    if name not in _BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(_BUILDERS)}")
    cfg = config or get_training_config()
    return _BUILDERS[name](cfg, params or {})


def make_factory(
    name: str,
    *,
    params: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> Callable[[], Any]:
    """Return a zero-argument callable producing a fresh estimator."""

    # One fresh estimator per fold; reusing an instance would leak fitted state between folds.
    def factory() -> Any:
        return build_model(name, params=params, config=config)

    return factory


def candidate_names(config: dict[str, Any] | None = None) -> list[str]:
    """Benchmark candidates listed in the training config."""
    cfg = config or get_training_config()
    return list(cfg["models"]["candidates"])


def supports_native_missing(name: str) -> bool:
    """Whether the model splits on NaN directly instead of needing imputation."""
    return name in NATIVE_MISSING_SUPPORT
