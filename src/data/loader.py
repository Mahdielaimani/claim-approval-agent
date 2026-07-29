"""Dataset loading and structural validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.common.logger import get_logger
from src.common.settings import get_settings, get_training_config

logger = get_logger(__name__)

EXPECTED_COLUMNS: tuple[str, ...] = (
    "excessFee",
    "rrp",
    "balanceRRP",
    "oldBalanceRRP",
    "productName",
    "productDesc",
    "coverage",
    "productCode",
    "policyStartDate",
    "policyEndDate",
    "policyStatus",
    "retailerName",
    "deviceType",
    "make",
    "model",
    "purchaseDate",
    "deviceCost",
    "relationship",
    "channel",
    "claimType",
    "country",
    "status",
    "turnOnOff",
    "touchScreen",
    "smashed",
    "frontCamera",
    "backCamera",
    "frontOrBackCamera",
    "audio",
    "mic",
    "buttons",
    "connection",
    "charging",
    "other",
    "issueDesc",
)

EXPECTED_TARGET_CLASSES: frozenset[str] = frozenset({"Completed", "Declined"})


class DataLoadError(RuntimeError):
    """Raised when the dataset is missing, empty, or does not match the schema."""


@dataclass
class DatasetProfile:
    """Structural and class-balance measurements taken at load time."""

    n_rows: int
    n_cols: int
    target_counts: dict[str, int]
    imbalance_ratio: float
    missing_columns: list[str] = field(default_factory=list)
    unexpected_columns: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """One-line description of shape and class balance."""
        maj, mino = max(self.target_counts.values()), min(self.target_counts.values())
        return (
            f"{self.n_rows} rows x {self.n_cols} cols | "
            f"{self.target_counts} | imbalance {maj}:{mino} = "
            f"{self.imbalance_ratio:.2f}:1"
        )


def load_raw(path: str | Path | None = None, *, strict: bool = True) -> pd.DataFrame:
    """Read the source dataset and validate its structure."""
    settings = get_settings()
    resolved = settings.resolve(Path(path) if path else settings.data_path)

    if not resolved.exists():
        raise DataLoadError(
            f"Dataset not found at {resolved}. The file is proprietary and "
            f"git-ignored; place it there manually."
        )

    df = pd.read_excel(resolved)
    if df.empty:
        raise DataLoadError(f"Dataset at {resolved} is empty.")

    profile = profile_dataset(df)
    # A renamed column would otherwise train the model on the wrong features, silently.
    if strict and profile.missing_columns:
        raise DataLoadError(
            f"Schema mismatch — expected columns absent: {profile.missing_columns}."
        )

    logger.info(
        "dataset loaded",
        extra={
            "path": str(resolved),
            "rows": profile.n_rows,
            "cols": profile.n_cols,
            "target_counts": profile.target_counts,
            "imbalance_ratio": round(profile.imbalance_ratio, 2),
        },
    )
    return df


def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    """Measure shape, class balance, and schema drift without modifying the frame."""
    cfg = get_training_config()
    target = cfg["data"]["target_column"]

    if target not in df.columns:
        raise DataLoadError(f"Target column '{target}' missing from the dataset.")

    counts = {str(k): int(v) for k, v in df[target].value_counts(dropna=False).items()}

    unknown = set(counts) - EXPECTED_TARGET_CLASSES
    if unknown:
        raise DataLoadError(
            f"Unexpected target classes {sorted(unknown)}; expected exactly "
            f"{sorted(EXPECTED_TARGET_CLASSES)}."
        )

    minority = min(counts.values()) or 1
    return DatasetProfile(
        n_rows=len(df),
        n_cols=df.shape[1],
        target_counts=counts,
        imbalance_ratio=max(counts.values()) / minority,
        missing_columns=[c for c in EXPECTED_COLUMNS if c not in df.columns],
        unexpected_columns=[c for c in df.columns if c not in EXPECTED_COLUMNS],
    )


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Null share, cardinality and dtype per column, ordered by missingness."""
    return (
        pd.DataFrame(
            {
                "null_count": df.isna().sum(),
                "null_pct": (df.isna().mean() * 100).round(2),
                "n_unique": df.nunique(dropna=True),
                "dtype": df.dtypes.astype(str),
            }
        )
        .sort_values("null_pct", ascending=False)
        .rename_axis("column")
        .reset_index()
    )
