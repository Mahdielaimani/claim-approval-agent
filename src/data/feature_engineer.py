"""Feature derivation and encoding, producing the model-ready matrix."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.common.logger import get_logger
from src.common.settings import get_training_config

logger = get_logger(__name__)

DEVICE_TOKENS: dict[str, str] = {
    "SMARTPHONE": "SMARTPHONES",
    "SMARTPHONES": "SMARTPHONES",
    "TABLET": "TABLET",
    "TABLETS": "TABLET",
    "LAPTOP": "LAPTOP",
    "WEARABLE": "WEARABLES",
    "WEARABLES": "WEARABLES",
    "EARBUDS": "EARBUDS",
}

PAYMENT_TOKENS: dict[str, str] = {
    "MONTHLY": "MONTHLY",
    "UPFRONT": "UPFRONT",
    "UPFRONT1Y": "UPFRONT",
    "UPFRONT2Y": "UPFRONT",
}

# The same contract lengths are spelled three ways across the catalogue.
DURATION_TOKENS: dict[str, int] = {
    "3MTHFOC": 3,
    "6M": 6,
    "6MTHFOC": 6,
    "12M": 12,
    "1Y": 12,
    "12MTHFOC": 12,
    "UPFRONT1Y": 12,
    "24M": 24,
    "UPFRONT2Y": 24,
    "36MTHPID": 36,
}

POLICY_CLASS_TOKENS: frozenset[str] = frozenset({"MANDATORY", "VOLUNTARY", "DISCOUNT", "EPP"})

# Excluded from the matrix: currency_original duplicates country, year memorises the calendar.
NON_FEATURE_COLUMNS: tuple[str, ...] = ("currency_original", "policy_start_year")


def parse_product_name(name: Any) -> dict[str, Any]:
    """Decompose the composite product code into its four usable signals."""
    if not isinstance(name, str) or not name.strip():
        return {
            "device_type_parsed": None,
            "payment_mode": None,
            "contract_duration_months": None,
            "policy_class": None,
        }

    # Matched by membership, not position: the code runs 5 to 8 segments long.
    tokens = [t.strip().upper() for t in name.split("_")]
    return {
        "device_type_parsed": next((DEVICE_TOKENS[t] for t in tokens if t in DEVICE_TOKENS), None),
        "payment_mode": next((PAYMENT_TOKENS[t] for t in tokens if t in PAYMENT_TOKENS), None),
        "contract_duration_months": next(
            (DURATION_TOKENS[t] for t in tokens if t in DURATION_TOKENS), None
        ),
        "policy_class": next((t for t in tokens if t in POLICY_CLASS_TOKENS), None),
    }


@dataclass
class FeatureArtifacts:
    """Encoding maps learned during fit, persisted alongside the model."""

    frequency_maps: dict[str, dict[str, float]] = field(default_factory=dict)
    category_levels: dict[str, list[str]] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)


class FeatureEngineer:
    """Builds the model-ready matrix; fit on training rows only."""

    UNKNOWN = "UNKNOWN"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or get_training_config()
        self.artifacts = FeatureArtifacts()
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> FeatureEngineer:
        """Learn frequency maps and category orderings from the training split."""
        derived = self._derive(df.copy())

        for col in self.config["features"]["frequency_encoded"]:
            if col in derived.columns:
                self.artifacts.frequency_maps[col] = (
                    derived[col].astype(str).value_counts(normalize=True).to_dict()
                )

        # Sorted level lists so index N means the same category months later at serving.
        for col in self._ordinal_columns(derived):
            self.artifacts.category_levels[col] = sorted(
                derived[col].astype(str).dropna().unique().tolist()
            )

        self._fitted = True
        logger.info(
            "feature engineer fitted",
            extra={
                "frequency_encoded": list(self.artifacts.frequency_maps),
                "ordinal_encoded": list(self.artifacts.category_levels),
            },
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive and encode features using the maps learned at fit time."""
        if not self._fitted:
            raise RuntimeError("FeatureEngineer.transform called before fit.")

        out = self._finalise(self._encode(self._derive(df.copy())))

        if not self.artifacts.feature_names:
            self.artifacts.feature_names = [
                c for c in out.columns if c != self.config["data"]["target_column"]
            ]
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on the frame, then transform it."""
        return self.fit(df).transform(df)

    @property
    def feature_names(self) -> list[str]:
        """Feature column names, in matrix order."""
        return list(self.artifacts.feature_names)

    def _derive(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._from_product_name(df)
        df = self._ratios(df)
        df = self._date_features(df)
        df = self._diagnostic_meta(df)
        return self._text_meta(df)

    def _from_product_name(self, df: pd.DataFrame) -> pd.DataFrame:
        if "productName" not in df.columns:
            return df

        parsed = pd.DataFrame([parse_product_name(v) for v in df["productName"]], index=df.index)
        df = pd.concat([df, parsed], axis=1)

        # Backfill only: a stated device type is never overwritten by an inferred one.
        if "deviceType" in df.columns:
            stated = df["deviceType"].astype("object").where(df["deviceType"] != self.UNKNOWN)
            df["deviceType"] = stated.fillna(df["device_type_parsed"]).fillna(self.UNKNOWN)
        else:
            df["deviceType"] = df["device_type_parsed"].fillna(self.UNKNOWN)

        df["is_mandatory_policy"] = (df["policy_class"] == "MANDATORY").astype(int)
        df["policy_class"] = df["policy_class"].fillna("STANDARD")
        df["payment_mode"] = df["payment_mode"].fillna(self.UNKNOWN)
        df["contract_duration_months"] = pd.to_numeric(
            df["contract_duration_months"], errors="coerce"
        ).fillna(12)

        return df.drop(columns=["productName", "device_type_parsed"], errors="ignore")

    def _ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        # Currency-invariant by construction, so they survive an imperfect FX rate.
        eps = 1e-9
        if {"excessFee", "rrp"} <= set(df.columns):
            df["fee_to_rrp_ratio"] = df["excessFee"] / (df["rrp"] + eps)
        if {"balanceRRP", "rrp"} <= set(df.columns):
            df["balance_ratio"] = df["balanceRRP"] / (df["rrp"] + eps)
            df["remaining_balance_ratio"] = (df["rrp"] - df["balanceRRP"]) / (df["rrp"] + eps)

        for col in ("fee_to_rrp_ratio", "balance_ratio", "remaining_balance_ratio"):
            if col in df.columns:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        return df

    def _date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Intervals, not timestamps: raw dates would teach the model 2022.
        cols = set(df.columns)
        if {"policyStartDate", "policyEndDate"} <= cols:
            df["policy_duration_days"] = (df["policyEndDate"] - df["policyStartDate"]).dt.days
            df["policy_start_month"] = df["policyStartDate"].dt.month
            df["policy_start_year"] = df["policyStartDate"].dt.year
        if {"purchaseDate", "policyStartDate"} <= cols:
            # A policy bought long after the device is a different risk profile.
            df["days_purchase_to_policy"] = (df["policyStartDate"] - df["purchaseDate"]).dt.days

        return df.drop(
            columns=["policyStartDate", "policyEndDate", "purchaseDate"], errors="ignore"
        )

    def _diagnostic_meta(self, df: pd.DataFrame) -> pd.DataFrame:
        mv = self.config["missing_values"]
        cols = [c for c in mv["diagnostic_columns"] if c in df.columns]
        if not cols:
            return df

        block = df[cols]
        # How much evidence exists about the device is predictive in its own right.
        df["diag_completeness"] = (block != mv["diagnostic_sentinel"]).sum(axis=1)
        # Counted separately, not summed: the source never documents which value means broken.
        df["diag_ones"] = (block == 1).sum(axis=1)
        df["diag_zeros"] = (block == 0).sum(axis=1)
        return df

    def _text_meta(self, df: pd.DataFrame) -> pd.DataFrame:
        # Length is language-independent, unlike the narrative itself.
        if "issueDesc" in df.columns:
            text = df["issueDesc"].fillna("").astype(str)
            df["issue_desc_length"] = text.str.len()
            df["issue_desc_word_count"] = text.str.split().str.len().fillna(0)
        if "other" in df.columns:
            df["has_other_notes"] = df["other"].notna().astype(int)

        hold_out = self.config["cleaning"].get("hold_out_text", [])
        return df.drop(columns=[c for c in hold_out if c in df.columns])

    def _ordinal_columns(self, df: pd.DataFrame) -> list[str]:
        configured = list(self.config["features"]["label_encoded"])
        extra = ["policy_class", "payment_mode"]
        return [c for c in dict.fromkeys(configured + extra) if c in df.columns]

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        # Frequency over one-hot for high-cardinality ids: rarity is a signal, 2880 rows is not.
        for col, mapping in self.artifacts.frequency_maps.items():
            if col in df.columns:
                df[f"{col}_freq"] = df[col].astype(str).map(mapping).fillna(0.0)
                df = df.drop(columns=[col])

        # Ordinal implies a false order: costs the linear baseline, costs trees nothing.
        for col, levels in self.artifacts.category_levels.items():
            if col in df.columns:
                index = {level: i for i, level in enumerate(levels)}
                # Unseen levels get an explicit state rather than crashing at serving time.
                df[col] = df[col].astype(str).map(index).fillna(-1).astype(int)

        return df

    def _finalise(self, df: pd.DataFrame) -> pd.DataFrame:
        target = self.config["data"]["target_column"]
        df = df.drop(columns=[c for c in NON_FEATURE_COLUMNS if c in df.columns])

        leftovers = [
            c for c in df.columns if c != target and not pd.api.types.is_numeric_dtype(df[c])
        ]
        if leftovers:
            logger.warning("non-numeric columns dropped", extra={"columns": leftovers})
            df = df.drop(columns=leftovers)

        if not self.config["features"].get("keep_absolute_amounts", True):
            df = df.drop(columns=[c for c in ("rrp", "excessFee", "balanceRRP") if c in df.columns])

        # Stable order prevents a silent feature mismatch between training and a single row.
        features = sorted(c for c in df.columns if c != target)
        return df[features + ([target] if target in df.columns else [])]
