"""Cleaning: makes the raw extract correct. Derivation lives in feature_engineer.py."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.common.logger import get_logger
from src.common.settings import get_training_config

logger = get_logger(__name__)

_MOJIBAKE_PATTERN = re.compile(r"Ã.|Â.|â€")

TEXT_COLUMNS: tuple[str, ...] = ("issueDesc", "other", "productDesc", "model")


def repair_mojibake(value: Any, max_passes: int = 3) -> Any:
    """Undo UTF-8 text that was decoded as cp1252, up to max_passes times."""
    if not isinstance(value, str):
        return value
    out = value
    for _ in range(max_passes):
        try:
            # cp1252, not latin-1: double-encoded rows contain U+0192, outside latin-1.
            candidate = out.encode("cp1252", errors="strict").decode("utf-8", errors="strict")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break
        if candidate == out:
            break
        out = candidate
    return out


def count_mojibake(series: pd.Series) -> int:
    """Number of non-null values still showing corruption markers."""
    return int(series.dropna().astype(str).map(lambda s: bool(_MOJIBAKE_PATTERN.search(s))).sum())


@dataclass
class PreprocessArtifacts:
    """Imputation values learned during fit, persisted alongside the model."""

    excess_fee_median_by_product: dict[str, float] = field(default_factory=dict)
    excess_fee_global_median: float = 0.0
    dropped_columns: list[str] = field(default_factory=list)
    fx_rate_sek_eur: float = 1.0


class Preprocessor:
    """Stateful cleaner; fit on training rows only, transform on any rows."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or get_training_config()
        self.artifacts = PreprocessArtifacts()
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> Preprocessor:
        """Learn imputation values from the training split."""
        work = self._repair_text(df.copy())
        work = self._normalise_currency(work)

        if {"excessFee", "productCode"} <= set(work.columns):
            grouped = work.groupby("productCode")["excessFee"].median()
            self.artifacts.excess_fee_median_by_product = {
                str(k): float(v) for k, v in grouped.dropna().items()
            }
        self.artifacts.excess_fee_global_median = float(
            work["excessFee"].median() if "excessFee" in work.columns else 0.0
        )
        self.artifacts.fx_rate_sek_eur = float(self.config["currency"]["sek_to_eur_rate"])
        self._fitted = True

        logger.info(
            "preprocessor fitted",
            extra={
                "product_medians": len(self.artifacts.excess_fee_median_by_product),
                "global_median": self.artifacts.excess_fee_global_median,
                "fx_rate": self.artifacts.fx_rate_sek_eur,
            },
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the cleaning steps without mutating the input frame."""
        if not self._fitted:
            raise RuntimeError("Preprocessor.transform called before fit.")

        out = df.copy()
        out = self._repair_text(out)
        out = self._normalise_currency(out)
        out = self._drop_columns(out)
        out = self._parse_dates(out)
        out = self._fix_anomalies(out)
        out = self._flag_diagnostics(out)
        out = self._fill_categoricals(out)
        out = self._impute_numeric(out)
        out = self._encode_target(out)

        logger.info("preprocessing complete", extra={"rows": len(out), "cols": out.shape[1]})
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on the frame, then transform it."""
        return self.fit(df).transform(df)

    def _repair_text(self, df: pd.DataFrame) -> pd.DataFrame:
        repaired: list[dict[str, Any]] = []
        for col in TEXT_COLUMNS:
            if col not in df.columns:
                continue
            before = count_mojibake(df[col])
            if before:
                df[col] = df[col].map(repair_mojibake)
                # Reported as {"column": name}: the log redactor blanks values under PII keys.
                repaired.append({"column": col, "rows_fixed": before - count_mojibake(df[col])})
        if repaired:
            logger.info("text encoding repaired", extra={"repairs": repaired})
        return df

    def _normalise_currency(self, df: pd.DataFrame) -> pd.DataFrame:
        # SE files in SEK, NL/FI in EUR; uncorrected, amount proxies for country and splits SHAP.
        cur = self.config["currency"]
        rate = float(cur["sek_to_eur_rate"])
        sek_countries = set(cur["countries_in_sek"])
        columns = [c for c in cur["columns"] if c in df.columns]

        if "country" not in df.columns or not columns:
            return df

        mask = df["country"].isin(sek_countries)
        for col in columns:
            df.loc[mask, col] = df.loc[mask, col] / rate
        # Converted too, so the drop step stays order-independent.
        if "oldBalanceRRP" in df.columns:
            df.loc[mask, "oldBalanceRRP"] = df.loc[mask, "oldBalanceRRP"] / rate

        # Kept for the audit trail: an auditor needs the currency actually filed.
        df["currency_original"] = np.where(mask, "SEK", "EUR")
        return df

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # productName stays: it encodes the missing device type, contract length and payment mode.
        clean = self.config["cleaning"]
        to_drop: list[str] = []
        for key in ("drop_constant", "drop_duplicate", "drop_degenerate", "drop_redundant_text"):
            to_drop.extend(clean.get(key, []) or [])

        present = [c for c in dict.fromkeys(to_drop) if c in df.columns]
        self.artifacts.dropped_columns = present
        return df.drop(columns=present)

    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        clean = self.config["cleaning"]
        for col in clean.get("date_columns", []):
            if col not in df.columns:
                continue
            # dayfirst suits NL/SE/FI; month-first would silently reinterpret any day <= 12.
            df[col] = pd.to_datetime(df[col], format="mixed", dayfirst=True, errors="coerce")
            unparsed = int(df[col].isna().sum())
            if unparsed:
                logger.warning("unparsed dates", extra={"column": col, "count": unparsed})
        return df

    def _fix_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        # NaN, not 0: XGBoost models missingness natively and 0 would distort every ratio.
        counts: dict[str, int] = {}
        if "rrp" in df.columns:
            bad = df["rrp"] <= 0
            counts["rrp_non_positive"] = int(bad.sum())
            df.loc[bad, "rrp"] = np.nan
        if "balanceRRP" in df.columns:
            bad = df["balanceRRP"] < 0
            counts["balance_negative"] = int(bad.sum())
            df.loc[bad, "balanceRRP"] = np.nan
        if any(counts.values()):
            logger.info("anomalies neutralised", extra=counts)
        return df

    def _flag_diagnostics(self, df: pd.DataFrame) -> pd.DataFrame:
        # Missing not-at-random: absence tracks intake channel, and 0 would assert "tested, failed".
        mv = self.config["missing_values"]
        sentinel = mv["diagnostic_sentinel"]
        for col in mv["diagnostic_columns"]:
            if col in df.columns:
                df[col] = df[col].fillna(sentinel).astype(int)
        return df

    def _fill_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        # Mode-filling retailerName would invent a retailer; UNKNOWN lets the model price it.
        mv = self.config["missing_values"]
        for col in mv["categorical_columns"]:
            if col in df.columns:
                df[col] = df[col].astype("object").fillna(mv["categorical_fill"])
        return df

    def _impute_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        if "excessFee" not in df.columns:
            return df

        missing = df["excessFee"].isna()
        if not missing.any():
            return df

        # The excess is a property of the policy, so productCode is the right reference class.
        if "productCode" in df.columns:
            df.loc[missing, "excessFee"] = (
                df.loc[missing, "productCode"]
                .astype(str)
                .map(self.artifacts.excess_fee_median_by_product)
            )

        # Medians come from fit; recomputing here would leak the serving distribution.
        df["excessFee"] = df["excessFee"].fillna(self.artifacts.excess_fee_global_median)
        return df

    def _encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        # Declined = 1 so precision and recall read against the rare, business-relevant class.
        cfg = self.config["data"]
        target = cfg["target_column"]
        if target in df.columns:
            df[target] = (df[target] == cfg["positive_class"]).astype(int)
        return df


def get_text_for_llm(raw_row: pd.Series) -> dict[str, str]:
    """Repaired free text for the explanation layer, held out of the feature matrix."""
    # Multilingual PII-masked prose: noise for the classifier, but useful to a human reader.
    return {
        "issue_desc": repair_mojibake(raw_row.get("issueDesc", "")) or "",
        "other": repair_mojibake(raw_row.get("other", "")) or "",
    }
