"""Tests for the cleaning stage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import (
    Preprocessor,
    count_mojibake,
    get_text_for_llm,
    repair_mojibake,
)


class TestMojibakeRepair:
    def test_single_encoding_pass(self):
        assert repair_mojibake("frÃ¥n") == "från"
        assert repair_mojibake("skÃ¤rmen") == "skärmen"

    def test_double_encoding_pass(self):
        """Doubly-corrupted rows are the ones a latin-1 repair silently skips."""
        assert repair_mojibake("vÃƒÂ¤tskeskador") == "vätskeskador"

    def test_clean_text_untouched(self):
        for value in ("Screen cracked", "från", "", "Nestevahinko keittiössä"):
            assert repair_mojibake(value) == value

    def test_non_strings_pass_through(self):
        assert repair_mojibake(None) is None
        assert repair_mojibake(3.5) == 3.5
        assert np.isnan(repair_mojibake(np.nan))

    def test_pipeline_repairs_the_narrative(self, raw_frame):
        """Corrupted text reaching the LLM is the regression this guards against."""
        assert count_mojibake(raw_frame["issueDesc"]) > 0

        cleaned = Preprocessor().fit_transform(raw_frame)
        assert count_mojibake(cleaned["issueDesc"]) == 0


class TestCurrencyNormalisation:
    def test_sek_rows_converted_eur_rows_untouched(self, raw_frame, training_config):
        rate = training_config["currency"]["sek_to_eur_rate"]
        out = Preprocessor().fit_transform(raw_frame)

        se = raw_frame["country"] == "SE"
        assert out.loc[se, "rrp"].max() == pytest.approx(14990.0 / rate, rel=1e-6)
        assert out.loc[~se, "rrp"].max() == pytest.approx(1449.0, rel=1e-6)

    def test_country_no_longer_predicted_by_amount(self, raw_frame):
        """The whole point of the conversion: amount must stop proxying for country."""
        se = raw_frame["country"] == "SE"
        before = raw_frame.loc[se, "rrp"].median() / raw_frame.loc[~se, "rrp"].median()
        assert before > 5

        out = Preprocessor().fit_transform(raw_frame)
        after = out.loc[se, "rrp"].median() / out.loc[~se, "rrp"].median()
        assert 0.8 < after < 1.25

    def test_original_currency_recorded(self, raw_frame):
        """An auditor still needs to see the currency actually filed."""
        out = Preprocessor().fit_transform(raw_frame)
        assert set(out["currency_original"].unique()) == {"SEK", "EUR"}


class TestColumnDrops:
    def test_constant_duplicate_and_degenerate_removed(self, raw_frame):
        out = Preprocessor().fit_transform(raw_frame)
        for col in (
            "deviceCost",
            "relationship",
            "make",
            "oldBalanceRRP",
            "smashed",
            "frontOrBackCamera",
            "productDesc",
        ):
            assert col not in out.columns

    def test_product_name_survives_cleaning(self, raw_frame):
        """It looks droppable; the feature engineer mines four signals out of it."""
        out = Preprocessor().fit_transform(raw_frame)
        assert "productName" in out.columns


class TestDateParsing:
    def test_both_source_formats_parse(self, raw_frame):
        out = Preprocessor().fit_transform(raw_frame)
        for col in ("policyStartDate", "policyEndDate", "purchaseDate"):
            assert pd.api.types.is_datetime64_any_dtype(out[col])
            assert out[col].isna().sum() == 0

    def test_day_first_interpretation(self, raw_frame):
        """Month-first would silently corrupt every date whose day is <= 12."""
        out = Preprocessor().fit_transform(raw_frame)
        parsed = out.loc[1, "policyStartDate"]
        assert (parsed.day, parsed.month) == (28, 2)


class TestAnomalies:
    def test_impossible_values_become_nan_not_zero(self, raw_frame):
        """NaN is a state the model understands; 0 is a lie about the device."""
        out = Preprocessor().fit_transform(raw_frame)
        assert out["rrp"].isna().sum() == 1
        assert (out["rrp"] <= 0).sum() == 0
        assert (out["balanceRRP"] < 0).sum() == 0

    def test_rows_are_not_deleted(self, raw_frame):
        out = Preprocessor().fit_transform(raw_frame)
        assert len(out) == len(raw_frame)


class TestDiagnostics:
    def test_sentinel_marks_never_assessed(self, raw_frame, training_config):
        sentinel = training_config["missing_values"]["diagnostic_sentinel"]
        out = Preprocessor().fit_transform(raw_frame)

        for col in training_config["missing_values"]["diagnostic_columns"]:
            if col in out.columns:
                assert out[col].isna().sum() == 0
                assert set(out[col].unique()) <= {sentinel, 0, 1}

    def test_sentinel_is_distinguishable_from_a_real_zero(self, raw_frame):
        """'Not tested' and 'tested, value 0' must not collapse into one state."""
        out = Preprocessor().fit_transform(raw_frame)
        assert {-1, 0} <= set(out["turnOnOff"].unique())


class TestFitTransformSeparation:
    def test_transform_before_fit_raises(self, raw_frame):
        with pytest.raises(RuntimeError, match="before fit"):
            Preprocessor().transform(raw_frame)

    def test_medians_come_from_fit_not_from_the_serving_batch(self, raw_frame):
        """A single claim must be imputed with the training median, not its own."""
        pre = Preprocessor().fit(raw_frame)
        learned = pre.artifacts.excess_fee_global_median

        one = raw_frame.iloc[[4]].copy()
        one["productCode"] = "NEVER_SEEN"
        out = pre.transform(one)

        assert out["excessFee"].iloc[0] == pytest.approx(learned)

    def test_input_frame_is_not_mutated(self, raw_frame):
        before = raw_frame.copy(deep=True)
        Preprocessor().fit_transform(raw_frame)
        pd.testing.assert_frame_equal(raw_frame, before)


class TestTarget:
    def test_declined_is_the_positive_class(self, raw_frame):
        out = Preprocessor().fit_transform(raw_frame)
        assert set(out["status"].unique()) <= {0, 1}
        assert out["status"].sum() == (raw_frame["status"] == "Declined").sum()


class TestTextForLLM:
    def test_text_is_repaired_for_the_llm(self, raw_frame):
        text = get_text_for_llm(raw_frame.iloc[0])
        assert "Ã" not in text["issue_desc"]
        assert "från" in text["issue_desc"]

    def test_missing_notes_become_empty_string(self, raw_frame):
        assert get_text_for_llm(raw_frame.iloc[0])["other"] == ""
