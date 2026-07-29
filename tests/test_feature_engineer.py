"""Tests for feature derivation and encoding."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.feature_engineer import (
    NON_FEATURE_COLUMNS,
    FeatureEngineer,
    parse_product_name,
)
from src.data.preprocessor import Preprocessor


@pytest.fixture
def clean_frame(raw_frame) -> pd.DataFrame:
    """Preprocessed frame, the input the feature engineer actually receives."""
    return Preprocessor().fit_transform(raw_frame)


class TestProductNameParsing:
    def test_canonical_six_segment_code(self):
        assert parse_product_name("SE_MANDATORY_ADLD_24M_UPFRONT_SMARTPHONE") == {
            "device_type_parsed": "SMARTPHONES",
            "payment_mode": "UPFRONT",
            "contract_duration_months": 24,
            "policy_class": "MANDATORY",
        }

    @pytest.mark.parametrize(
        "code,expected_device",
        [
            ("NL_ADLD_12M_MONTHLY_TABLET", "TABLET"),
            ("SE_MANDATORY_ADLD_6M_UPFRONT_WEARABLES_SENA-ALWAYS-ON", "WEARABLES"),
            ("NL_ADLD_12M_MONTHLY_EARBUDS", "EARBUDS"),
            ("FI_ADLD_24M_UPFRONT_LAPTOP", "LAPTOP"),
        ],
    )
    def test_device_extracted_regardless_of_segment_count(self, code, expected_device):
        """A positional parser reads the wrong field on 40% of the catalogue."""
        assert parse_product_name(code)["device_type_parsed"] == expected_device

    @pytest.mark.parametrize(
        "code,months",
        [
            ("NL_VOLUNTARY_ADLD_1Y_UPFRONT_SMARTPHONE_Q4B4", 12),
            ("SE_ADLD_24M_UPFRONT_TABLET", 24),
            ("SE_MANDATORY_ADLD_6M_UPFRONT_SMARTPHONE_SENA-ALWAYS-ON", 6),
        ],
    )
    def test_duration_normalised_across_spellings(self, code, months):
        assert parse_product_name(code)["contract_duration_months"] == months

    def test_unparseable_input_returns_nones_not_guesses(self):
        for value in (None, "", np.nan, 12345):
            assert set(parse_product_name(value).values()) == {None}

    def test_device_type_missingness_is_eliminated(self, clean_frame):
        """48.4% missing in the real extract becomes 0% after parsing."""
        assert (clean_frame["deviceType"] == "UNKNOWN").sum() > 0

        derived = FeatureEngineer()._from_product_name(clean_frame.copy())
        assert (derived["deviceType"] == "UNKNOWN").sum() == 0

    def test_original_device_type_wins_over_the_parse(self, clean_frame):
        """Backfill only: a stated value is never overwritten by an inference."""
        frame = clean_frame.copy()
        frame.loc[frame.index[1], "deviceType"] = "SMARTPHONES"
        frame.loc[frame.index[1], "productName"] = "NL_ADLD_12M_MONTHLY_TABLET"

        derived = FeatureEngineer()._from_product_name(frame)
        assert derived.loc[derived.index[1], "deviceType"] == "SMARTPHONES"


class TestRatios:
    def test_ratios_are_currency_invariant(self):
        """This property is what makes them robust to an imperfect FX rate."""
        fe = FeatureEngineer()
        eur = pd.DataFrame({"excessFee": [59.0], "rrp": [1449.0], "balanceRRP": [1399.0]})
        sek = eur * 10.3

        assert fe._ratios(eur.copy())["fee_to_rrp_ratio"].iloc[0] == pytest.approx(
            fe._ratios(sek.copy())["fee_to_rrp_ratio"].iloc[0], rel=1e-6
        )

    def test_division_by_missing_amount_stays_missing(self):
        frame = pd.DataFrame({"excessFee": [59.0], "rrp": [np.nan], "balanceRRP": [10.0]})
        assert FeatureEngineer()._ratios(frame)["fee_to_rrp_ratio"].isna().all()

    def test_no_infinities_reach_the_matrix(self, clean_frame):
        X = FeatureEngineer().fit_transform(clean_frame)
        assert not np.isinf(X.select_dtypes("number").to_numpy(dtype=float)).any()


class TestDateFeatures:
    def test_durations_replace_timestamps(self, clean_frame):
        X = FeatureEngineer().fit_transform(clean_frame)
        for raw in ("policyStartDate", "policyEndDate", "purchaseDate"):
            assert raw not in X.columns
        assert {"policy_duration_days", "days_purchase_to_policy"} <= set(X.columns)

    def test_duration_is_positive_and_plausible(self, clean_frame):
        X = FeatureEngineer().fit_transform(clean_frame)
        assert (X["policy_duration_days"] > 0).all()
        assert X["policy_duration_days"].max() < 365 * 3

    def test_calendar_year_is_excluded(self, clean_frame):
        """Year is a constant to any model serving next year's claims."""
        X = FeatureEngineer().fit_transform(clean_frame)
        assert "policy_start_year" not in X.columns
        assert "policy_start_month" in X.columns


class TestDiagnosticMeta:
    def test_completeness_counts_only_checks_that_ran(self, clean_frame):
        X = FeatureEngineer().fit_transform(clean_frame)
        assert X["diag_completeness"].between(0, 9).all()

    def test_values_are_counted_separately_not_summed(self, clean_frame):
        """No total_failures: the 0/1 polarity is undocumented in the source."""
        X = FeatureEngineer().fit_transform(clean_frame)
        assert {"diag_ones", "diag_zeros"} <= set(X.columns)
        assert "total_failures" not in X.columns
        assert (X["diag_ones"] + X["diag_zeros"] == X["diag_completeness"]).all()


class TestTextMeta:
    def test_narrative_metadata_kept_narrative_dropped(self, clean_frame):
        X = FeatureEngineer().fit_transform(clean_frame)
        assert {"issue_desc_length", "issue_desc_word_count"} <= set(X.columns)
        assert "issueDesc" not in X.columns

    def test_length_matches_the_repaired_text(self, clean_frame):
        X = FeatureEngineer().fit_transform(clean_frame)
        assert X["issue_desc_length"].iloc[0] == len(clean_frame["issueDesc"].iloc[0])


class TestEncoding:
    def test_high_cardinality_uses_frequency_not_one_hot(self, clean_frame):
        X = FeatureEngineer().fit_transform(clean_frame)
        assert "model_freq" in X.columns
        assert "model" not in X.columns
        assert not [c for c in X.columns if c.startswith("model_M")]

    def test_unseen_level_maps_to_an_explicit_state_not_a_crash(self, clean_frame):
        fe = FeatureEngineer().fit(clean_frame)
        novel = clean_frame.iloc[[0]].copy()
        novel["model"] = "MODEL_NEVER_SEEN"
        novel["claimType"] = "Meteor Strike"

        out = fe.transform(novel)
        assert out["model_freq"].iloc[0] == 0.0
        assert out["claimType"].iloc[0] == -1

    def test_frequency_map_learned_on_fit_rows_only(self, clean_frame):
        fe = FeatureEngineer().fit(clean_frame)
        learned = dict(fe.artifacts.frequency_maps["model"])
        fe.transform(clean_frame.iloc[[0]])
        assert fe.artifacts.frequency_maps["model"] == learned


class TestMatrixContract:
    def test_matrix_is_entirely_numeric(self, clean_frame):
        X = FeatureEngineer().fit_transform(clean_frame)
        assert all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns)

    def test_collinear_and_calendar_columns_excluded(self, clean_frame):
        X = FeatureEngineer().fit_transform(clean_frame)
        for col in NON_FEATURE_COLUMNS:
            assert col not in X.columns

    def test_target_is_last_and_features_are_ordered(self, clean_frame):
        X = FeatureEngineer().fit_transform(clean_frame)
        assert X.columns[-1] == "status"
        assert list(X.columns[:-1]) == sorted(X.columns[:-1])

    def test_single_row_encodes_identically_to_the_batch(self, clean_frame):
        """Train/serve parity; the defect this catches is silent and fatal."""
        fe = FeatureEngineer()
        batch = fe.fit_transform(clean_frame)
        single = fe.transform(clean_frame.iloc[[3]])

        assert list(single.columns) == list(batch.columns)
        pd.testing.assert_frame_equal(
            single.reset_index(drop=True),
            batch.iloc[[3]].reset_index(drop=True),
            check_dtype=False,
        )

    def test_transform_before_fit_raises(self, clean_frame):
        with pytest.raises(RuntimeError, match="before fit"):
            FeatureEngineer().transform(clean_frame)
