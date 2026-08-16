"""Tests for src.robustness_experiment.

Two things are worth locking down here, and neither is about the numbers
being good:

* The corruption/healing code must stay **out** of the main pipeline.
  It was the headline result once, and nothing should quietly reattach it.
* The measured degradation must be reported, not tuned away. A test that
  asserted "healing improves fidelity" would create pressure to adjust
  the compensator until it passed; these tests assert the *reporting* is
  honest, and deliberately do not assert that healing works.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import MissingDataConfig, SensorDriftConfig
from src.robustness_experiment import (
    IMPUTATION_COLUMNS,
    TARGET_COLUMN,
    build_variants,
    evaluate_variants,
    measure_signal_quality,
    summarise_variants,
)


# ── Fixtures ──────────────────────────────────────────────────────────


def _frame(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2022-07-12 04:00", periods=n, freq="1h")
    hours = np.arange(n)
    moisture = 55.0 + 18.0 * np.sin(hours / 22.0) + rng.normal(0, 1.2, n)
    return pd.DataFrame({
        "timestamp": ts,
        "soil_moisture": moisture,
        "air_temp": 22.0 + 9.0 * np.sin(hours / 24.0 * 2 * np.pi),
        "humidity": rng.uniform(40, 90, n),
        "wind_speed": rng.uniform(0, 8, n),
        "solar_radiation": np.clip(
            800 * np.cos(2 * np.pi * (ts.hour - 16) / 24.0), 0, None
        ),
        "irrigation_event": (moisture < 45).astype(float),
        "flow_l": rng.uniform(0, 50, n),
        "flow_l_cumulative": np.cumsum(rng.uniform(0, 50, n)),
    })


@pytest.fixture(scope="module")
def variants() -> dict:
    """Variants at a missingness rate the lag design can survive.

    Deliberately 5 % MCAR, not the 20 % the experiment defaults to. A row
    of the causal design matrix needs all 24 preceding hours present, so
    under MCAR its survival probability is ``(1 − rate)^25``: 28 % at
    5 %, but 0.4 % at 20 %, which leaves a synthetic frame with one
    usable row and nothing to test. The real experiment uses
    heat-dependent missingness, which clusters into runs and so destroys
    far fewer rows for the same nominal rate.
    """
    return build_variants(
        _frame(n=900),
        missing_config=MissingDataConfig(rate=0.05, mechanism="mcar", seed=42),
        drift_config=SensorDriftConfig(seed=42),
    )


# ── Separation from the main pipeline ────────────────────────────────


class TestSeparationFromMainPipeline:
    def test_main_pipeline_does_not_import_corruption_or_healing(self) -> None:
        """The experiment must not creep back into the headline result."""
        source = (
            pd.read_csv  # noqa: F841  (keeps the import list honest)
        )
        text = open("src/evaluate_pipeline.py", encoding="utf-8").read()
        assert "data_corruption" not in text
        assert "data_healing" not in text

    def test_main_pipeline_has_no_scenario_vocabulary(self) -> None:
        text = open("src/evaluate_pipeline.py", encoding="utf-8").read()
        for token in ("create_scenarios", "df_corrupted", "df_healed"):
            assert token not in text

    def test_corruption_and_healing_modules_still_exist(self) -> None:
        """Relocated, not deleted."""
        import src.data_corruption  # noqa: F401
        import src.data_healing  # noqa: F401


# ── Variant construction ─────────────────────────────────────────────


class TestBuildVariants:
    def test_produces_all_four_variants(self, variants) -> None:
        assert set(variants) == {
            "clean", "corrupted", "healed_mice",
            "healed_mice_plus_compensator",
        }

    def test_healing_stages_are_kept_separate(self, variants) -> None:
        """Collapsing them into one 'healed' variant hid the damage."""
        mice = variants["healed_mice"][TARGET_COLUMN]
        compensated = variants["healed_mice_plus_compensator"][TARGET_COLUMN]
        assert not np.allclose(mice, compensated)

    def test_corruption_introduces_missing_values(self, variants) -> None:
        clean_nan = variants["clean"][TARGET_COLUMN].isna().sum()
        corrupted_nan = variants["corrupted"][TARGET_COLUMN].isna().sum()
        assert corrupted_nan > clean_nan

    def test_imputation_fills_every_gap(self, variants) -> None:
        assert variants["healed_mice"][TARGET_COLUMN].isna().sum() == 0

    def test_clean_variant_is_untouched(self, variants) -> None:
        original = _frame(n=900)[TARGET_COLUMN]
        np.testing.assert_allclose(
            variants["clean"][TARGET_COLUMN].to_numpy(), original.to_numpy(),
        )

    def test_imputation_excludes_target_and_flow(self) -> None:
        """MICE must not reconstruct a sensor from the label or the meter."""
        assert "irrigation_event" not in IMPUTATION_COLUMNS
        assert not [c for c in IMPUTATION_COLUMNS if c.startswith("flow_l")]


# ── Signal quality ───────────────────────────────────────────────────


class TestMeasureSignalQuality:
    def test_reference_variant_is_perfect_by_definition(
        self, variants,
    ) -> None:
        table = measure_signal_quality(variants).set_index("variant")
        assert table.loc["clean", "correlation_with_truth"] == pytest.approx(
            1.0
        )
        assert table.loc["clean", "mae"] == pytest.approx(0.0)

    def test_row_per_variant(self, variants) -> None:
        assert len(measure_signal_quality(variants)) == len(variants)

    def test_reports_the_measured_degradation_without_adjustment(
        self, variants, caplog,
    ) -> None:
        """The finding must surface, and must not be tuned away.

        This asserts the *warning* fires when healing reduces fidelity —
        not that healing works. Asserting the latter would turn a
        measurement into a target.
        """
        with caplog.at_level("WARNING"):
            table = measure_signal_quality(variants).set_index("variant")

        corrupted = table.loc["corrupted", "correlation_with_truth"]
        healed = table.loc[
            "healed_mice_plus_compensator", "correlation_with_truth"
        ]
        if healed < corrupted:
            assert "Healing REDUCED signal fidelity" in caplog.text
            assert "reported unadjusted" in caplog.text

    def test_correlations_stay_in_range(self, variants) -> None:
        table = measure_signal_quality(variants)
        finite = table["correlation_with_truth"].dropna()
        assert (finite >= -1.0).all() and (finite <= 1.0).all()

    def test_compares_only_mutually_present_rows(self, variants) -> None:
        """A variant is not penalised for the NaN it was handed."""
        table = measure_signal_quality(variants).set_index("variant")
        assert table.loc["corrupted", "n_compared"] < table.loc[
            "clean", "n_compared"
        ]


# ── Fair comparison ──────────────────────────────────────────────────


class TestEvaluateVariants:
    @pytest.fixture(scope="class")
    def as_available(self, variants) -> pd.DataFrame:
        return evaluate_variants(
            variants, model_name="xgboost", restrict_to_common_rows=False,
        )

    @pytest.fixture(scope="class")
    def common_rows(self, variants) -> pd.DataFrame:
        return evaluate_variants(
            variants, model_name="xgboost", restrict_to_common_rows=True,
        )

    def test_as_available_sizes_differ(self, as_available) -> None:
        """Corruption removes whole rows — that is a real cost."""
        assert as_available.groupby("variant")["n_rows"].first().nunique() > 1

    def test_common_rows_equalises_sample_size(self, common_rows) -> None:
        """Otherwise a 4x sample-size gap masquerades as a drift effect."""
        assert common_rows.groupby("variant")["n_rows"].first().nunique() == 1

    def test_comparison_mode_is_labelled(
        self, as_available, common_rows,
    ) -> None:
        assert (as_available["comparison"] == "as_available").all()
        assert (common_rows["comparison"] == "common_rows").all()

    def test_common_rows_never_exceeds_as_available(
        self, as_available, common_rows,
    ) -> None:
        assert (
            common_rows["n_rows"].max() <= as_available["n_rows"].max()
        )

    def test_every_variant_is_evaluated(self, common_rows, variants) -> None:
        assert set(common_rows["variant"]) == set(variants)


class TestSummariseVariants:
    def test_carries_intervals_for_every_metric(self, variants) -> None:
        results = evaluate_variants(
            variants, model_name="xgboost", restrict_to_common_rows=True,
        )
        summary = summarise_variants(results)
        for metric in ("roc_auc", "pr_auc", "f1"):
            for suffix in ("mean", "std", "ci_low", "ci_high"):
                assert f"{metric}_{suffix}" in summary.columns

    def test_keeps_comparison_mode_in_the_summary(self, variants) -> None:
        results = evaluate_variants(
            variants, model_name="xgboost", restrict_to_common_rows=True,
        )
        summary = summarise_variants(results)
        assert "comparison" in summary.columns
        assert (summary["comparison"] == "common_rows").all()
