"""Tests for the feature-ablation study in src.evaluate_pipeline.

The property that makes an ablation interpretable is that its sets differ
in *features only*.  If each set were rebuilt from scratch it would also
drop a different warm-up period, and a gap between two sets could then be
caused by sample size rather than by the features — so the shared-rows
invariant is tested explicitly, not assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluate_pipeline import (
    ABLATION_DESCRIPTIONS,
    make_model_factory,
    ABLATION_SETS,
    build_ablation_feature_sets,
    build_design_matrix,
    load_modeling_frame,
    run_ablation,
    summarise_ablation,
)
from src.features import (
    BLOCK_CALENDAR,
    BLOCK_IRRIGATION,
    BLOCK_MOISTURE,
    BLOCK_ORDER,
    BLOCK_WEATHER,
    build_features,
)
from src.validation import rolling_origin_splits


# ── Fixtures ──────────────────────────────────────────────────────────


def _synthetic_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2022-07-12 04:00", periods=n, freq="1h")
    hours = np.arange(n)
    moisture = 50.0 + 20.0 * np.sin(hours / 20.0) + rng.normal(0, 1.5, n)
    return pd.DataFrame({
        "timestamp": ts,
        "soil_moisture": moisture,
        "air_temp": 20.0 + 8.0 * np.sin(hours / 24.0 * 2 * np.pi),
        "humidity": rng.uniform(40, 90, n),
        "wind_speed": rng.uniform(0, 8, n),
        "solar_radiation": np.clip(
            800 * np.cos(2 * np.pi * (ts.hour - 16) / 24.0), 0, None
        ),
        "irrigation_event": (moisture < 40).astype(float),
        "flow_l": rng.uniform(0, 50, n),
        "flow_l_cumulative": np.cumsum(rng.uniform(0, 50, n)),
    })


@pytest.fixture(scope="module")
def blocks() -> dict[str, list[str]]:
    _, blocks = build_features(_synthetic_frame())
    return blocks


@pytest.fixture(scope="module")
def design():
    X, y, ts, blocks = build_design_matrix(_synthetic_frame())
    return X, y, ts, blocks


# ── Set composition ──────────────────────────────────────────────────


class TestAblationSetDefinitions:
    def test_all_five_sets_defined(self) -> None:
        assert set(ABLATION_SETS) == {"A", "B", "C", "D", "E"}
        assert set(ABLATION_DESCRIPTIONS) == set(ABLATION_SETS)

    def test_sets_a_to_d_are_nested(self, blocks) -> None:
        """A ⊂ B ⊂ C ⊂ D, so each step adds exactly one block."""
        sets = build_ablation_feature_sets(blocks)
        for smaller, larger in [("A", "B"), ("B", "C"), ("C", "D")]:
            assert set(sets[smaller]) < set(sets[larger])

    def test_set_d_is_the_complete_feature_set(self, blocks) -> None:
        sets = build_ablation_feature_sets(blocks)
        everything = {n for b in BLOCK_ORDER for n in blocks[b]}
        assert set(sets["D"]) == everything

    def test_set_a_is_moisture_only(self, blocks) -> None:
        sets = build_ablation_feature_sets(blocks)
        assert set(sets["A"]) == set(blocks[BLOCK_MOISTURE])

    def test_set_e_is_weather_only(self, blocks) -> None:
        sets = build_ablation_feature_sets(blocks)
        assert set(sets["E"]) == set(blocks[BLOCK_WEATHER])

    def test_control_set_excludes_every_target_derived_column(
        self, blocks,
    ) -> None:
        """Set E is only a control if nothing in it touches the target.

        Both the autoregressive lags and hours-since-last-irrigation are
        computed from the valve state; either would invalidate the test.
        """
        sets = build_ablation_feature_sets(blocks)
        forbidden = set(blocks[BLOCK_IRRIGATION])
        assert not (set(sets["E"]) & forbidden)
        assert not any("irrigation" in name for name in sets["E"])

    def test_control_set_excludes_soil_moisture(self, blocks) -> None:
        sets = build_ablation_feature_sets(blocks)
        assert not any(n.startswith("soil_moisture") for n in sets["E"])

    def test_calendar_appears_only_from_set_c(self, blocks) -> None:
        sets = build_ablation_feature_sets(blocks)
        calendar = set(blocks[BLOCK_CALENDAR])
        assert not (set(sets["A"]) & calendar)
        assert not (set(sets["B"]) & calendar)
        assert calendar <= set(sets["C"])

    def test_no_set_contains_flow_features(self, blocks) -> None:
        sets = build_ablation_feature_sets(blocks)
        for name, columns in sets.items():
            assert not [c for c in columns if c.startswith("flow_l")], name


# ── The shared-rows invariant ────────────────────────────────────────


class TestAblationComparability:
    def test_every_set_is_scored_on_identical_rows(self, design) -> None:
        """The invariant that makes the comparison mean anything."""
        X, y, _, blocks = design
        splits = rolling_origin_splits(len(X))
        results = run_ablation(
            X, y, splits, build_ablation_feature_sets(blocks),
            model_names=("logistic",),
        )
        per_set = results.groupby("feature_set")["n_test"].sum()
        assert per_set.nunique() == 1, (
            f"Sets were scored on different row counts: {per_set.to_dict()}"
        )

    def test_train_sizes_also_match_across_sets(self, design) -> None:
        X, y, _, blocks = design
        splits = rolling_origin_splits(len(X))
        results = run_ablation(
            X, y, splits, build_ablation_feature_sets(blocks),
            model_names=("logistic",),
        )
        per_set = results.groupby("feature_set")["n_train"].sum()
        assert per_set.nunique() == 1

    def test_feature_counts_are_recorded(self, design) -> None:
        X, y, _, blocks = design
        splits = rolling_origin_splits(len(X))
        sets = build_ablation_feature_sets(blocks)
        results = run_ablation(
            X, y, splits, sets, model_names=("logistic",),
        )
        for name, columns in sets.items():
            recorded = results[results["feature_set"] == name]["n_features"]
            assert (recorded == len(columns)).all()


# ── Failure modes ────────────────────────────────────────────────────


class TestAblationGuards:
    def test_missing_column_rejected(self, design) -> None:
        X, y, _, _ = design
        splits = rolling_origin_splits(len(X))
        with pytest.raises(ValueError, match="absent from the design matrix"):
            run_ablation(
                X, y, splits, {"A": ["no_such_feature"]},
                model_names=("logistic",),
            )

    def test_flow_feature_in_a_set_is_rejected(self, design) -> None:
        """A leaked flow column would look like a discovery, not a bug."""
        X, y, _, _ = design
        X = X.copy()
        X["flow_l_lag1h"] = 1.0
        splits = rolling_origin_splits(len(X))
        with pytest.raises(ValueError, match="Forbidden flow-meter features"):
            run_ablation(
                X, y, splits, {"A": ["flow_l_lag1h"]},
                model_names=("logistic",),
            )

    def test_flow_feature_rejected_at_the_training_boundary(
        self, design,
    ) -> None:
        """The guard must fire even for a hand-assembled matrix.

        Feature construction and ablation composition both check, but a
        matrix built directly in a notebook bypasses those. Training is
        the boundary nothing gets past.
        """
        from src.evaluate_pipeline import evaluate_on_splits

        X, y, _, _ = design
        X = X.copy()
        X["flow_l_cumulative"] = np.arange(len(X), dtype=float)
        splits = rolling_origin_splits(len(X))
        with pytest.raises(ValueError, match="Forbidden flow-meter features"):
            evaluate_on_splits(
                make_model_factory("logistic"), X, y, splits,
            )


# ── Summary table ────────────────────────────────────────────────────


class TestSummariseAblation:
    def test_one_row_per_set_and_model(self, design) -> None:
        X, y, _, blocks = design
        splits = rolling_origin_splits(len(X))
        results = run_ablation(
            X, y, splits, build_ablation_feature_sets(blocks),
            model_names=("logistic", "xgboost"),
        )
        summary = summarise_ablation(results)
        assert len(summary) == 5 * 2

    def test_every_metric_carries_an_interval(self, design) -> None:
        X, y, _, blocks = design
        splits = rolling_origin_splits(len(X))
        results = run_ablation(
            X, y, splits, build_ablation_feature_sets(blocks),
            model_names=("logistic",),
        )
        summary = summarise_ablation(results)
        for metric in ("roc_auc", "pr_auc", "f1"):
            for suffix in ("mean", "std", "ci_low", "ci_high"):
                assert f"{metric}_{suffix}" in summary.columns

    def test_intervals_stay_within_unit_range(self, design) -> None:
        X, y, _, blocks = design
        splits = rolling_origin_splits(len(X))
        results = run_ablation(
            X, y, splits, build_ablation_feature_sets(blocks),
            model_names=("logistic",),
        )
        summary = summarise_ablation(results)
        for metric in ("roc_auc", "pr_auc"):
            assert (summary[f"{metric}_ci_low"] >= 0.0).all()
            assert (summary[f"{metric}_ci_high"] <= 1.0).all()


# ── The control, on the real dataset ─────────────────────────────────


class TestLeakageControlOnRealData:
    """Set E must sit near chance, or something upstream is wrong."""

    def test_weather_alone_is_near_chance(self) -> None:
        df = load_modeling_frame()
        X, y, _, blocks = build_design_matrix(df)
        splits = rolling_origin_splits(len(X))
        sets = build_ablation_feature_sets(blocks)

        results = run_ablation(
            X, y, splits, {"E": sets["E"], "D": sets["D"]},
            model_names=("logistic",),
        )
        summary = summarise_ablation(results).set_index("feature_set")

        e_roc = summary.loc["E", "roc_auc_mean"]
        d_roc = summary.loc["D", "roc_auc_mean"]

        assert e_roc < 0.70, (
            f"Weather alone reached ROC-AUC {e_roc:.4f}. Weather cannot "
            f"predict hourly valve state that well; suspect leakage."
        )
        assert d_roc > e_roc + 0.15, (
            "The full feature set barely beats the weather-only control, "
            "which would mean the moisture and target-history features "
            "contribute almost nothing."
        )
