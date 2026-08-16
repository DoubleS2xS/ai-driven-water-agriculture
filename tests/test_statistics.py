"""Tests for src.statistics — intervals and the bootstrap comparison.

The bootstrap tests are checked against constructed cases where the
answer is known in advance: two identical models must not be declared
different, and a clearly better model must be.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.statistics import (
    DEFAULT_BLOCK_HOURS,
    _moving_block_indices,
    aggregate_runs,
    bootstrap_auc_difference,
    format_p_value,
    summarize_metric,
)


# ── Confidence intervals ─────────────────────────────────────────────


class TestSummarizeMetric:
    def test_matches_manual_t_interval(self) -> None:
        values = [0.80, 0.84, 0.79, 0.91, 0.86]
        result = summarize_metric(values)
        arr = np.array(values)
        expected_hw = (
            stats.t.ppf(0.975, df=4) * arr.std(ddof=1) / np.sqrt(5)
        )
        assert result["mean"] == pytest.approx(arr.mean())
        assert result["std"] == pytest.approx(arr.std(ddof=1))
        assert result["half_width"] == pytest.approx(expected_hw)

    def test_uses_t_not_normal(self) -> None:
        """At n=5 the t interval must be materially wider than normal."""
        values = [0.5, 0.6, 0.7, 0.8, 0.9]
        result = summarize_metric(values, bounds=None)
        arr = np.array(values)
        normal_hw = 1.959964 * arr.std(ddof=1) / np.sqrt(5)
        assert result["half_width"] > normal_hw * 1.2

    def test_interval_is_clipped_to_unit_range(self) -> None:
        """An unclipped t interval on recall can exceed 1.0."""
        values = [0.95, 0.99, 1.0, 0.60, 1.0]
        result = summarize_metric(values)
        assert 0.0 <= result["ci_low"] <= result["ci_high"] <= 1.0

    def test_unclipped_form_still_available(self) -> None:
        values = [0.95, 0.99, 1.0, 0.60, 1.0]
        clipped = summarize_metric(values)
        raw = summarize_metric(values, bounds=None)
        assert raw["ci_high"] > 1.0
        assert clipped["half_width"] == pytest.approx(raw["half_width"])

    def test_zero_variance_is_flagged_degenerate(self) -> None:
        """Ten identical runs are not evidence of precision."""
        result = summarize_metric([0.75] * 10)
        assert result["degenerate"] is True
        assert result["std"] == 0.0
        assert result["ci_low"] == result["ci_high"] == pytest.approx(0.75)

    def test_nan_replicates_are_dropped(self) -> None:
        result = summarize_metric([0.8, np.nan, 0.9, np.nan])
        assert result["n"] == 2
        assert result["mean"] == pytest.approx(0.85)

    def test_all_nan_returns_nan(self) -> None:
        result = summarize_metric([np.nan, np.nan])
        assert np.isnan(result["mean"])
        assert result["n"] == 0

    def test_single_replicate_has_no_interval(self) -> None:
        """Dispersion is undefined at n=1 and must not read as zero."""
        result = summarize_metric([0.8])
        assert result["mean"] == pytest.approx(0.8)
        assert np.isnan(result["std"])
        assert np.isnan(result["ci_low"])

    def test_empty_returns_nan(self) -> None:
        assert summarize_metric([])["n"] == 0

    @pytest.mark.parametrize("confidence", [0.0, 1.0, -0.5, 1.5])
    def test_invalid_confidence_rejected(self, confidence: float) -> None:
        with pytest.raises(ValueError, match="confidence must be"):
            summarize_metric([0.1, 0.2], confidence=confidence)

    def test_wider_confidence_gives_wider_interval(self) -> None:
        values = [0.5, 0.6, 0.7, 0.8]
        narrow = summarize_metric(values, 0.90, bounds=None)
        wide = summarize_metric(values, 0.99, bounds=None)
        assert wide["half_width"] > narrow["half_width"]


# ── Aggregation ──────────────────────────────────────────────────────


def _results_frame(
    n_seeds: int = 10, n_folds: int = 5, deterministic: bool = True,
) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    fold_means = np.linspace(0.6, 0.95, n_folds)
    for seed in range(n_seeds):
        for fold in range(1, n_folds + 1):
            value = fold_means[fold - 1]
            if not deterministic:
                value += rng.normal(0, 0.01)
            rows.append({
                "model": "xgboost", "seed": seed, "fold": fold,
                "roc_auc": value,
            })
    return pd.DataFrame(rows)


class TestAggregateRuns:
    def test_one_row_per_model_metric(self) -> None:
        table = aggregate_runs(_results_frame(), ["roc_auc"])
        assert len(table) == 1
        assert table.iloc[0]["model"] == "xgboost"

    def test_reports_seed_and_fold_counts(self) -> None:
        table = aggregate_runs(_results_frame(), ["roc_auc"])
        assert table.iloc[0]["n_seeds"] == 10
        assert table.iloc[0]["n_folds"] == 5

    def test_primary_interval_is_fold_level(self) -> None:
        """A deterministic model must not get a zero-width headline CI."""
        table = aggregate_runs(_results_frame(deterministic=True), ["roc_auc"])
        row = table.iloc[0]
        assert row["deterministic_across_seeds"]
        assert row["seed_std"] == 0.0
        assert row["std"] > 0.0
        assert row["ci_high"] > row["ci_low"]

    def test_detects_stochastic_models(self) -> None:
        table = aggregate_runs(_results_frame(deterministic=False), ["roc_auc"])
        assert not table.iloc[0]["deterministic_across_seeds"]
        assert table.iloc[0]["seed_std"] > 0.0

    def test_fold_dispersion_dominates_seed_dispersion(self) -> None:
        """The point of reporting both: they differ by orders of magnitude."""
        table = aggregate_runs(_results_frame(deterministic=False), ["roc_auc"])
        row = table.iloc[0]
        assert row["std"] > 10 * row["seed_std"]


# ── Moving-block resampling ──────────────────────────────────────────


class TestMovingBlockIndices:
    def test_returns_requested_length(self) -> None:
        rng = np.random.default_rng(0)
        idx = _moving_block_indices(100, 24, rng)
        assert len(idx) == 100

    def test_indices_stay_in_range(self) -> None:
        rng = np.random.default_rng(0)
        idx = _moving_block_indices(100, 24, rng)
        assert idx.min() >= 0 and idx.max() < 100

    def test_blocks_are_contiguous(self) -> None:
        """Within a block, consecutive draws must be consecutive rows."""
        rng = np.random.default_rng(0)
        block = 10
        idx = _moving_block_indices(100, block, rng)
        first = idx[:block]
        assert np.all(np.diff(first) == 1)

    def test_block_size_one_is_iid(self) -> None:
        rng = np.random.default_rng(0)
        idx = _moving_block_indices(1000, 1, rng)
        # An i.i.d. draw of 1000 from 1000 leaves many rows unused.
        assert len(np.unique(idx)) < 1000

    def test_block_larger_than_sample_is_clamped(self) -> None:
        rng = np.random.default_rng(0)
        idx = _moving_block_indices(10, 50, rng)
        assert len(idx) == 10


# ── Bootstrap comparison ─────────────────────────────────────────────


def _paired_scores(n: int = 600, seed: int = 0):
    """Autocorrelated labels with a strong and a weak scorer."""
    rng = np.random.default_rng(seed)
    signal = np.sin(np.arange(n) / 20.0) + rng.normal(0, 0.3, n)
    y = (signal > 0).astype(int)
    strong = signal + rng.normal(0, 0.2, n)
    weak = signal + rng.normal(0, 2.0, n)
    return y, strong, weak


class TestBootstrapAUCDifference:
    def test_identical_models_are_not_declared_different(self) -> None:
        y, strong, _ = _paired_scores()
        result = bootstrap_auc_difference(
            y, strong, strong, n_iterations=500, seed=0,
        )
        assert result["observed_diff"] == pytest.approx(0.0)
        assert result["p_value"] == pytest.approx(1.0, abs=0.05)

    def test_clearly_better_model_is_detected(self) -> None:
        y, strong, weak = _paired_scores()
        result = bootstrap_auc_difference(
            y, strong, weak, n_iterations=500, seed=0,
        )
        assert result["observed_diff"] > 0
        assert result["p_value"] < 0.05
        assert result["ci_low"] > 0

    def test_direction_reverses_when_arguments_swap(self) -> None:
        y, strong, weak = _paired_scores()
        a = bootstrap_auc_difference(y, strong, weak, n_iterations=300, seed=0)
        b = bootstrap_auc_difference(y, weak, strong, n_iterations=300, seed=0)
        assert a["observed_diff"] == pytest.approx(-b["observed_diff"])

    def test_block_bootstrap_widens_the_interval_of_a_single_auc(self) -> None:
        """Honouring autocorrelation must widen the interval, not narrow it.

        Stated on a *single* model's AUC, which is where the property
        holds unconditionally: resampling individual hours from an
        autocorrelated series treats each as fresh evidence and
        understates the variance.  Comparing against a constant scorer
        isolates one AUC, since a constant always scores exactly 0.5.

        For a *paired* difference the shared dependence partly cancels,
        so the widening is real but smaller — on the project's own
        out-of-fold sample the paired interval grows from 0.019 (i.i.d.)
        to 0.031 (24 h blocks), and p from < 1e-3 to 0.004.
        """
        y, strong, _ = _paired_scores()
        constant = np.full(len(y), 0.5)

        block = bootstrap_auc_difference(
            y, strong, constant, n_iterations=1000,
            block_size=DEFAULT_BLOCK_HOURS, seed=0,
        )
        iid = bootstrap_auc_difference(
            y, strong, constant, n_iterations=1000, block_size=1, seed=0,
        )
        assert (block["ci_high"] - block["ci_low"]) > (
            iid["ci_high"] - iid["ci_low"]
        )

    def test_block_size_changes_the_conclusion_materially(self) -> None:
        """The choice of block length is not a cosmetic detail.

        Deliberately not asserting monotonicity in block size: on a short
        synthetic series that ordering is unstable, because a block
        approaching the autocorrelation period stops adding dependence
        and starts limiting resample diversity. What is stable, and what
        matters, is that ignoring dependence entirely (block = 1) gives a
        different answer from respecting it.
        """
        y, strong, _ = _paired_scores()
        constant = np.full(len(y), 0.5)
        iid = bootstrap_auc_difference(
            y, strong, constant, n_iterations=800, block_size=1, seed=0,
        )
        block = bootstrap_auc_difference(
            y, strong, constant, n_iterations=800, block_size=24, seed=0,
        )
        relative_change = abs(
            (block["ci_high"] - block["ci_low"])
            - (iid["ci_high"] - iid["ci_low"])
        ) / (iid["ci_high"] - iid["ci_low"])
        assert relative_change > 0.10

    def test_is_reproducible_for_a_seed(self) -> None:
        y, strong, weak = _paired_scores()
        a = bootstrap_auc_difference(y, strong, weak, n_iterations=200, seed=5)
        b = bootstrap_auc_difference(y, strong, weak, n_iterations=200, seed=5)
        assert a["p_value"] == b["p_value"]
        assert a["ci_low"] == b["ci_low"]

    def test_pr_auc_metric_supported(self) -> None:
        y, strong, weak = _paired_scores()
        result = bootstrap_auc_difference(
            y, strong, weak, n_iterations=200, seed=0, metric="pr_auc",
        )
        assert result["metric"] == "pr_auc"
        assert np.isfinite(result["observed_diff"])

    def test_unknown_metric_rejected(self) -> None:
        y, strong, weak = _paired_scores()
        with pytest.raises(ValueError, match="metric must be"):
            bootstrap_auc_difference(y, strong, weak, metric="accuracy")

    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            bootstrap_auc_difference([0, 1], [0.1], [0.2, 0.3])

    def test_single_class_rejected(self) -> None:
        with pytest.raises(ValueError, match="single class"):
            bootstrap_auc_difference(
                np.zeros(10, dtype=int), np.arange(10.0), np.arange(10.0),
            )

    def test_p_value_is_a_probability(self) -> None:
        y, strong, weak = _paired_scores()
        result = bootstrap_auc_difference(
            y, strong, weak, n_iterations=300, seed=0,
        )
        assert 0.0 <= result["p_value"] <= 1.0


class TestFormatPValue:
    def test_reports_resolution_floor(self) -> None:
        """A bootstrap cannot resolve p below 1/n_iterations."""
        assert format_p_value(0.0, 10_000) == "< 1e-04"

    def test_ordinary_value_printed_plainly(self) -> None:
        assert format_p_value(0.0234, 10_000) == "0.0234"
