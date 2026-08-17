"""Tests for src.sensitivity and the episode-dominance criterion.

The criterion decides which folds get excluded from an aggregate, so it
has to be right in both directions: a fold made of one continuous episode
must score 1.0, and a fold made of many short episodes must score low.
Anything in between would let the exclusion rule fire on the wrong folds
and quietly reshape the headline numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import ValidationConfig
from src.features import episode_labels
from src.sensitivity import (
    PER_FOLD_METRICS_CSV,
    SENSITIVITY_SUMMARY_CSV,
    SUBSET_ALL,
    SUBSET_REDUCED,
    collect_per_fold_metrics,
    dominance_metadata,
    ranking_is_preserved,
    sensitivity_summary,
    write_per_fold_metrics,
    write_sensitivity_summary,
)
from src.validation import (
    describe_folds,
    dominated_folds,
    episode_dominance,
    rolling_origin_splits,
)


# ── The dominance measure ────────────────────────────────────────────


class TestEpisodeDominance:
    def test_single_episode_scores_one(self) -> None:
        """A block covering one continuous episode is fully dominated."""
        labels = np.array([0.0] * 20)
        dominance, largest, n_eps = episode_dominance(labels)
        assert dominance == pytest.approx(1.0)
        assert largest == 20
        assert n_eps == 1

    def test_many_short_episodes_score_low(self) -> None:
        """Ten one-hour episodes must give 0.1, not something near 1."""
        labels = np.arange(10, dtype=float)
        dominance, largest, n_eps = episode_dominance(labels)
        assert dominance == pytest.approx(0.1)
        assert largest == 1
        assert n_eps == 10

    def test_one_long_episode_among_short_ones(self) -> None:
        """The real shape: 117 h beside a few brief runs."""
        labels = np.array([0.0] * 117 + [1.0, 2.0, 3.0])
        dominance, largest, n_eps = episode_dominance(labels)
        assert dominance == pytest.approx(117 / 120)
        assert largest == 117
        assert n_eps == 4

    def test_non_irrigating_hours_are_ignored(self) -> None:
        """NaN marks hours with the valve closed; they carry no episode."""
        labels = np.array([np.nan, 0.0, 0.0, np.nan, 1.0, np.nan])
        dominance, largest, n_eps = episode_dominance(labels)
        assert dominance == pytest.approx(2 / 3)
        assert n_eps == 2

    def test_block_without_irrigation_is_undefined(self) -> None:
        """Undefined, not zero: there is no mass to concentrate."""
        dominance, largest, n_eps = episode_dominance(np.array([np.nan] * 5))
        assert np.isnan(dominance)
        assert np.isnan(largest)
        assert n_eps == 0

    def test_empty_block_is_undefined(self) -> None:
        dominance, _largest, _n = episode_dominance(np.array([]))
        assert np.isnan(dominance)

    def test_dominance_is_bounded(self) -> None:
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 6, size=200).astype(float)
        dominance, _largest, _n = episode_dominance(labels)
        assert 0.0 < dominance <= 1.0

    def test_two_equal_episodes_give_half(self) -> None:
        labels = np.array([0.0] * 10 + [1.0] * 10)
        dominance, _largest, _n = episode_dominance(labels)
        assert dominance == pytest.approx(0.5)


# ── Threshold behaviour ──────────────────────────────────────────────


class TestDominanceThreshold:
    @staticmethod
    def _fold_inputs(dominant_length: int = 80, n_short: int = 40,
                     n: int = 600):
        """Target whose final fold holds one long episode plus short ones.

        With ``n = 600`` and five folds the blocks are 100 rows wide, so
        the long episode is placed inside the last block rather than
        straddling the boundary — otherwise two folds are dominated and
        the test would not isolate the behaviour it checks.
        """
        target = np.zeros(n, dtype=float)
        # Single-hour episodes spread across every block.
        for i in range(n_short):
            position = 10 + i * 12
            if position < n - dominant_length - 5:
                target[position] = 1.0
        # One long episode wholly inside the final test block.
        target[n - dominant_length:] = 1.0
        # Plus one short episode inside that same block, so its dominance
        # is high but below 1.0 — otherwise no threshold below 1 could
        # ever clear the flag and the configurability test would be
        # untestable.
        target[n - dominant_length - 4] = 1.0
        series = pd.Series(target)
        return series, episode_labels(series)

    def test_long_episode_fold_is_flagged(self) -> None:
        y, labels = self._fold_inputs()
        splits = rolling_origin_splits(len(y))
        table = describe_folds(splits, y, episode_labels=labels)
        assert table["episode_dominated"].iloc[-1]

    def test_folds_of_short_episodes_are_not_flagged(self) -> None:
        y, labels = self._fold_inputs()
        splits = rolling_origin_splits(len(y))
        table = describe_folds(splits, y, episode_labels=labels)
        early = table.iloc[:-1]
        flagged = early.loc[early["episode_dominated"], "fold"].tolist()
        assert flagged == []

    def test_threshold_is_configurable(self) -> None:
        """Raising it above the observed dominance must clear the flag."""
        y, labels = self._fold_inputs()
        splits = rolling_origin_splits(len(y))

        strict = describe_folds(
            splits, y, config=ValidationConfig(
                episode_dominance_threshold=0.999,
            ),
            episode_labels=labels,
        )
        assert not strict["episode_dominated"].any()

    def test_no_dominance_columns_without_labels(self) -> None:
        """Backwards compatible: the columns appear only when asked for."""
        y, _labels = self._fold_inputs()
        table = describe_folds(rolling_origin_splits(len(y)), y)
        assert "episode_dominance" not in table.columns
        assert dominated_folds(table) == []

    def test_dominated_folds_lists_flagged_numbers(self) -> None:
        y, labels = self._fold_inputs()
        splits = rolling_origin_splits(len(y))
        table = describe_folds(splits, y, episode_labels=labels)
        assert dominated_folds(table) == [5]


class TestRealDatasetDominance:
    """The rule must select the known dominated fold, and only it."""

    @pytest.fixture(scope="class")
    def fold_table(self) -> pd.DataFrame:
        from src.evaluate_pipeline import build_design_matrix, load_modeling_frame

        df = load_modeling_frame()
        X, y, ts, _blocks = build_design_matrix(df)
        labels = pd.Series(
            episode_labels(df["irrigation_event"]).to_numpy(),
            index=pd.to_datetime(df["timestamp"]),
        )
        return describe_folds(
            rolling_origin_splits(len(X)), y, ts,
            episode_labels=labels.reindex(pd.to_datetime(ts)).to_numpy(),
        )

    def test_exactly_one_fold_is_dominated(self, fold_table) -> None:
        assert len(dominated_folds(fold_table)) == 1

    def test_the_dominated_fold_holds_the_longest_episode(
        self, fold_table,
    ) -> None:
        flagged = fold_table[fold_table["episode_dominated"]].iloc[0]
        assert flagged["largest_episode_hours"] == 117

    def test_separation_from_the_other_folds_is_wide(
        self, fold_table,
    ) -> None:
        """Not a borderline call: 0.67 against 0.26-0.36 elsewhere."""
        flagged = fold_table[fold_table["episode_dominated"]]
        clean = fold_table[~fold_table["episode_dominated"]]
        assert flagged["episode_dominance"].min() > (
            clean["episode_dominance"].max() + 0.25
        )


# ── Per-fold export ──────────────────────────────────────────────────


def _results(model_names=("a", "b"), n_folds: int = 5) -> pd.DataFrame:
    rows = []
    for model in model_names:
        for fold in range(1, n_folds + 1):
            rows.append({
                "model": model, "fold": fold,
                "roc_auc": 0.7 + 0.05 * fold,
                "pr_auc": 0.4 + 0.1 * fold,
                "f1": 0.5, "precision": 0.5, "recall": 0.5, "brier": 0.1,
                "n_test": 218, "n_positive": 20 * fold,
                "positive_rate": 0.05 * fold, "n_train": 200 * fold,
            })
    return pd.DataFrame(rows)


class TestCollectPerFoldMetrics:
    def test_row_per_protocol_model_fold(self) -> None:
        table = collect_per_fold_metrics(
            {"main": _results(), "onset": _results(("a",))}
        )
        assert len(table) == 2 * 5 + 1 * 5
        assert set(table["protocol"]) == {"main", "onset"}

    def test_renames_positive_rate_for_clarity(self) -> None:
        table = collect_per_fold_metrics({"main": _results()})
        assert "test_positive_rate" in table.columns
        assert "positive_rate" not in table.columns

    def test_carries_every_metric(self) -> None:
        table = collect_per_fold_metrics({"main": _results()})
        for metric in ("roc_auc", "pr_auc", "f1", "precision", "recall",
                       "brier"):
            assert metric in table.columns

    def test_carries_fold_size_context(self) -> None:
        table = collect_per_fold_metrics({"main": _results()})
        assert "n_test" in table.columns
        assert "test_positive_rate" in table.columns

    def test_empty_protocol_is_skipped(self) -> None:
        table = collect_per_fold_metrics(
            {"main": _results(), "onset": pd.DataFrame()}
        )
        assert set(table["protocol"]) == {"main"}

    def test_missing_required_column_rejected(self) -> None:
        with pytest.raises(ValueError, match="missing required columns"):
            collect_per_fold_metrics({"main": pd.DataFrame({"roc_auc": [0.9]})})

    def test_writes_a_file(self, tmp_path) -> None:
        table = collect_per_fold_metrics({"main": _results()})
        path = write_per_fold_metrics(table, tmp_path)
        assert path.name == PER_FOLD_METRICS_CSV
        assert len(pd.read_csv(path)) == len(table)


# ── Sensitivity table ────────────────────────────────────────────────


class TestSensitivitySummary:
    def test_two_subsets_per_model_metric(self) -> None:
        table = sensitivity_summary({"main": _results()}, [4], metrics=("pr_auc",))
        assert set(table["subset"]) == {SUBSET_ALL, SUBSET_REDUCED}
        assert len(table) == 2 * 2  # two models, two subsets

    def test_reduced_subset_drops_the_fold(self) -> None:
        table = sensitivity_summary({"main": _results()}, [4], metrics=("pr_auc",))
        reduced = table[table["subset"] == SUBSET_REDUCED]
        assert (reduced["n_folds"] == 4).all()
        assert (reduced["excluded_folds"] == "4").all()

    def test_all_folds_subset_keeps_everything(self) -> None:
        table = sensitivity_summary({"main": _results()}, [4], metrics=("pr_auc",))
        allf = table[table["subset"] == SUBSET_ALL]
        assert (allf["n_folds"] == 5).all()
        assert (allf["excluded_folds"] == "none").all()

    def test_delta_is_zero_for_the_all_folds_row(self) -> None:
        table = sensitivity_summary({"main": _results()}, [4], metrics=("pr_auc",))
        allf = table[table["subset"] == SUBSET_ALL]
        assert (allf["delta_vs_all_folds"] == 0.0).all()

    def test_delta_measures_the_shift(self) -> None:
        """Fold 5 has the highest pr_auc, so dropping it must lower the mean."""
        table = sensitivity_summary({"main": _results()}, [5], metrics=("pr_auc",))
        reduced = table[table["subset"] == SUBSET_REDUCED]
        assert (reduced["delta_vs_all_folds"] < 0).all()

    def test_empty_exclusion_makes_the_subsets_identical(self) -> None:
        table = sensitivity_summary({"main": _results()}, [], metrics=("pr_auc",))
        means = table.groupby("subset")["mean"].mean()
        assert means[SUBSET_ALL] == pytest.approx(means[SUBSET_REDUCED])

    def test_per_protocol_exclusions_are_independent(self) -> None:
        """Fold numbers index different periods when row sets differ.

        Applying the main task's excluded fold to the onset protocol
        would drop an arbitrary period from it.
        """
        table = sensitivity_summary(
            {"main": _results(), "onset": _results(("a",))},
            {"main": [4], "onset": []},
            metrics=("pr_auc",),
        )
        main_reduced = table[
            (table["protocol"] == "main") & (table["subset"] == SUBSET_REDUCED)
        ]
        onset_reduced = table[
            (table["protocol"] == "onset") & (table["subset"] == SUBSET_REDUCED)
        ]
        assert (main_reduced["n_folds"] == 4).all()
        assert (onset_reduced["n_folds"] == 5).all()
        assert (onset_reduced["excluded_folds"] == "none").all()

    def test_writes_a_file(self, tmp_path) -> None:
        table = sensitivity_summary({"main": _results()}, [4])
        path = write_sensitivity_summary(table, tmp_path)
        assert path.name == SENSITIVITY_SUMMARY_CSV
        assert len(pd.read_csv(path)) == len(table)


class TestRankingPreserved:
    def test_detects_a_preserved_ranking(self) -> None:
        table = sensitivity_summary({"main": _results()}, [4], metrics=("pr_auc",))
        assert ranking_is_preserved(table, "main", "pr_auc") is True

    def test_detects_a_broken_ranking(self) -> None:
        """Construct an overtake: b wins overall, a wins without fold 5."""
        rows = []
        for model, values in [("a", [0.9, 0.9, 0.9, 0.9, 0.1]),
                              ("b", [0.2, 0.2, 0.2, 0.2, 0.99])]:
            for fold, value in enumerate(values, start=1):
                rows.append({
                    "model": model, "fold": fold, "pr_auc": value,
                    "n_test": 100, "positive_rate": 0.2,
                })
        table = sensitivity_summary(
            {"main": pd.DataFrame(rows)}, [1, 2, 3], metrics=("pr_auc",),
        )
        assert ranking_is_preserved(table, "main", "pr_auc") is False


# ── Metadata block ───────────────────────────────────────────────────


class TestDominanceMetadata:
    @pytest.fixture
    def block(self) -> dict:
        fold_table = pd.DataFrame({
            "fold": [1, 2, 3, 4, 5],
            "episode_dominance": [0.29, 0.36, 0.31, 0.67, 0.26],
            "episode_dominated": [False, False, False, True, False],
        })
        return dominance_metadata(
            fold_table, [4], threshold=0.5,
            largest_episode_hours=117,
            n_positive_design_matrix=310,
            n_positive_full_record=333,
        )

    def test_records_the_threshold_and_rule(self, block) -> None:
        """The file must state that the rule preceded the results."""
        assert block["threshold"] == 0.5
        assert "stated in advance" in block["rule"].lower()
        assert "not chosen after" in block["rule"].lower()

    def test_records_excluded_folds(self, block) -> None:
        assert block["excluded_folds"] == [4]

    def test_records_dominance_per_fold(self, block) -> None:
        assert block["dominance_by_fold"]["4"] == pytest.approx(0.67)
        assert len(block["dominance_by_fold"]) == 5

    def test_records_both_positive_class_shares(self, block) -> None:
        """37.7 % of the design matrix, 35.1 % of the full record."""
        largest = block["largest_episode"]
        assert largest["duration_hours"] == 117
        assert largest["share_of_design_matrix_positives"] == pytest.approx(
            0.3774, abs=1e-4
        )
        assert largest["share_of_full_record_positives"] == pytest.approx(
            0.3514, abs=1e-4
        )

    def test_serialises_to_json(self, block) -> None:
        import json

        json.loads(json.dumps(block, default=str))

    def test_handles_undefined_dominance(self) -> None:
        fold_table = pd.DataFrame({
            "fold": [1],
            "episode_dominance": [float("nan")],
            "episode_dominated": [False],
        })
        block = dominance_metadata(
            fold_table, [], threshold=0.5, largest_episode_hours=117,
            n_positive_design_matrix=310, n_positive_full_record=333,
        )
        assert block["dominance_by_fold"]["1"] is None
