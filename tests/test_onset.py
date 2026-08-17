"""Tests for src.onset — the irrigation-onset protocol.

The protocol's validity rests on two constructions, both tested against
hand-built cases: the onset target must mark only episode *starts*, and
the evaluation must be restricted to hours where an onset is possible.
Getting either wrong produces plausible-looking numbers that answer a
different question.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import describe_episodes, find_episodes, irrigation_onset
from src.onset import (
    CONSTANT_UNDER_RESTRICTION,
    ONSET_BASELINES,
    ONSET_TARGET_COLUMN,
    build_onset_design_matrix,
    evaluate_onset,
    summarise_onset,
)
from src.validation import rolling_origin_splits


# ── Fixtures ──────────────────────────────────────────────────────────


def _frame(n: int = 700, seed: int = 0) -> pd.DataFrame:
    """Frame whose irrigation runs in blocks driven by moisture."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2022-07-12 04:00", periods=n, freq="1h")
    hours = np.arange(n)
    moisture = 55.0 + 20.0 * np.sin(hours / 21.0) + rng.normal(0, 1.0, n)
    return pd.DataFrame({
        "timestamp": ts,
        "soil_moisture": moisture,
        "air_temp": 22.0 + 9.0 * np.sin(hours / 24.0 * 2 * np.pi),
        "humidity": rng.uniform(40, 90, n),
        "wind_speed": rng.uniform(0, 8, n),
        "solar_radiation": np.clip(
            800 * np.cos(2 * np.pi * (ts.hour - 16) / 24.0), 0, None
        ),
        "irrigation_event": (moisture < 42).astype(float),
        "flow_l": rng.uniform(0, 50, n),
        "flow_l_cumulative": np.cumsum(rng.uniform(0, 50, n)),
    })


# ── Episode detection ────────────────────────────────────────────────


class TestFindEpisodes:
    def test_single_episode(self) -> None:
        target = pd.Series([0, 0, 1, 1, 1, 0, 0])
        episodes = find_episodes(target)
        assert len(episodes) == 1
        assert episodes.iloc[0]["start_index"] == 2
        assert episodes.iloc[0]["end_index"] == 4
        assert episodes.iloc[0]["length_hours"] == 3

    def test_two_separate_episodes(self) -> None:
        target = pd.Series([1, 0, 1, 1, 0])
        episodes = find_episodes(target)
        assert list(episodes["length_hours"]) == [1, 2]

    def test_episode_at_the_very_start(self) -> None:
        target = pd.Series([1, 1, 0])
        episodes = find_episodes(target)
        assert episodes.iloc[0]["start_index"] == 0
        assert episodes.iloc[0]["length_hours"] == 2

    def test_episode_running_to_the_very_end(self) -> None:
        """A run that never closes must still be detected."""
        target = pd.Series([0, 1, 1])
        episodes = find_episodes(target)
        assert len(episodes) == 1
        assert episodes.iloc[0]["end_index"] == 2

    def test_no_irrigation_gives_empty_table(self) -> None:
        assert find_episodes(pd.Series([0, 0, 0])).empty

    def test_unknown_hour_breaks_a_run(self) -> None:
        """NaN splits an episode rather than bridging it."""
        target = pd.Series([1.0, np.nan, 1.0])
        episodes = find_episodes(target)
        assert len(episodes) == 2
        assert list(episodes["length_hours"]) == [1, 1]

    def test_timestamps_are_reported(self) -> None:
        target = pd.Series([0, 1, 1, 0])
        ts = pd.Series(pd.date_range("2022-08-01", periods=4, freq="1h"))
        episodes = find_episodes(target, ts)
        assert episodes.iloc[0]["start_time"] == pd.Timestamp("2022-08-01 01:00")
        assert episodes.iloc[0]["end_time"] == pd.Timestamp("2022-08-01 02:00")


class TestDescribeEpisodes:
    def test_counts_and_durations(self) -> None:
        target = pd.Series([1, 0, 1, 1, 1, 0, 1, 1])
        summary = describe_episodes(target)
        assert summary["n_episodes"] == 3
        assert summary["n_irrigating_hours"] == 6
        assert summary["duration_hours"]["min"] == 1
        assert summary["duration_hours"]["max"] == 3
        assert summary["duration_hours"]["median"] == 2.0
        assert summary["duration_hours"]["total"] == 6

    def test_histogram_covers_every_episode(self) -> None:
        target = pd.Series([1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1])
        summary = describe_episodes(target)
        assert sum(summary["duration_histogram_hours"].values()) == (
            summary["n_episodes"]
        )

    def test_reports_unknown_hours_as_a_caveat(self) -> None:
        """The count is only an upper bound while gaps exist."""
        target = pd.Series([1.0, np.nan, 1.0])
        summary = describe_episodes(target)
        assert summary["n_unknown_hours"] == 1
        assert "upper bound" in summary["note"]

    def test_empty_record_is_handled(self) -> None:
        summary = describe_episodes(pd.Series([0, 0]))
        assert summary["n_episodes"] == 0
        assert summary["duration_hours"] is None

    def test_serialises_to_json(self) -> None:
        import json

        summary = describe_episodes(pd.Series([1, 0, 1, 1]))
        json.loads(json.dumps(summary, default=str))

    def test_matches_the_real_record(self) -> None:
        df = pd.read_csv(
            "data/processed/merged_hourly.csv", parse_dates=["timestamp"],
        )
        summary = describe_episodes(df["irrigation_event"], df["timestamp"])
        assert summary["n_episodes"] == 52
        assert summary["duration_hours"]["max"] == 117
        assert summary["n_irrigating_hours"] == 333


# ── Onset target ─────────────────────────────────────────────────────


class TestIrrigationOnset:
    def test_marks_only_the_first_hour(self) -> None:
        target = pd.Series([0, 1, 1, 1, 0])
        onset = irrigation_onset(target)
        assert list(onset[1:]) == [1.0, 0.0, 0.0, 0.0]

    def test_marks_each_episode_separately(self) -> None:
        target = pd.Series([0, 1, 0, 1, 1])
        onset = irrigation_onset(target)
        assert onset.sum() == 2

    def test_first_row_is_nan_not_zero(self) -> None:
        """Whether the record opens mid-episode is unknowable."""
        onset = irrigation_onset(pd.Series([1, 1, 0]))
        assert np.isnan(onset.iloc[0])

    def test_unknown_previous_state_gives_nan(self) -> None:
        target = pd.Series([0.0, np.nan, 1.0, 1.0])
        onset = irrigation_onset(target)
        assert np.isnan(onset.iloc[2])

    def test_onset_is_rarer_than_the_raw_target(self) -> None:
        df = _frame()
        raw = df["irrigation_event"]
        onset = irrigation_onset(raw)
        assert onset.sum() < raw.sum()

    def test_onset_count_matches_episode_count(self) -> None:
        """Every episode contributes exactly one onset, given known history."""
        target = pd.Series([0, 1, 1, 0, 0, 1, 0, 1, 1, 1])
        assert irrigation_onset(target).sum() == len(find_episodes(target))


# ── Design matrix ────────────────────────────────────────────────────


class TestOnsetDesignMatrix:
    @pytest.fixture(scope="class")
    def design(self):
        return build_onset_design_matrix(_frame())

    def test_target_is_the_onset_indicator(self, design) -> None:
        _X, y, _ts, _blocks = design
        assert set(y.unique()) <= {0, 1}
        assert 0.0 < y.mean() < 0.25

    def test_restricted_to_hours_an_onset_is_possible(self) -> None:
        """Every retained row must have had the valve closed at t-1."""
        df = _frame()
        X_all, _y, _ts, _b = build_onset_design_matrix(
            df, restrict_to_closed_valve=False,
        )
        X, _y2, _ts2, _b2 = build_onset_design_matrix(df)
        assert len(X) < len(X_all)
        assert CONSTANT_UNDER_RESTRICTION in X_all.columns

    def test_constant_column_is_dropped(self, design) -> None:
        """Zero variance by construction — it cannot contribute."""
        X, _y, _ts, _blocks = design
        assert CONSTANT_UNDER_RESTRICTION not in X.columns

    def test_no_onset_is_lost_by_the_restriction(self) -> None:
        """An onset always has the valve closed at t-1, so none may drop."""
        df = _frame()
        X_all, y_all, _ts, _b = build_onset_design_matrix(
            df, restrict_to_closed_valve=False,
        )
        _X, y, _ts2, _b2 = build_onset_design_matrix(df)
        assert y.sum() == y_all.sum()

    def test_timestamps_stay_ordered(self, design) -> None:
        _X, _y, ts, _blocks = design
        assert ts.is_monotonic_increasing

    def test_no_flow_features_survive(self, design) -> None:
        X, _y, _ts, _blocks = design
        assert not [c for c in X.columns if c.startswith("flow_l")]

    def test_raw_target_is_not_a_feature(self, design) -> None:
        X, _y, _ts, _blocks = design
        assert ONSET_TARGET_COLUMN not in X.columns
        assert "irrigation_event" not in X.columns


# ── Evaluation ───────────────────────────────────────────────────────


class TestEvaluateOnset:
    @pytest.fixture(scope="class")
    def results(self):
        X, y, _ts, _b = build_onset_design_matrix(_frame())
        splits = rolling_origin_splits(len(X))
        return evaluate_onset(X, y, splits, model_names=("majority", "logistic"))

    def test_persistence_is_excluded_by_default(self) -> None:
        """It always predicts 0 here and duplicates the majority baseline."""
        assert "persistence" not in ONSET_BASELINES
        assert "majority" in ONSET_BASELINES

    def test_row_per_model_and_fold(self, results) -> None:
        assert len(results) == 2 * 5

    def test_summary_reports_folds_actually_scored(self, results) -> None:
        """A fold with no onset yields NaN, not a silently-counted zero."""
        summary = summarise_onset(results)
        assert "pr_auc_n_folds_scored" in summary.columns
        assert (summary["pr_auc_n_folds_scored"] <= summary["n_folds"]).all()

    def test_summary_records_the_onset_count(self, results) -> None:
        summary = summarise_onset(results)
        assert (summary["n_onsets"] > 0).all()

    def test_intervals_present_for_every_metric(self, results) -> None:
        summary = summarise_onset(results)
        for metric in ("roc_auc", "pr_auc", "f1"):
            for suffix in ("mean", "std", "ci_low", "ci_high"):
                assert f"{metric}_{suffix}" in summary.columns


class TestOnsetOnRealData:
    def test_onset_is_harder_than_continuation(self) -> None:
        """The point of the protocol: it strips away the easy signal."""
        from src.evaluate_pipeline import build_design_matrix, load_modeling_frame

        df = load_modeling_frame()
        _X_cont, y_cont, _ts, _b = build_design_matrix(df)
        _X_on, y_on, _ts2, _b2 = build_onset_design_matrix(df)

        assert y_on.mean() < y_cont.mean() / 3, (
            "Onset should be several times rarer than the raw target"
        )
        assert y_on.sum() >= 20, "Too few onsets to evaluate at all"
