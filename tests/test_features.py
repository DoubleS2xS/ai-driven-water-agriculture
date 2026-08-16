"""Tests for src.features — causality, block composition, leakage guard.

The centrepiece is :class:`TestNoLookAhead`, which enforces the module's
core contract empirically rather than by inspection: overwrite the tail
of the input with NaN, rebuild, and require that every feature value
before the cut is bit-for-bit unchanged.  A feature that peeked at the
present or the future would move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import FeatureConfig
from src.features import (
    BLOCK_CALENDAR,
    BLOCK_IRRIGATION,
    BLOCK_MOISTURE,
    BLOCK_ORDER,
    BLOCK_WEATHER,
    assert_no_forbidden_features,
    build_features,
    prepare_supervised,
)


# ── Fixtures ──────────────────────────────────────────────────────────


def _make_frame(n: int = 240, seed: int = 0) -> pd.DataFrame:
    """Build a synthetic hourly frame with realistic column names."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2022-07-12 04:00", periods=n, freq="1h")
    hours = np.arange(n)
    return pd.DataFrame({
        "timestamp": ts,
        "soil_moisture": 50.0 + 10.0 * np.sin(hours / 12.0)
        + rng.normal(0, 1, n),
        "air_temp": 20.0 + 8.0 * np.sin((hours - 6) / 24.0 * 2 * np.pi),
        "humidity": rng.uniform(40, 90, n),
        "wind_speed": rng.uniform(0, 8, n),
        "solar_radiation": np.clip(
            800 * np.cos(2 * np.pi * (ts.hour - 16) / 24.0), 0, None
        ),
        "irrigation_event": rng.choice([0.0, 1.0], n, p=[0.8, 0.2]),
        "flow_l": rng.uniform(0, 50, n),
        "flow_l_cumulative": np.cumsum(rng.uniform(0, 50, n)),
    })


@pytest.fixture
def frame() -> pd.DataFrame:
    return _make_frame()


@pytest.fixture
def built(frame: pd.DataFrame):
    return build_features(frame)


# ── The causality contract ───────────────────────────────────────────


class TestNoLookAhead:
    """No feature on row t may depend on data from row t or later."""

    @pytest.mark.parametrize("cut", [50, 120, 200])
    def test_future_nan_does_not_change_past_features(
        self, frame: pd.DataFrame, cut: int,
    ) -> None:
        """Blank everything from `cut` onwards; rows < cut must not move.

        This is the decisive test. If any feature read row t (or later),
        destroying the tail would propagate backwards into rows before
        the cut and the comparison would fail.
        """
        baseline, _ = build_features(frame)

        mutilated = frame.copy()
        data_cols = [c for c in mutilated.columns if c != "timestamp"]
        mutilated.loc[mutilated.index[cut:], data_cols] = np.nan

        perturbed, _ = build_features(mutilated)

        # Row `cut` itself may legitimately use data up to cut-1, so the
        # guarantee covers rows strictly before the cut.
        pd.testing.assert_frame_equal(
            baseline.iloc[:cut],
            perturbed.iloc[:cut],
            check_exact=False,
            obj=f"features before cut={cut}",
        )

    @pytest.mark.parametrize("row", [40, 100, 180])
    def test_blanking_current_row_does_not_change_its_features(
        self, frame: pd.DataFrame, row: int,
    ) -> None:
        """Blank row t alone; row t's own features must not move.

        Complements the NaN-tail test above, which by construction can
        only detect a feature reaching to t+1 or beyond — row t sits
        inside the preserved region there. Blanking a single row closes
        the remaining case: reading one's own present. The two together
        pin the contract to `t-1` and earlier exactly.
        """
        baseline, _ = build_features(frame)

        mutilated = frame.copy()
        data_cols = [c for c in mutilated.columns if c != "timestamp"]
        mutilated.loc[mutilated.index[row], data_cols] = np.nan

        perturbed, _ = build_features(mutilated)

        pd.testing.assert_series_equal(
            baseline.iloc[row].drop("irrigation_event"),
            perturbed.iloc[row].drop("irrigation_event"),
            check_exact=False,
            obj=f"features on blanked row {row}",
        )

    def test_target_column_is_not_a_feature(self, built) -> None:
        """The raw target must never appear among the feature names."""
        _, blocks = built
        cfg = FeatureConfig()
        all_names = [n for b in BLOCK_ORDER for n in blocks[b]]
        assert cfg.target_col not in all_names

    def test_current_soil_moisture_is_not_a_feature(self, built) -> None:
        """soil_moisture(t) is a consequence of irrigation — excluded."""
        _, blocks = built
        all_names = [n for b in BLOCK_ORDER for n in blocks[b]]
        assert "soil_moisture" not in all_names

    def test_lag_values_match_shifted_source(self, frame: pd.DataFrame) -> None:
        """soil_moisture_lag3h on row t must equal soil_moisture(t-3)."""
        features, _ = build_features(frame)
        expected = frame["soil_moisture"].shift(3)
        pd.testing.assert_series_equal(
            features["soil_moisture_lag3h"],
            expected,
            check_names=False,
        )

    def test_rolling_window_excludes_current_row(
        self, frame: pd.DataFrame,
    ) -> None:
        """A 6 h window on row t must cover exactly [t-6, t-1]."""
        features, _ = build_features(frame)
        t = 100
        window = frame["soil_moisture"].iloc[t - 6:t]
        assert features["soil_moisture_roll6h_mean"].iloc[t] == pytest.approx(
            window.mean()
        )
        assert features["soil_moisture_roll6h_max"].iloc[t] == pytest.approx(
            window.max()
        )

    def test_diff_measures_change_ending_at_previous_hour(
        self, frame: pd.DataFrame,
    ) -> None:
        """diff3h on row t is soil_moisture(t-1) - soil_moisture(t-4)."""
        features, _ = build_features(frame)
        sm = frame["soil_moisture"]
        t = 100
        assert features["soil_moisture_diff3h"].iloc[t] == pytest.approx(
            sm.iloc[t - 1] - sm.iloc[t - 4]
        )

    def test_causal_shift_zero_rejected(self, frame: pd.DataFrame) -> None:
        cfg = FeatureConfig(causal_shift=0)
        with pytest.raises(ValueError, match="causal_shift must be >= 1"):
            build_features(frame, cfg)


class TestHoursSinceLastIrrigation:
    def test_event_at_previous_hour_gives_one(self) -> None:
        df = _make_frame(n=48)
        df["irrigation_event"] = 0.0
        df.loc[10, "irrigation_event"] = 1.0
        features, _ = build_features(df)
        assert features["hours_since_last_irrigation"].iloc[11] == 1.0
        assert features["hours_since_last_irrigation"].iloc[15] == 5.0

    def test_nan_before_first_event(self) -> None:
        df = _make_frame(n=48)
        df["irrigation_event"] = 0.0
        df.loc[20, "irrigation_event"] = 1.0
        features, _ = build_features(df)
        assert features["hours_since_last_irrigation"].iloc[:21].isna().all()

    def test_does_not_see_concurrent_event(self) -> None:
        """An event during hour t must not reset the counter on row t."""
        df = _make_frame(n=48)
        df["irrigation_event"] = 0.0
        df.loc[[5, 30], "irrigation_event"] = 1.0
        features, _ = build_features(df)
        # Row 30 has an event, but only rows 31+ may know about it.
        assert features["hours_since_last_irrigation"].iloc[30] == 25.0
        assert features["hours_since_last_irrigation"].iloc[31] == 1.0


# ── Leakage guard ────────────────────────────────────────────────────


class TestForbiddenFeatures:
    def test_flow_l_rejected(self) -> None:
        with pytest.raises(ValueError, match="Forbidden flow-meter features"):
            assert_no_forbidden_features(["soil_moisture_lag1h", "flow_l"])

    def test_flow_cumulative_rejected(self) -> None:
        with pytest.raises(ValueError, match="Forbidden flow-meter features"):
            assert_no_forbidden_features(["flow_l_cumulative"])

    def test_derived_flow_feature_rejected(self) -> None:
        """Lags and rolling stats of flow are equally forbidden."""
        with pytest.raises(ValueError, match="Forbidden flow-meter features"):
            assert_no_forbidden_features(["flow_l_roll24h_mean"])

    def test_clean_set_passes(self) -> None:
        assert_no_forbidden_features(
            ["soil_moisture_lag1h", "air_temp_roll24h_mean", "hour_sin"]
        )

    def test_builder_never_emits_flow(self, built) -> None:
        """Flow columns are present in the input but must not be built."""
        _, blocks = built
        all_names = [n for b in BLOCK_ORDER for n in blocks[b]]
        assert not [n for n in all_names if n.startswith("flow_l")]

    def test_prepare_supervised_rejects_flow(self, built) -> None:
        features, _ = built
        features = features.copy()
        features["flow_l"] = 1.0
        with pytest.raises(ValueError, match="Forbidden flow-meter features"):
            prepare_supervised(features, ["soil_moisture_lag1h", "flow_l"])


# ── Block composition ────────────────────────────────────────────────


class TestBlocks:
    def test_all_blocks_present_and_non_empty(self, built) -> None:
        _, blocks = built
        assert set(blocks) == set(BLOCK_ORDER)
        for name in BLOCK_ORDER:
            assert blocks[name], f"block '{name}' is empty"

    def test_blocks_are_disjoint(self, built) -> None:
        _, blocks = built
        seen: set[str] = set()
        for name in BLOCK_ORDER:
            block = set(blocks[name])
            assert not (block & seen), f"block '{name}' overlaps an earlier one"
            seen |= block

    def test_expected_moisture_features(self, built) -> None:
        _, blocks = built
        cfg = FeatureConfig()
        expected = (
            len(cfg.moisture_lags)
            + len(cfg.moisture_diff_lags)
            + len(cfg.moisture_roll_windows) * 4  # mean/min/max/std
        )
        assert len(blocks[BLOCK_MOISTURE]) == expected

    def test_expected_weather_features(self, built) -> None:
        _, blocks = built
        cfg = FeatureConfig()
        expected = len(cfg.weather_cols) * (
            len(cfg.weather_lags) + len(cfg.weather_roll_windows)
        )
        assert len(blocks[BLOCK_WEATHER]) == expected

    def test_calendar_block_contents(self, built) -> None:
        _, blocks = built
        assert set(blocks[BLOCK_CALENDAR]) == {
            "hour_sin", "hour_cos", "days_since_start",
        }

    def test_hours_since_grouped_with_irrigation_not_moisture(
        self, built,
    ) -> None:
        """It derives from the target, so it must not sit in set A."""
        _, blocks = built
        assert "hours_since_last_irrigation" in blocks[BLOCK_IRRIGATION]
        assert "hours_since_last_irrigation" not in blocks[BLOCK_MOISTURE]

    def test_moisture_block_has_no_target_derived_columns(self, built) -> None:
        _, blocks = built
        assert all(
            n.startswith("soil_moisture_") for n in blocks[BLOCK_MOISTURE]
        )


class TestCalendarFeatures:
    def test_hour_encoding_is_circular(self, frame: pd.DataFrame) -> None:
        """23:00 and 00:00 must be neighbours in (sin, cos) space."""
        features, _ = build_features(frame)
        ts = pd.to_datetime(frame["timestamp"])
        i23 = int(np.flatnonzero(ts.dt.hour == 23)[0])
        i00 = int(np.flatnonzero(ts.dt.hour == 0)[0])
        dist = np.hypot(
            features["hour_sin"].iloc[i23] - features["hour_sin"].iloc[i00],
            features["hour_cos"].iloc[i23] - features["hour_cos"].iloc[i00],
        )
        assert dist == pytest.approx(2 * np.sin(np.pi / 24), abs=1e-9)

    def test_days_since_start_begins_at_zero(self, frame: pd.DataFrame) -> None:
        features, _ = build_features(frame)
        assert features["days_since_start"].iloc[0] == 0.0
        assert features["days_since_start"].iloc[24] == pytest.approx(1.0)

    def test_calendar_features_never_nan(self, built) -> None:
        """Clock features are known in advance — no warm-up period."""
        features, blocks = built
        assert not features[blocks[BLOCK_CALENDAR]].isna().any().any()


# ── Input validation ─────────────────────────────────────────────────


class TestInputValidation:
    def test_missing_target_raises(self, frame: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Missing required columns"):
            build_features(frame.drop(columns=["irrigation_event"]))

    def test_missing_weather_column_raises(self, frame: pd.DataFrame) -> None:
        with pytest.raises(KeyError, match="humidity"):
            build_features(frame.drop(columns=["humidity"]))

    def test_unsorted_input_raises(self, frame: pd.DataFrame) -> None:
        shuffled = frame.iloc[::-1].reset_index(drop=True)
        with pytest.raises(ValueError, match="sorted ascending by timestamp"):
            build_features(shuffled)


# ── Supervised matrix ────────────────────────────────────────────────


class TestPrepareSupervised:
    def test_shapes_align(self, built) -> None:
        features, blocks = built
        names = blocks[BLOCK_MOISTURE] + blocks[BLOCK_WEATHER]
        X, y, ts = prepare_supervised(features, names)
        assert len(X) == len(y) == len(ts)
        assert list(X.columns) == names

    def test_no_nan_survives(self, built) -> None:
        features, blocks = built
        names = [n for b in BLOCK_ORDER for n in blocks[b]]
        X, y, _ = prepare_supervised(features, names)
        assert not X.isna().any().any()
        assert not y.isna().any()

    def test_narrow_set_keeps_more_rows(self, built) -> None:
        """Set A must not pay the warm-up cost of unused 24 h windows."""
        features, blocks = built
        X_wide, _, _ = prepare_supervised(
            features, [n for b in BLOCK_ORDER for n in blocks[b]]
        )
        X_narrow, _, _ = prepare_supervised(
            features, ["soil_moisture_lag1h"]
        )
        assert len(X_narrow) > len(X_wide)

    def test_target_is_integer(self, built) -> None:
        features, blocks = built
        _, y, _ = prepare_supervised(features, blocks[BLOCK_MOISTURE])
        assert y.dtype == np.int64 or y.dtype == int
        assert set(y.unique()) <= {0, 1}

    def test_timestamps_stay_ordered(self, built) -> None:
        features, blocks = built
        _, _, ts = prepare_supervised(features, blocks[BLOCK_MOISTURE])
        assert ts.is_monotonic_increasing

    def test_unknown_feature_raises(self, built) -> None:
        features, _ = built
        with pytest.raises(ValueError, match="not present"):
            prepare_supervised(features, ["no_such_feature"])


# ── Real data ────────────────────────────────────────────────────────


class TestOnRealDataset:
    """Guard the contract on the actual merged dataset, not just synthetic."""

    def test_builds_and_survives_causality_check(self) -> None:
        df = pd.read_csv(
            "data/processed/merged_hourly.csv", parse_dates=["timestamp"],
        )
        df["irrigation_event"] = df["irrigation_event"].fillna(0)

        baseline, blocks = build_features(df)

        cut = len(df) // 2
        mutilated = df.copy()
        data_cols = [c for c in mutilated.columns if c != "timestamp"]
        mutilated.loc[mutilated.index[cut:], data_cols] = np.nan
        perturbed, _ = build_features(mutilated)

        pd.testing.assert_frame_equal(
            baseline.iloc[:cut], perturbed.iloc[:cut], check_exact=False,
        )

        names = [n for b in BLOCK_ORDER for n in blocks[b]]
        X, y, _ = prepare_supervised(baseline, names)
        assert len(X) > 0
        assert 0.0 < y.mean() < 1.0
