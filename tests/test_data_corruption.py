"""Tests for src.data_corruption — missing-data injection, sensor-drift
injection, determinism, ground-truth preservation, and parameter bounds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import MissingDataConfig, SensorDriftConfig
from src.data_corruption import inject_missing, inject_sensor_drift


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Hourly DataFrame with realistic values for corruption tests."""
    rng = np.random.default_rng(99)
    n = 200
    timestamps = pd.date_range("2022-07-12", periods=n, freq="1h")
    return pd.DataFrame({
        "timestamp": timestamps,
        "soil_moisture": rng.uniform(30, 80, n),
        "air_temp": rng.uniform(15, 42, n),
        "humidity": rng.uniform(20, 80, n),
    })


# =====================================================================
# inject_missing
# =====================================================================


class TestInjectMissing:

    # ── Rate bounds ───────────────────────────────────────────────────

    def test_rate_below_minimum(self, sample_df: pd.DataFrame) -> None:
        config = MissingDataConfig(rate=0.03)
        with pytest.raises(ValueError, match="rate must be in"):
            inject_missing(sample_df, "soil_moisture", config)

    def test_rate_above_maximum(self, sample_df: pd.DataFrame) -> None:
        config = MissingDataConfig(rate=0.35)
        with pytest.raises(ValueError, match="rate must be in"):
            inject_missing(sample_df, "soil_moisture", config)

    def test_rate_at_boundaries(self, sample_df: pd.DataFrame) -> None:
        """Exactly 0.05 and 0.30 should be accepted."""
        for rate in (0.05, 0.30):
            config = MissingDataConfig(rate=rate, seed=1)
            df_c, _ = inject_missing(sample_df, "soil_moisture", config)
            assert df_c["soil_moisture"].isna().sum() > 0

    # ── Determinism ───────────────────────────────────────────────────

    def test_deterministic_mcar(self, sample_df: pd.DataFrame) -> None:
        config = MissingDataConfig(rate=0.15, mechanism="mcar", seed=77)
        df_c1, _ = inject_missing(sample_df, "soil_moisture", config)
        df_c2, _ = inject_missing(sample_df, "soil_moisture", config)
        pd.testing.assert_frame_equal(df_c1, df_c2)

    def test_deterministic_heat(self, sample_df: pd.DataFrame) -> None:
        config = MissingDataConfig(
            rate=0.15, mechanism="heat_dependent", seed=77
        )
        df_c1, _ = inject_missing(sample_df, "soil_moisture", config)
        df_c2, _ = inject_missing(sample_df, "soil_moisture", config)
        pd.testing.assert_frame_equal(df_c1, df_c2)

    # ── MCAR fraction ─────────────────────────────────────────────────

    def test_mcar_actual_fraction(self, sample_df: pd.DataFrame) -> None:
        """Actual NaN fraction should be within ±5 pp of target rate."""
        config = MissingDataConfig(rate=0.20, mechanism="mcar", seed=42)
        df_c, _ = inject_missing(sample_df, "soil_moisture", config)
        actual = df_c["soil_moisture"].isna().mean()
        assert abs(actual - 0.20) < 0.05

    # ── Heat-dependent: hotter → more missing ─────────────────────────

    def test_heat_dependent_bias(self, sample_df: pd.DataFrame) -> None:
        """Rows with above-median temperature should have more NaN."""
        config = MissingDataConfig(
            rate=0.20, mechanism="heat_dependent", seed=42
        )
        df_c, _ = inject_missing(sample_df, "soil_moisture", config)

        median_temp = sample_df["air_temp"].median()
        hot = sample_df["air_temp"] > median_temp
        cold = ~hot

        nan_hot = df_c.loc[hot, "soil_moisture"].isna().mean()
        nan_cold = df_c.loc[cold, "soil_moisture"].isna().mean()
        assert nan_hot > nan_cold, (
            f"Expected more NaN in hot rows ({nan_hot:.3f}) "
            f"than cold ({nan_cold:.3f})"
        )

    # ── Ground truth preserved ────────────────────────────────────────

    def test_ground_truth_untouched(self, sample_df: pd.DataFrame) -> None:
        config = MissingDataConfig(rate=0.10, seed=1)
        _, df_clean = inject_missing(sample_df, "soil_moisture", config)
        pd.testing.assert_frame_equal(df_clean, sample_df)

    # ── Column not found ──────────────────────────────────────────────

    def test_column_not_found(self, sample_df: pd.DataFrame) -> None:
        config = MissingDataConfig(rate=0.10)
        with pytest.raises(ValueError, match="not found"):
            inject_missing(sample_df, "nonexistent_col", config)


# =====================================================================
# inject_sensor_drift
# =====================================================================


class TestInjectSensorDrift:

    # ── Max drift before reset ≈ 5.3% ────────────────────────────────

    def test_max_drift_approx_5_3(self, sample_df: pd.DataFrame) -> None:
        """Peak drift before recalibration should be ≈ 5.3% (±0.05 pp)."""
        config = SensorDriftConfig(
            a=5.545,
            b=0.08,
            recalibration_interval_hours=(40, 40),  # force 40h
            seed=42,
        )
        df_c, df_clean = inject_sensor_drift(
            sample_df, "soil_moisture", config
        )
        # Drift is additive: diff = corrupted - clean
        diff = df_c["soil_moisture"] - df_clean["soil_moisture"]
        max_diff = diff.max()
        # drift(39) = 5.545 * (1 - exp(-0.08*39)) ≈ 5.30
        # (t=39 is the last step before reset at t=40)
        assert abs(max_diff - 5.3) < 0.05, f"max_diff={max_diff:.4f}"

    # ── Drift resets after recalibration ──────────────────────────────

    def test_drift_resets(self, sample_df: pd.DataFrame) -> None:
        """After a recalibration, drift should drop back near 0."""
        config = SensorDriftConfig(
            a=5.545,
            b=0.08,
            recalibration_interval_hours=(35, 35),  # force 35h
            seed=42,
        )
        df_c, df_clean = inject_sensor_drift(
            sample_df, "soil_moisture", config
        )
        diff = (df_c["soil_moisture"] - df_clean["soil_moisture"]).values

        # At index 35, drift should have reset → diff[35] should be
        # close to 0 (it's the first step after recal, so drift(0)=0)
        assert abs(diff[35]) < 0.1, f"diff[35]={diff[35]:.4f}"

    # ── Determinism ───────────────────────────────────────────────────

    def test_deterministic(self, sample_df: pd.DataFrame) -> None:
        config = SensorDriftConfig(seed=123)
        df_c1, _ = inject_sensor_drift(sample_df, "soil_moisture", config)
        df_c2, _ = inject_sensor_drift(sample_df, "soil_moisture", config)
        pd.testing.assert_frame_equal(df_c1, df_c2)

    # ── Ground truth preserved ────────────────────────────────────────

    def test_ground_truth_preserved(self, sample_df: pd.DataFrame) -> None:
        config = SensorDriftConfig(seed=42)
        _, df_clean = inject_sensor_drift(sample_df, "soil_moisture", config)
        pd.testing.assert_frame_equal(df_clean, sample_df)

    # ── EC factor increases total distortion ──────────────────────────

    def test_ec_factor_increases_drift(self, sample_df: pd.DataFrame) -> None:
        cfg_base = SensorDriftConfig(ec_factor=None, seed=42)
        cfg_ec = SensorDriftConfig(ec_factor=0.1, seed=42)

        df_base, df_clean = inject_sensor_drift(
            sample_df, "soil_moisture", cfg_base
        )
        df_ec, _ = inject_sensor_drift(
            sample_df, "soil_moisture", cfg_ec
        )

        drift_base = (df_base["soil_moisture"] - df_clean["soil_moisture"]).abs().mean()
        drift_ec = (df_ec["soil_moisture"] - df_clean["soil_moisture"]).abs().mean()
        assert drift_ec > drift_base, (
            f"EC drift ({drift_ec:.4f}) should exceed base ({drift_base:.4f})"
        )

    # ── Column not found ──────────────────────────────────────────────

    def test_column_not_found(self, sample_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="not found"):
            inject_sensor_drift(sample_df, "no_such_col")

    # ── Timestamp required ────────────────────────────────────────────

    def test_timestamp_required(self) -> None:
        df = pd.DataFrame({"soil_moisture": [1, 2, 3]})
        with pytest.raises(ValueError, match="timestamp"):
            inject_sensor_drift(df, "soil_moisture")
