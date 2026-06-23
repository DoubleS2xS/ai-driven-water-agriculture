"""Tests for src.data_healing — imputation quality, drift compensation,
and evaluation metrics on synthetic sensor data.

The fixture generates a deterministic 100-row hourly time series with
correlated features (soil_moisture, air_temp, humidity), injects 10 %
MCAR NaN, and adds a +5.3 step/drift to a sub-section, mirroring the
corruption profile from Phase 1.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from src.data_healing import DataImputer, DriftCompensator, HealingEvaluator


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate ground-truth, missing-corrupted, and drift-corrupted
    DataFrames.

    Returns:
        Tuple of (df_clean, df_missing, df_drifted):
            - df_clean: 100-row ground truth with correlated features.
            - df_missing: copy with ~10 % MCAR NaN in soil_moisture.
            - df_drifted: copy with +5.3 additive drift on rows 30–69.
    """
    rng = np.random.default_rng(42)
    n = 100

    # Correlated features: soil_moisture loosely tracks humidity
    # and inversely correlates with air_temp.
    air_temp = 20.0 + 10.0 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.normal(0, 1.5, n)
    humidity = 60.0 - 0.5 * air_temp + rng.normal(0, 3.0, n)
    soil_moisture = 50.0 - 0.3 * air_temp + 0.2 * humidity + rng.normal(0, 2.0, n)
    soil_moisture = np.clip(soil_moisture, 5.0, 95.0)

    timestamps = pd.date_range("2022-07-12", periods=n, freq="1h")

    df_clean = pd.DataFrame({
        "timestamp": timestamps,
        "soil_moisture": soil_moisture,
        "air_temp": air_temp,
        "humidity": humidity,
    })

    # ── Missing-data corruption (MCAR, ~10 %) ────────────────────────
    df_missing = df_clean.copy(deep=True)
    nan_indices = rng.choice(n, size=int(n * 0.10), replace=False)
    df_missing.loc[nan_indices, "soil_moisture"] = np.nan

    # ── Drift corruption (+5.3 step on rows 30–69) ───────────────────
    df_drifted = df_clean.copy(deep=True)
    drift_start, drift_end = 30, 70  # 40 rows: indices 30..69
    t_local = np.arange(drift_end - drift_start, dtype=float)
    drift_signal = 5.545 * (1.0 - np.exp(-0.08 * t_local))
    df_drifted.loc[drift_start:drift_end - 1, "soil_moisture"] = (
        df_drifted.loc[drift_start:drift_end - 1, "soil_moisture"].values
        + drift_signal
    )

    return df_clean, df_missing, df_drifted


# =====================================================================
# Imputation tests
# =====================================================================


class TestImputationQuality:
    """Validate that imputation strategies produce reasonable fills."""

    def test_mice_imputation_better_than_linear(
        self,
        synthetic_data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """MICE MAE should be ≤ linear interpolation MAE (or within a
        small tolerance) because MICE exploits cross-feature
        correlations that linear interpolation ignores.
        """
        df_clean, df_missing, _ = synthetic_data
        feature_cols = ["soil_moisture", "air_temp", "humidity"]

        imputer = DataImputer()
        evaluator = HealingEvaluator()

        # Heal with both strategies
        df_mice = imputer.impute_mice(df_missing, feature_cols)
        df_linear = imputer.impute_linear(df_missing, feature_cols)

        # Evaluate on corrupted indices only
        metrics_mice = evaluator.calculate_metrics(
            df_clean, df_missing, df_mice, "soil_moisture",
        )
        metrics_linear = evaluator.calculate_metrics(
            df_clean, df_missing, df_linear, "soil_moisture",
        )

        mae_mice = metrics_mice["mae"]
        mae_linear = metrics_linear["mae"]

        # MICE should be at least as good as linear (allow 20 % margin
        # because with only 10 NaN the difference can be small)
        tolerance_factor = 1.20
        assert mae_mice <= mae_linear * tolerance_factor, (
            f"MICE MAE ({mae_mice:.4f}) should be ≤ "
            f"linear MAE ({mae_linear:.4f}) × {tolerance_factor}"
        )

    def test_mice_no_remaining_nan(
        self,
        synthetic_data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """After MICE imputation, no NaN should remain in feature cols."""
        _, df_missing, _ = synthetic_data
        feature_cols = ["soil_moisture", "air_temp", "humidity"]

        imputer = DataImputer()
        df_healed = imputer.impute_mice(df_missing, feature_cols)

        assert df_healed[feature_cols].isna().sum().sum() == 0

    def test_knn_imputation_fills_gaps(
        self,
        synthetic_data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """KNN imputation should fill all NaN in feature columns."""
        _, df_missing, _ = synthetic_data
        feature_cols = ["soil_moisture", "air_temp", "humidity"]

        imputer = DataImputer()
        df_healed = imputer.impute_knn(df_missing, feature_cols)

        assert df_healed[feature_cols].isna().sum().sum() == 0

    def test_linear_interpolation_fills_gaps(
        self,
        synthetic_data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Linear interpolation should fill all NaN."""
        _, df_missing, _ = synthetic_data
        feature_cols = ["soil_moisture", "air_temp", "humidity"]

        imputer = DataImputer()
        df_healed = imputer.impute_linear(df_missing, feature_cols)

        assert df_healed[feature_cols].isna().sum().sum() == 0

    def test_imputation_preserves_non_corrupted(
        self,
        synthetic_data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Values that were NOT NaN should remain unchanged after MICE."""
        df_clean, df_missing, _ = synthetic_data
        feature_cols = ["soil_moisture", "air_temp", "humidity"]

        imputer = DataImputer()
        df_healed = imputer.impute_mice(df_missing, feature_cols)

        # Check non-NaN soil_moisture values are close to original
        not_nan = df_missing["soil_moisture"].notna()
        original = df_missing.loc[not_nan, "soil_moisture"].values
        healed = df_healed.loc[not_nan, "soil_moisture"].values
        np.testing.assert_allclose(healed, original, atol=1e-6)


# =====================================================================
# Drift compensation tests
# =====================================================================


class TestDriftCompensation:
    """Validate that drift compensation reduces error vs. ground truth."""

    def test_drift_compensation_reduces_error(
        self,
        synthetic_data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """After compensation, MAE on the drifted sub-section should be
        lower than the raw drifted signal's MAE.
        """
        df_clean, _, df_drifted = synthetic_data

        compensator = DriftCompensator()
        evaluator = HealingEvaluator()

        df_compensated = compensator.compensate_exponential_drift(
            df_drifted, "soil_moisture", window_hours=24,
        )

        # Error of the raw drifted signal
        metrics_before = evaluator.calculate_metrics(
            df_clean, df_drifted, df_drifted, "soil_moisture",
        )
        # Error after compensation
        metrics_after = evaluator.calculate_metrics(
            df_clean, df_drifted, df_compensated, "soil_moisture",
        )

        assert metrics_after["mae"] < metrics_before["mae"], (
            f"Compensated MAE ({metrics_after['mae']:.4f}) should be < "
            f"raw drifted MAE ({metrics_before['mae']:.4f})"
        )

    def test_drift_compensation_column_not_found(self) -> None:
        """Raise ValueError when column is missing."""
        df = pd.DataFrame({"soil_moisture": [1, 2, 3]})
        compensator = DriftCompensator()
        with pytest.raises(ValueError, match="not found"):
            compensator.compensate_exponential_drift(df, "nonexistent")

    def test_drift_compensation_clips_range(
        self,
        synthetic_data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Compensated values must be clipped to [0, 100]."""
        _, _, df_drifted = synthetic_data

        compensator = DriftCompensator()
        df_compensated = compensator.compensate_exponential_drift(
            df_drifted, "soil_moisture",
        )

        vals = df_compensated["soil_moisture"].values
        assert np.all(vals >= 0.0), "Found values below 0.0"
        assert np.all(vals <= 100.0), "Found values above 100.0"


# =====================================================================
# Evaluator tests
# =====================================================================


class TestHealingEvaluator:
    """Validate that metrics are computed on corrupted indices only."""

    def test_metrics_on_missing_indices(
        self,
        synthetic_data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """n_corrupted should equal the number of NaN injected."""
        df_clean, df_missing, _ = synthetic_data
        feature_cols = ["soil_moisture", "air_temp", "humidity"]

        imputer = DataImputer()
        evaluator = HealingEvaluator()

        df_healed = imputer.impute_mice(df_missing, feature_cols)
        metrics = evaluator.calculate_metrics(
            df_clean, df_missing, df_healed, "soil_moisture",
        )

        expected_n = int(df_missing["soil_moisture"].isna().sum())
        assert metrics["n_corrupted"] == expected_n

    def test_metrics_on_drift_indices(
        self,
        synthetic_data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """n_corrupted should equal the number of drifted rows (40)."""
        df_clean, _, df_drifted = synthetic_data

        evaluator = HealingEvaluator()
        metrics = evaluator.calculate_metrics(
            df_clean, df_drifted, df_drifted, "soil_moisture",
        )

        # 40 rows were modified, but at t=0 the drift is 0.0 (below
        # the tolerance), so 39 are detected as corrupted.
        assert metrics["n_corrupted"] == 39

    def test_perfect_healing_r2_one(self) -> None:
        """If healed == clean at corrupted indices, R² should be 1.0."""
        n = 50
        df_clean = pd.DataFrame({
            "val": np.arange(n, dtype=float),
        })
        df_corrupted = df_clean.copy()
        df_corrupted.loc[10:19, "val"] = np.nan  # 10 NaN

        # "Perfect" healing = just copy clean values back
        df_healed = df_clean.copy()

        evaluator = HealingEvaluator()
        metrics = evaluator.calculate_metrics(
            df_clean, df_corrupted, df_healed, "val",
        )

        assert metrics["mae"] == pytest.approx(0.0, abs=1e-10)
        assert metrics["r2"] == pytest.approx(1.0, abs=1e-10)

    def test_no_corruption_raises(self) -> None:
        """Should raise when there are no corrupted indices."""
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0]})
        evaluator = HealingEvaluator()
        with pytest.raises(ValueError, match="No corrupted indices"):
            evaluator.calculate_metrics(df, df, df, "val")

    def test_missing_column_raises(self) -> None:
        """Should raise when column is absent."""
        df = pd.DataFrame({"val": [1.0]})
        evaluator = HealingEvaluator()
        with pytest.raises(ValueError, match="not found"):
            evaluator.calculate_metrics(df, df, df, "nonexistent")
