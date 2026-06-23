"""Phase 2 — Data Healing: imputation, drift compensation, and evaluation.

This module provides three classes that form the healing stage of the
AIoT precision-agriculture pipeline:

1. **DataImputer** — fills sensor gaps using MICE, KNN, or linear
   interpolation so that downstream models receive complete feature
   matrices.
2. **DriftCompensator** — removes the exponential sensor-drift trend
   injected by :mod:`src.data_corruption` (or present in real telemetry)
   via a rolling-minimum baseline extraction followed by EWM smoothing.
3. **HealingEvaluator** — computes MAE, RMSE, and R² **only at
   corrupted indices**, ensuring that healing quality is assessed where
   it matters rather than being diluted by intact readings.

Design notes
------------
* Every public method returns a **new** DataFrame — inputs are never
  mutated in place.
* Determinism is enforced through explicit ``random_state`` / ``seed``
  arguments passed to scikit-learn estimators.
* All operations are logged at INFO level for pipeline auditability.

References
----------
* Review §2.3.1, source [5]: MICE benchmark — MAE 0.018 at 5 % missing.
* Review §2.3.1, source [1]: baseline drift 5.3 %, recalibration cadence
  35–40 h.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


# ======================================================================
# Imputation
# ======================================================================


class DataImputer:
    """Sensor-gap imputation using multiple strategies.

    Each method accepts a corrupted DataFrame and a list of feature
    columns, returns a healed copy with NaN values filled, and leaves
    all other columns untouched.

    Attributes:
        mice_max_iter: Maximum iterations for the MICE imputer.
        mice_random_state: Random seed for MICE reproducibility.
        knn_n_neighbors: Default number of neighbours for KNN.
    """

    def __init__(
        self,
        mice_max_iter: int = 10,
        mice_random_state: int = 42,
        knn_n_neighbors: int = 5,
    ) -> None:
        """Initialise imputer with default hyper-parameters.

        Args:
            mice_max_iter: Maximum MICE (IterativeImputer) iterations.
            mice_random_state: Seed for deterministic MICE output.
            knn_n_neighbors: Default *k* for KNN imputation.
        """
        self.mice_max_iter: int = mice_max_iter
        self.mice_random_state: int = mice_random_state
        self.knn_n_neighbors: int = knn_n_neighbors

    # ── MICE ──────────────────────────────────────────────────────────

    def impute_mice(
        self,
        df_corrupted: pd.DataFrame,
        feature_cols: List[str],
    ) -> pd.DataFrame:
        """Impute missing values using MICE (Multiple Imputation by
        Chained Equations) via scikit-learn's ``IterativeImputer``.

        MICE leverages cross-correlations between features (e.g.
        soil_moisture ↔ air_temp ↔ humidity) to produce statistically
        plausible fill values.  This is the recommended strategy when
        multiple sensor channels are available (Review §2.3.1,
        source [5]).

        Args:
            df_corrupted: DataFrame with NaN gaps in one or more of
                *feature_cols*.
            feature_cols: Columns to include in the multivariate
                imputation model.

        Returns:
            A new DataFrame with NaN values in *feature_cols* filled.
        """
        logger.info(
            "MICE imputation: cols=%s, max_iter=%d, seed=%d",
            feature_cols,
            self.mice_max_iter,
            self.mice_random_state,
        )

        df_healed = df_corrupted.copy(deep=True)

        imputer = IterativeImputer(
            max_iter=self.mice_max_iter,
            random_state=self.mice_random_state,
            sample_posterior=False,
        )

        data_matrix = df_healed[feature_cols].values
        n_nan_before = int(np.isnan(data_matrix).sum())

        imputed_matrix = imputer.fit_transform(data_matrix)
        df_healed[feature_cols] = imputed_matrix

        n_nan_after = int(np.isnan(df_healed[feature_cols].values).sum())
        logger.info(
            "MICE complete: %d NaN → %d NaN (%d filled)",
            n_nan_before,
            n_nan_after,
            n_nan_before - n_nan_after,
        )
        return df_healed

    # ── KNN ───────────────────────────────────────────────────────────

    def impute_knn(
        self,
        df_corrupted: pd.DataFrame,
        feature_cols: List[str],
        n_neighbors: Optional[int] = None,
    ) -> pd.DataFrame:
        """Impute missing values using K-Nearest Neighbours.

        Each NaN is filled with the weighted mean of the *k* nearest
        complete samples (Euclidean distance on *feature_cols*).

        Args:
            df_corrupted: DataFrame with NaN gaps.
            feature_cols: Columns for the neighbour search.
            n_neighbors: Number of neighbours.  Falls back to
                ``self.knn_n_neighbors`` if *None*.

        Returns:
            A new DataFrame with NaN values in *feature_cols* filled.
        """
        k = n_neighbors if n_neighbors is not None else self.knn_n_neighbors
        logger.info("KNN imputation: cols=%s, k=%d", feature_cols, k)

        df_healed = df_corrupted.copy(deep=True)

        imputer = KNNImputer(n_neighbors=k)

        data_matrix = df_healed[feature_cols].values
        n_nan_before = int(np.isnan(data_matrix).sum())

        imputed_matrix = imputer.fit_transform(data_matrix)
        df_healed[feature_cols] = imputed_matrix

        n_nan_after = int(np.isnan(df_healed[feature_cols].values).sum())
        logger.info(
            "KNN complete: %d NaN → %d NaN (%d filled)",
            n_nan_before,
            n_nan_after,
            n_nan_before - n_nan_after,
        )
        return df_healed

    # ── Linear interpolation (baseline) ───────────────────────────────

    def impute_linear(
        self,
        df_corrupted: pd.DataFrame,
        feature_cols: List[str],
    ) -> pd.DataFrame:
        """Impute missing values using linear interpolation.

        This is the simplest baseline: each column is interpolated
        independently along the time axis, then forward-filled and
        backward-filled to handle leading/trailing NaN.

        Args:
            df_corrupted: DataFrame with NaN gaps.
            feature_cols: Columns to interpolate.

        Returns:
            A new DataFrame with NaN values in *feature_cols* filled.
        """
        logger.info("Linear interpolation: cols=%s", feature_cols)

        df_healed = df_corrupted.copy(deep=True)

        n_nan_before = int(df_healed[feature_cols].isna().sum().sum())

        for col in feature_cols:
            df_healed[col] = (
                df_healed[col]
                .interpolate(method="linear")
                .ffill()
                .bfill()
            )

        n_nan_after = int(df_healed[feature_cols].isna().sum().sum())
        logger.info(
            "Linear interpolation complete: %d NaN → %d NaN (%d filled)",
            n_nan_before,
            n_nan_after,
            n_nan_before - n_nan_after,
        )
        return df_healed


# ======================================================================
# Drift compensation
# ======================================================================


class DriftCompensator:
    """Remove exponential sensor drift from a time-series column.

    The algorithm:

    1. Extract a *baseline drift envelope* using a rolling minimum
       (``window_hours``).  The minimum tracks the lower bound of the
       signal, which in a drifting sensor rises over time.
    2. Smooth the envelope with an Exponentially Weighted Mean
       (``ewm(span=window_hours)``) to suppress step artefacts at
       recalibration boundaries.
    3. Subtract the smoothed trend from the raw signal to recover the
       drift-free reading.
    4. Clip the result to ``[0.0, 100.0]`` (physical range for
       volumetric soil moisture, %).

    This is *not* a literal inversion of the injection model in
    :mod:`src.data_corruption`; it is a signal-processing heuristic
    suitable for benchmarking more sophisticated compensation
    algorithms (e.g. SNN residual learning — Review §2.3.1,
    source [1]).
    """

    def compensate_exponential_drift(
        self,
        df_corrupted: pd.DataFrame,
        column: str,
        window_hours: int = 24,
    ) -> pd.DataFrame:
        """Compensate exponential drift on *column*.

        Args:
            df_corrupted: DataFrame with a drifted *column*.
            column: Name of the drifted column (e.g. ``"soil_moisture"``).
            window_hours: Window size (in rows = hours for hourly data)
                for the rolling-minimum baseline extraction and the EWM
                smoothing span.

        Returns:
            A new DataFrame with drift removed from *column*.

        Raises:
            ValueError: If *column* is not present in *df_corrupted*.
        """
        if column not in df_corrupted.columns:
            raise ValueError(
                f"Column '{column}' not found in DataFrame. "
                f"Available: {list(df_corrupted.columns)}"
            )

        logger.info(
            "Drift compensation: column='%s', window=%d h",
            column,
            window_hours,
        )

        df_healed = df_corrupted.copy(deep=True)
        signal = df_healed[column].astype(float)

        # Step 1: rolling-minimum baseline envelope
        baseline_raw = signal.rolling(
            window=window_hours, min_periods=1, center=True,
        ).min()

        # Step 2: smooth the envelope to suppress step artefacts
        baseline_smooth = baseline_raw.ewm(
            span=window_hours, adjust=False,
        ).mean()

        # Step 3: estimate drift as departure from the global minimum
        #   drift_trend ≈ baseline_smooth - global_minimum
        global_min = signal.min()
        drift_trend = baseline_smooth - global_min

        # Step 4: subtract drift and clip to physical range
        corrected = signal - drift_trend
        corrected = corrected.clip(lower=0.0, upper=100.0)

        df_healed[column] = corrected

        max_correction = float(drift_trend.max())
        mean_correction = float(drift_trend.mean())
        logger.info(
            "Drift compensation complete: max_correction=%.4f, "
            "mean_correction=%.4f",
            max_correction,
            mean_correction,
        )
        return df_healed


# ======================================================================
# Evaluation
# ======================================================================


class HealingEvaluator:
    """Compute healing-quality metrics at corrupted indices only.

    Evaluating imputation / drift-compensation quality on the *full*
    time series would dilute the signal — most rows were never
    corrupted.  This class restricts MAE, RMSE, and R² to the subset
    of indices where corruption actually occurred, providing an honest
    measure of healing effectiveness.
    """

    @staticmethod
    def calculate_metrics(
        df_clean: pd.DataFrame,
        df_corrupted: pd.DataFrame,
        df_healed: pd.DataFrame,
        column: str,
    ) -> Dict[str, float]:
        """Compute MAE, RMSE, and R² at corrupted positions only.

        A position is considered "corrupted" if:

        * the value is NaN in *df_corrupted* (missing-data corruption), **or**
        * the value differs from *df_clean* by more than a negligible
          tolerance (drift corruption).

        Args:
            df_clean: Ground-truth DataFrame (pre-corruption).
            df_corrupted: DataFrame after corruption injection.
            df_healed: DataFrame after healing (imputation / drift
                compensation).
            column: Target column to evaluate.

        Returns:
            Dictionary with keys ``"mae"``, ``"rmse"``, ``"r2"``, and
            ``"n_corrupted"`` (number of evaluated indices).

        Raises:
            ValueError: If *column* is missing from any input, or no
                corrupted indices are found.
        """
        for label, df in [
            ("df_clean", df_clean),
            ("df_corrupted", df_corrupted),
            ("df_healed", df_healed),
        ]:
            if column not in df.columns:
                raise ValueError(
                    f"Column '{column}' not found in {label}. "
                    f"Available: {list(df.columns)}"
                )

        clean_vals = df_clean[column].values.astype(float)
        corrupted_vals = df_corrupted[column].values.astype(float)
        healed_vals = df_healed[column].values.astype(float)

        # Identify corrupted indices:
        #   - NaN in corrupted (missing-data injection)
        #   - value differs from clean (drift injection)
        _DRIFT_TOL = 1e-6
        is_nan = np.isnan(corrupted_vals)
        is_drifted = (
            ~np.isnan(clean_vals)
            & ~np.isnan(corrupted_vals)
            & (np.abs(corrupted_vals - clean_vals) > _DRIFT_TOL)
        )
        corrupted_mask = is_nan | is_drifted

        n_corrupted = int(corrupted_mask.sum())
        if n_corrupted == 0:
            raise ValueError(
                "No corrupted indices found — cannot compute metrics."
            )

        y_true = clean_vals[corrupted_mask]
        y_pred = healed_vals[corrupted_mask]

        # Guard against NaN in healed output (incomplete healing)
        valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
        if valid.sum() == 0:
            raise ValueError(
                "All healed values at corrupted indices are NaN."
            )
        y_true = y_true[valid]
        y_pred = y_pred[valid]

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))

        logger.info(
            "Healing metrics for '%s' (n=%d corrupted indices): "
            "MAE=%.6f, RMSE=%.6f, R²=%.6f",
            column,
            n_corrupted,
            mae,
            rmse,
            r2,
        )
        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "n_corrupted": n_corrupted,
        }
