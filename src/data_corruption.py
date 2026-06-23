"""Data corruption utilities — deterministic injection of realistic sensor
failures for benchmarking imputation and drift-compensation algorithms.

Every corruption parameter is grounded in the empirical benchmarks of
**Review §2.3.1** and documented in the per-function docstrings to allow
direct citation in the Methods section of the manuscript.

Usage
-----
::

    from src.config import CorruptionConfig
    from src.data_corruption import inject_missing, inject_sensor_drift

    cfg = CorruptionConfig()
    df_corrupted, df_clean = inject_missing(df, "soil_moisture", cfg.missing)
    df_drifted,   df_clean = inject_sensor_drift(df, "soil_moisture", cfg.drift)

Design notes
------------
* Both functions return ``(df_corrupted, df_clean)`` so that MAE / RMSE
  can be computed against ground truth at the validation stage (cf. MICE
  benchmark: MAE 0.018 at 5 % missing, Review §2.3.1, source [5]).
* Deterministic behaviour is guaranteed via explicit ``seed`` control.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import MissingDataConfig, SensorDriftConfig

logger = logging.getLogger(__name__)


# ======================================================================
# Missing-data injection
# ======================================================================

def inject_missing(
    df: pd.DataFrame,
    column: str,
    config: MissingDataConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inject missing values (NaN) into a column of the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input (clean) data.  Must contain *column* and, for the
        ``"heat_dependent"`` mechanism, an ``"air_temp"`` column.
    column : str
        Target column for NaN injection.
    config : MissingDataConfig
        rate : float ∈ [0.05, 0.30]
            Fraction of values to remove.
            Bounds from Review §2.3.1, source [5] — MICE imputation
            benchmark simulates 5 %–30 % missingness.
        mechanism : ``"mcar"`` | ``"heat_dependent"``
            * ``"mcar"`` — Missing Completely At Random: each row has
              equal probability ``rate`` of being set to NaN.
            * ``"heat_dependent"`` — probability of missingness is a
              logistic function of ``air_temp``, reflecting the empirical
              observation (Review §2.3.1, sources [3, 4]) that extreme
              heat accelerates sensor failures in arid/semi-arid zones.
              The logistic curve is scaled so that the *expected* overall
              NaN fraction equals ``rate``.
        seed : int or None
            For deterministic reproducibility.

    Returns
    -------
    df_corrupted : pd.DataFrame
        Copy with NaN injected into *column*.
    df_clean : pd.DataFrame
        Untouched deep copy of the original input (ground truth).

    Raises
    ------
    ValueError
        If ``rate`` is outside [0.05, 0.30] or *column* is not in *df*.

    References
    ----------
    * Review §2.3.1, source [5]: MICE achieves MAE 0.018 at 5 % missing
      rate, validating the use of controlled NaN injection for
      benchmarking.
    * Review §2.3.1, sources [3, 4]: extreme heat accelerates sensor
      degradation in arid Central Asian environments.
    """
    # ── Validation ─────────────────────────────────────────────────────
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    if not 0.05 <= config.rate <= 0.30:
        raise ValueError(
            f"rate must be in [0.05, 0.30], got {config.rate}. "
            "Bounds from Review §2.3.1, source [5]."
        )
    if config.mechanism not in ("mcar", "heat_dependent"):
        raise ValueError(
            f"mechanism must be 'mcar' or 'heat_dependent', "
            f"got '{config.mechanism}'."
        )

    df_clean = df.copy(deep=True)
    df_corrupted = df.copy(deep=True)
    rng = np.random.default_rng(config.seed)
    n = len(df_corrupted)

    if config.mechanism == "mcar":
        # ── MCAR: uniform random masking ──────────────────────────────
        mask = rng.random(n) < config.rate

    elif config.mechanism == "heat_dependent":
        # ── Heat-dependent: logistic P(missing | air_temp) ────────────
        if "air_temp" not in df.columns:
            raise ValueError(
                "'heat_dependent' mechanism requires an 'air_temp' column."
            )
        temps = df_corrupted["air_temp"].values.astype(float)
        # Raw logistic probabilities
        raw_probs = 1.0 / (
            1.0 + np.exp(
                -config.heat_steepness * (temps - config.heat_midpoint_c)
            )
        )
        # Scale so that mean(probs) == rate
        mean_raw = np.nanmean(raw_probs)
        if mean_raw > 0:
            probs = raw_probs * (config.rate / mean_raw)
        else:
            probs = np.full(n, config.rate)
        probs = np.clip(probs, 0.0, 1.0)
        mask = rng.random(n) < probs
    else:
        # Should not reach here due to validation above
        raise ValueError(f"Unknown mechanism: {config.mechanism}")

    df_corrupted.loc[mask, column] = np.nan

    n_removed = int(mask.sum())
    logger.info(
        "inject_missing: column='%s', mechanism='%s', "
        "target_rate=%.2f, actual_rate=%.4f (%d / %d rows)",
        column,
        config.mechanism,
        config.rate,
        n_removed / n if n > 0 else 0.0,
        n_removed,
        n,
    )
    return df_corrupted, df_clean


# ======================================================================
# Sensor-drift injection
# ======================================================================

def inject_sensor_drift(
    df: pd.DataFrame,
    column: str = "soil_moisture",
    config: Optional[SensorDriftConfig] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inject exponential sensor drift with periodic recalibration resets.

    Drift model
    -----------
    ::

        drift(t) = a × (1 − exp(−b × t))

    where *t* is hours since the last recalibration event.

    With the defaults ``a = 5.545``, ``b = 0.08``:

    * ``drift(35 h) ≈ 5.06 %``  (lower bound of recalibration window)
    * ``drift(39 h) ≈ 5.30 %``  (realized peak, last step before reset)

    The baseline drift profile of **5.3 %** reproduces the empirical
    hardware drift measured in two-month field trials (Review §2.3.1,
    source [1]).  The peak occurs at t = 39 h (not t = 40 h) because
    drift is computed before the recalibration check advances the
    counter.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.  Must contain *column* and a ``"timestamp"`` column
        with hourly spacing.
    column : str
        Target column (default ``"soil_moisture"``).
    config : SensorDriftConfig or None
        If ``None``, :class:`SensorDriftConfig` defaults are used.

        Key attributes:

        a, b : float
            Drift curve parameters.
        recalibration_interval_hours : tuple[int, int]
            ``(lo, hi)`` — the next recalibration fires after a uniformly
            sampled interval from this range (hours).

            The 35–40 h default reproduces the *empirically observed
            average cadence* of the event-triggered (threshold-based)
            recalibration mechanism reported in source [1] (Review
            §2.3.1).  This is a **stochastic approximation** of the
            observed recalibration frequency — not a literal
            re-implementation of the SNN residual-learning algorithm
            itself.  The approximation is appropriate for generating
            realistic drift profiles for benchmarking imputation /
            compensation models.
        ec_factor : float or None
            If set, enables an *additional* multiplicative distortion
            proportional to estimated soil electrical conductivity (EC).
            **This is a documented approximation, NOT a calibrated
            physical model.**  Reflects the salinity-induced dielectric
            distortion mechanism described in Review §2.3.1, sources
            [3, 4].

            EC is approximated as::

                EC_proxy = soil_moisture × ec_moisture_coeff
                           × ec_salt_concentration

            Default ``ec_salt_concentration = 2.0 dS/m`` corresponds to
            the secondary salinization threshold for irrigated Haplic
            Kastanozem soils in the Kazakhstan steppe zone (Pavlodar
            region study, Soil Systems, 2025,
            doi:10.3390/soilsystems9020057).
        seed : int or None
            For deterministic reproducibility.

    Returns
    -------
    df_corrupted : pd.DataFrame
        With drift applied to *column*.
    df_clean : pd.DataFrame
        Untouched deep copy of the original input (ground truth).

    Raises
    ------
    ValueError
        If *column* or ``"timestamp"`` is missing from *df*.

    References
    ----------
    * Review §2.3.1, source [1]: baseline drift 5.3 % (a=5.545,
      b=0.08, peak at t=39 h), event-triggered recalibration cadence
      35–40 h, two-month field validation.
    * Review §2.3.1, sources [3, 4]: salinity-driven dielectric
      distortion in arid Central Asian soils.
    * Soil Systems, 2025, doi:10.3390/soilsystems9020057: EC threshold
      of 2.0 dS/m for Haplic Kastanozem (Kazakhstan steppe zone).
    """
    if config is None:
        config = SensorDriftConfig()

    # ── Validation ─────────────────────────────────────────────────────
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must contain a 'timestamp' column.")

    df_clean = df.copy(deep=True)
    df_corrupted = df.copy(deep=True)
    rng = np.random.default_rng(config.seed)

    n = len(df_corrupted)
    values = df_corrupted[column].values.astype(float).copy()

    # ── Compute drift with periodic recalibration ─────────────────────
    drift_values = np.zeros(n, dtype=float)
    t_since_cal = 0.0  # hours since last recalibration
    next_recal = rng.integers(
        config.recalibration_interval_hours[0],
        config.recalibration_interval_hours[1] + 1,
    )

    for i in range(n):
        # Base exponential drift
        drift_t = config.a * (1.0 - np.exp(-config.b * t_since_cal))

        # Optional EC-based multiplicative distortion
        if config.ec_factor is not None and not np.isnan(values[i]):
            ec_proxy = (
                values[i]
                * config.ec_moisture_coeff
                * config.ec_salt_concentration
            )
            drift_t *= 1.0 + config.ec_factor * ec_proxy

        drift_values[i] = drift_t
        t_since_cal += 1.0  # assume 1-hour step

        # Check for recalibration reset
        if t_since_cal >= next_recal:
            t_since_cal = 0.0
            next_recal = rng.integers(
                config.recalibration_interval_hours[0],
                config.recalibration_interval_hours[1] + 1,
            )

    # Apply drift (additive — percentage-point shift)
    mask_valid = ~np.isnan(values)
    values[mask_valid] += drift_values[mask_valid]
    df_corrupted[column] = values

    # ── Logging ────────────────────────────────────────────────────────
    max_drift = float(np.max(drift_values))
    mean_drift = float(np.mean(drift_values[mask_valid]))
    logger.info(
        "inject_sensor_drift: column='%s', a=%.2f, b=%.3f, "
        "recal_interval=%s, ec_factor=%s, "
        "max_drift=%.4f%%, mean_drift=%.4f%%",
        column,
        config.a,
        config.b,
        config.recalibration_interval_hours,
        config.ec_factor,
        max_drift,
        mean_drift,
    )
    return df_corrupted, df_clean
