"""Centralised configuration dataclasses for data loading and corruption.

All numeric constants are parameterised here and cross-referenced with the
literature review "AIoT Precision Agriculture Review" (sections and source
numbers given inline).  No magic numbers should appear anywhere else in the
codebase.

Usage
-----
Override defaults via ``configs/default.yaml`` or by constructing dataclass
instances directly in application code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import yaml


# ---------------------------------------------------------------------------
# NASA POWER
# ---------------------------------------------------------------------------

@dataclass
class NASAPowerConfig:
    """Configuration for NASA POWER API requests.

    The coordinates must match the location of the *sensor* dataset.  The
    Mendeley smart-irrigation dataset (doi:10.17632/cjb4vy4mzj.3) was
    recorded on a strawberry field in Areguá, Central Department,
    Paraguay — the country's principal strawberry-producing district.
    Any other coordinate pair makes the meteorological covariates
    physically unrelated to the soil-moisture record.

    Attributes
    ----------
    latitude : float
        Site latitude (default: Areguá, Central, PY — 25.31 °S).
    longitude : float
        Site longitude (default: Areguá, Central, PY — 57.39 °W).
    site_name : str
        Human-readable site label, propagated to the provenance file.
    country : str
        Country of the sensor deployment.
    crop : str
        Crop grown at the site.
    parameters : list[str]
        POWER parameter short-names.
    community : str
        POWER community identifier (``"AG"`` for agroclimatology).
    temporal : str
        Temporal resolution (``"hourly"`` or ``"daily"``).
    base_url : str
        Root URL of the POWER API.
    time_standard : str
        Time standard passed to the API.  **Must be ``"UTC"``** to avoid
        silent misalignment with the Mendeley dataset timestamps during
        merge.  The POWER Hourly API defaults to Local Solar Time (LST)
        which does NOT correspond to the clock-time convention of the
        Mendeley sensors.
    """

    latitude: float = -25.31
    longitude: float = -57.39
    site_name: str = "Areguá, Central Department, Paraguay"
    country: str = "Paraguay"
    crop: str = "strawberry"
    parameters: list[str] = field(
        default_factory=lambda: ["T2M", "RH2M", "WS2M", "ALLSKY_SFC_SW_DWN"],
    )
    community: str = "AG"
    temporal: str = "hourly"
    base_url: str = "https://power.larc.nasa.gov/api/temporal"
    time_standard: str = "UTC"


# ---------------------------------------------------------------------------
# Data Loader
# ---------------------------------------------------------------------------

@dataclass
class DataLoaderConfig:
    """Configuration for :func:`src.data_loader.load_dataset`.

    Attributes
    ----------
    mendeley_dir : str
        Relative (or absolute) path to the directory containing the three
        Mendeley TXT/CSV files.
    raw_cache_dir : str
        Directory for cached raw NASA POWER JSON responses.
    processed_dir : str
        Directory where the final merged DataFrame is saved.
    resample_freq : str
        Pandas offset alias for the target resampling frequency
        (default ``"1h"``).
    use_cache : bool
        If ``True`` the loader reads *only* from local cache — no network
        requests are made.  Essential for reviewer reproducibility in
        offline environments.
    mendeley_utc_offset_hours : float
        UTC offset of the **wall clock** used by the Mendeley sensor
        loggers, expressed the usual way (local = UTC + offset).

        Paraguay observed ``-4`` (PYT, UTC−04:00) across the whole
        observation window 2022-07-12 → 2022-09-16: Paraguayan daylight
        saving time (PYST, UTC−03:00) runs from the first Sunday of
        October to the fourth Sunday of March, so it does not overlap
        the data.  Consequently local timestamps are converted to UTC by
        **adding 4 hours** (``utc = local - offset``).
    solar_peak_tolerance_hours : float
        Half-width of the acceptance window used by the post-merge
        diurnal sanity check on ``solar_radiation``.  The expected peak
        hour is derived from *mendeley_utc_offset_hours*, never
        hard-coded.

        Two hours, because the check compares a *clock*-derived
        expectation against *solar* geometry and the two never coincide
        exactly.  Areguá sits 2.6° west of the UTC−4 zone meridian
        (60 °W), so true solar noon falls at 15.83 UTC — inside hourly
        bin 15, one bin short of the clock-derived 16:00.  Hourly binning
        contributes a further ±0.5 h.  The window is still far tighter
        than the failure modes it exists to catch: the previous
        Karaganda configuration peaked at 07:00 UTC (9 h off) and a
        response served in Local Solar Time peaks at 12:00 (4 h off).
    nasa_power : NASAPowerConfig
        Nested configuration for NASA POWER API calls.
    """

    mendeley_dir: str = (
        "Smart irrigation control system data with soil moisture, "
        "flow meter and electrovalve relay"
    )
    raw_cache_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    resample_freq: str = "1h"
    use_cache: bool = False
    mendeley_utc_offset_hours: float = -4.0
    solar_peak_tolerance_hours: float = 2.0
    nasa_power: NASAPowerConfig = field(default_factory=NASAPowerConfig)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    """Configuration for :func:`src.features.build_features`.

    Every feature declared here is **strictly causal**: the value placed
    on row *t* is a function of observations at *t − 1* or earlier only.
    The single exception is the calendar block, which is a deterministic
    function of the clock and is therefore known at prediction time.

    ``causal_shift`` is applied before every rolling window, so a window
    of *W* hours on row *t* covers the closed interval
    ``[t − W, t − 1]`` — it never touches row *t* itself.

    Attributes
    ----------
    moisture_col : str
        Name of the soil-moisture column.
    weather_cols : tuple[str, ...]
        Meteorological columns to lag and aggregate.
    target_col : str
        Binary irrigation target; also the source of the autoregressive
        lags and of ``hours_since_last_irrigation``.
    timestamp_col : str
        UTC timestamp column, used only for the calendar block.
    moisture_lags : tuple[int, ...]
        Lags (hours) applied to *moisture_col*.
    moisture_diff_lags : tuple[int, ...]
        Horizons (hours) for first differences of *moisture_col*,
        measuring the drying rate over the window ending at ``t − 1``.
    moisture_roll_windows : tuple[int, ...]
        Causal rolling-window widths (hours) for mean/min/max/std of
        *moisture_col*.
    weather_lags : tuple[int, ...]
        Lags (hours) applied to each weather column.
    weather_roll_windows : tuple[int, ...]
        Causal rolling-window widths (hours) for weather aggregates.
    target_lags : tuple[int, ...]
        Autoregressive lags (hours) of *target_col*.  Irrigation is often
        scheduled or hysteretic, so the model is allowed to see its own
        recent history.
    causal_shift : int
        Rows by which every measured series is shifted before any window
        is applied.  **Must be ≥ 1**; 1 means "the most recent
        observation the model may use is ``t − 1``".
    hours_per_day : int
        Period of the hour-of-day sine/cosine encoding.
    require_full_window : bool
        If ``True`` a rolling statistic is emitted only once its whole
        window is populated (``min_periods = window``).  Leading rows are
        therefore NaN and are dropped downstream, which avoids feeding
        the model unstable estimates computed from one or two points.
    forbidden_cols : tuple[str, ...]
        Columns that must never reach the model, checked by
        :func:`src.features.assert_no_forbidden_features`.

        ``flow_l`` and ``flow_l_cumulative`` are *direct consequences of
        the valve opening* — they are the metered volume delivered by the
        very irrigation event being predicted.  Including them, at any
        lag, would be target leakage dressed up as a feature.
    """

    moisture_col: str = "soil_moisture"
    weather_cols: Tuple[str, ...] = (
        "air_temp", "humidity", "wind_speed", "solar_radiation",
    )
    target_col: str = "irrigation_event"
    timestamp_col: str = "timestamp"

    moisture_lags: Tuple[int, ...] = (1, 2, 3, 6, 12, 24)
    moisture_diff_lags: Tuple[int, ...] = (1, 3)
    moisture_roll_windows: Tuple[int, ...] = (6, 24)

    weather_lags: Tuple[int, ...] = (1, 3, 24)
    weather_roll_windows: Tuple[int, ...] = (24,)

    target_lags: Tuple[int, ...] = (1, 2, 3, 24)

    causal_shift: int = 1
    hours_per_day: int = 24
    require_full_window: bool = True
    forbidden_cols: Tuple[str, ...] = ("flow_l", "flow_l_cumulative")


# ---------------------------------------------------------------------------
# Validation protocol
# ---------------------------------------------------------------------------

@dataclass
class ValidationConfig:
    """Configuration for the temporal validation protocol.

    The data are an hourly time series, so every split is **ordered**:
    a fold's test block always lies strictly after its training block and
    nothing is ever shuffled.  Shuffled *k*-fold on this dataset would
    interleave hours from the same irrigation episode across train and
    test and report a wildly optimistic score.

    Attributes
    ----------
    n_folds : int
        Number of rolling-origin folds.  Five is the documented minimum
        for the paper; each fold's test block is
        ``n_samples / (n_folds + 1)`` rows.
    expanding : bool
        ``True`` — expanding origin: fold *k* trains on everything before
        its test block, so the training set grows monotonically.
        ``False`` — sliding window of fixed width, useful for probing
        whether older data still helps.
    holdout_train_fraction : float
        Train share of the single chronological 80/20 split, retained
        alongside cross-validation for comparability with the earlier
        revision of this pipeline and with the wider literature.
    gap_hours : int
        Rows discarded between a fold's train block and its test block.

        Zero is correct here and is *not* an oversight.  Every feature is
        causal (see :mod:`src.features`), so a test row at time *t* uses
        observations from ``t − 1`` and earlier — exactly what an
        operator would have on hand at deployment time.  The
        autoregressive lags read past *valve states*, which are likewise
        observed in deployment, not hidden labels.  An embargo would
        therefore simulate a handicap the deployed system does not have.
        Set a non-zero gap only if the feature set ever stops being
        strictly causal.
    min_test_positives : int
        Minimum positive examples a fold's test block must contain for
        its metrics to be meaningful.  Folds below this threshold are
        reported but flagged, since ROC-AUC and PR-AUC become unstable
        (and F1 degenerate) on a handful of positives.
    random_seeds : tuple[int, ...]
        Seeds for the repeated-run statistics.  Ten runs, 0…9.
    """

    n_folds: int = 5
    expanding: bool = True
    holdout_train_fraction: float = 0.80
    gap_hours: int = 0
    min_test_positives: int = 5
    random_seeds: Tuple[int, ...] = tuple(range(10))


# ---------------------------------------------------------------------------
# Missing-data injection
# ---------------------------------------------------------------------------

@dataclass
class MissingDataConfig:
    """Configuration for :func:`src.data_corruption.inject_missing`.

    Attributes
    ----------
    rate : float
        Fraction of values to replace with NaN.  Valid range
        ``[0.05, 0.30]`` — bounds taken from Review §2.3.1, source [5]
        (MICE imputation benchmark: 5 %–30 % missingness).
    mechanism : str
        ``"mcar"`` — Missing Completely At Random (uniform).
        ``"heat_dependent"`` — probability of missingness increases with
        ``air_temp`` via a logistic function, reflecting the empirical
        observation (Review §2.3.1) that extreme heat accelerates sensor
        failures.
    seed : int or None
        Random seed for deterministic reproducibility.
    heat_midpoint_c : float
        Midpoint (T₅₀) of the logistic curve for ``"heat_dependent"``
        mechanism (°C).
    heat_steepness : float
        Steepness (k) of the logistic curve.
    """

    rate: float = 0.10
    mechanism: str = "mcar"
    seed: Optional[int] = 42
    heat_midpoint_c: float = 35.0
    heat_steepness: float = 0.3


# ---------------------------------------------------------------------------
# Sensor-drift injection
# ---------------------------------------------------------------------------

@dataclass
class SensorDriftConfig:
    """Configuration for :func:`src.data_corruption.inject_sensor_drift`.

    Drift model
    -----------
    ``drift(t) = a × (1 − exp(−b × t))``

    where *t* is hours since the last recalibration event.

    With the defaults ``a = 5.545, b = 0.08``:

    * ``drift(35 h) ≈ 5.06 %``
    * ``drift(39 h) ≈ 5.30 %``  ← realized peak (last step before reset)

    The peak occurs at **t = 39 h**, not t = 40 h, because the drift
    value is computed *before* the recalibration check increments ``t``.
    With ``recalibration_interval_hours = (40, 40)``, the reset fires
    when ``t_since_cal`` reaches 40, so the last drift-affected step
    is t = 39.  This matches the baseline hardware drift of **5.3 %**
    reported in two-month field trials (Review §2.3.1, source [1]).

    Attributes
    ----------
    a : float
        Asymptotic maximum drift (%).
    b : float
        Exponential growth-rate constant (1/h).
    recalibration_interval_hours : tuple[int, int]
        ``(lo, hi)`` — the next recalibration fires after a uniformly
        sampled interval from this range.  The 35–40 h default reproduces
        the empirically observed *average cadence* of the event-triggered
        (threshold-based) recalibration mechanism reported in source [1]
        (Review §2.3.1).  This is a stochastic approximation of the
        observed recalibration frequency — **not** a literal
        re-implementation of the SNN residual-learning algorithm itself.
    ec_factor : float or None
        If set, enables an *additional* multiplicative distortion
        proportional to estimated soil electrical conductivity (EC).
        **Documented approximation, not a calibrated physical model.**
        Reflects the salinity-induced dielectric distortion described in
        Review §2.3.1, sources [3, 4].
    ec_moisture_coeff : float
        Proportionality constant mapping ``soil_moisture`` to EC
        contribution.
    ec_salt_concentration : float
        Assumed background salt concentration (dS/m).  Default 2.0 dS/m
        corresponds to the secondary salinization threshold for irrigated
        Haplic Kastanozem soils in the Kazakhstan steppe zone, as reported
        in a Pavlodar-region study (Soil Systems, 2025,
        doi:10.3390/soilsystems9020057).
    seed : int or None
        Random seed for deterministic reproducibility.
    """

    a: float = 5.545
    b: float = 0.08
    recalibration_interval_hours: Tuple[int, int] = (35, 40)
    ec_factor: Optional[float] = None
    ec_moisture_coeff: float = 0.005
    ec_salt_concentration: float = 2.0
    seed: Optional[int] = 42


# ---------------------------------------------------------------------------
# Top-level corruption config
# ---------------------------------------------------------------------------

@dataclass
class CorruptionConfig:
    """Aggregated corruption configuration.

    Attributes
    ----------
    missing : MissingDataConfig
        Settings for missing-value injection.
    drift : SensorDriftConfig
        Settings for sensor-drift injection.
    """

    missing: MissingDataConfig = field(default_factory=MissingDataConfig)
    drift: SensorDriftConfig = field(default_factory=SensorDriftConfig)


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def load_config_from_yaml(path: str | Path) -> dict:
    """Load a YAML configuration file and return a raw dict.

    Parameters
    ----------
    path : str or Path
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed YAML contents.
    """
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_loader_config(yaml_path: str | Path | None = None) -> DataLoaderConfig:
    """Build a :class:`DataLoaderConfig` from an optional YAML file.

    If *yaml_path* is ``None`` or the file does not exist, pure defaults
    are returned.
    """
    if yaml_path is None:
        return DataLoaderConfig()
    path = Path(yaml_path)
    if not path.exists():
        return DataLoaderConfig()
    raw = load_config_from_yaml(path)
    loader_raw = raw.get("data_loader", {})
    nasa_raw = loader_raw.pop("nasa_power", {})
    nasa_cfg = NASAPowerConfig(**nasa_raw) if nasa_raw else NASAPowerConfig()
    return DataLoaderConfig(**loader_raw, nasa_power=nasa_cfg)


def build_corruption_config(
    yaml_path: str | Path | None = None,
) -> CorruptionConfig:
    """Build a :class:`CorruptionConfig` from an optional YAML file."""
    if yaml_path is None:
        return CorruptionConfig()
    path = Path(yaml_path)
    if not path.exists():
        return CorruptionConfig()
    raw = load_config_from_yaml(path)
    corruption_raw = raw.get("corruption", {})
    missing_raw = corruption_raw.get("missing", {})
    drift_raw = corruption_raw.get("drift", {})
    # Convert tuple from list if loaded from YAML
    if "recalibration_interval_hours" in drift_raw:
        val = drift_raw["recalibration_interval_hours"]
        if isinstance(val, list):
            drift_raw["recalibration_interval_hours"] = tuple(val)
    return CorruptionConfig(
        missing=MissingDataConfig(**missing_raw) if missing_raw else MissingDataConfig(),
        drift=SensorDriftConfig(**drift_raw) if drift_raw else SensorDriftConfig(),
    )
