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

    Attributes
    ----------
    latitude : float
        Site latitude (default: Karaganda, KZ — 49.80 °N).
    longitude : float
        Site longitude (default: Karaganda, KZ — 73.10 °E).
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

    latitude: float = 49.80
    longitude: float = 73.10
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
    nasa_power: NASAPowerConfig = field(default_factory=NASAPowerConfig)


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
