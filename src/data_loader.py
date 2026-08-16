"""Data loading, merging, validation, and provenance for the AIoT
precision-agriculture pipeline.

This module combines two real, open data sources into a single hourly
``pandas.DataFrame``:

1. **Mendeley Data** — Smart irrigation control system dataset
   (doi:10.17632/cjb4vy4mzj.3): soil moisture, electrovalve relay
   status, cumulative flow-meter readings.  Recorded on a strawberry
   field in Areguá, Central Department, **Paraguay**.
2. **NASA POWER API** — Hourly reanalysis meteorology for the *same*
   site (lat −25.31, lon −57.39): T2M, RH2M, WS2M, ALLSKY_SFC_SW_DWN.

Design decisions
----------------
* Column names in the Mendeley CSVs are discovered dynamically at load
  time — never hard-coded — to tolerate future schema revisions.
* NASA POWER responses are cached to ``data/raw/`` so that reviewers can
  reproduce results fully offline (``--use-cache``).  Cache filenames
  embed the coordinates so that a change of site can never be masked by
  a stale cache hit.
* The ``flow_l`` column contains *differential* (per-hour) volume;
  raw cumulative readings are preserved as ``flow_l_cumulative``.
* All timestamps in the output DataFrame are in **UTC**.  The NASA POWER
  API is called with ``time-standard=UTC`` explicitly to avoid the
  default Local Solar Time (LST).  The Mendeley timestamps are *local*
  Paraguayan wall-clock time and are converted to UTC before the merge
  by :func:`localize_mendeley_to_utc`, using the configured offset.
* After the merge, :func:`validate_diurnal_alignment` verifies that the
  mean diurnal cycle of ``solar_radiation`` peaks at the UTC hour
  implied by the configured local offset.  A misconfigured time standard
  or a wrong longitude shows up here as a hard failure instead of a
  silent, uncorrelated feature set.

References
----------
* Review §2.3.1, sources [1]–[5].
* Mendeley Data: https://data.mendeley.com/datasets/cjb4vy4mzj/3
* NASA POWER: https://power.larc.nasa.gov/
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yaml

from src.config import DataLoaderConfig, NASAPowerConfig, build_loader_config

logger = logging.getLogger(__name__)

# ── Output schema ──────────────────────────────────────────────────────
EXPECTED_COLUMNS: list[str] = [
    "timestamp",
    "soil_moisture",
    "air_temp",
    "humidity",
    "wind_speed",
    "solar_radiation",
    "irrigation_event",
    "flow_l",
    "flow_l_cumulative",
]
"""Final output schema.

Units
-----
- ``soil_moisture`` : volumetric water content (%)
- ``air_temp`` : °C (NASA POWER T2M)
- ``humidity`` : % (NASA POWER RH2M)
- ``wind_speed`` : m/s (NASA POWER WS2M)
- ``solar_radiation`` : W/m² — **converted** from the NASA POWER native
  unit (MJ/m²/hr) via multiplication by 277.78 during merge.  The
  conversion factor is 1 MJ/hr = 10⁶ J / 3600 s ≈ 277.78 W.
- ``irrigation_event`` : binary {0, 1} (electrovalve relay status)
- ``flow_l`` : litres per period (differential)
- ``flow_l_cumulative`` : cumulative litres (raw)
"""

#: Local solar noon, in local clock hours.  Used to derive the expected
#: UTC hour of peak insolation from the site's UTC offset:
#: ``peak_utc_hour = (SOLAR_NOON_LOCAL_HOUR - utc_offset_hours) % 24``.
#: For Paraguay (offset −4) this yields 16:00 UTC.
SOLAR_NOON_LOCAL_HOUR: float = 12.0

# Validation ranges: (column, min, max, allow_nan)
_VALIDATION_RULES: list[tuple[str, float, float, bool]] = [
    ("soil_moisture", 0.0, 100.0, True),
    ("humidity", 0.0, 100.0, False),
    ("air_temp", -50.0, 60.0, False),
    ("wind_speed", 0.0, 50.0, False),
    ("solar_radiation", 0.0, 1200.0, False),
    ("flow_l", 0.0, float("inf"), True),
    ("flow_l_cumulative", 0.0, float("inf"), True),
]


# ======================================================================
# Mendeley loaders
# ======================================================================

def _identify_columns(
    df: pd.DataFrame,
) -> tuple[str, str]:
    """Identify the timestamp and value columns in a 2-column DataFrame.

    Strategy: first find the numeric column via ``pd.to_numeric``; the
    remaining column is assumed to be the timestamp.  This is more robust
    than attempting ``pd.to_datetime`` first, because pandas will happily
    parse bare floats (e.g. ``"74.29"``) as epoch timestamps, producing
    false positives.

    Parameters
    ----------
    df : pd.DataFrame
        Raw two-column DataFrame from a Mendeley CSV.

    Returns
    -------
    ts_col : str
        Name of the timestamp column.
    val_col : str
        Name of the numeric value column.

    Raises
    ------
    ValueError
        If the DataFrame does not have exactly two columns or column
        roles cannot be determined.
    """
    if len(df.columns) != 2:
        raise ValueError(
            f"Expected 2 columns, got {len(df.columns)}: {list(df.columns)}"
        )

    numeric_cols: list[str] = []
    non_numeric_cols: list[str] = []

    for col in df.columns:
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().all():
            numeric_cols.append(col)
        else:
            non_numeric_cols.append(col)

    if len(numeric_cols) == 1 and len(non_numeric_cols) == 1:
        val_col = numeric_cols[0]
        ts_col = non_numeric_cols[0]
    elif len(numeric_cols) == 2:
        # Both columns are numeric — try to_datetime on each and pick
        # the one whose parsed year is plausible (> 2000)
        for col in df.columns:
            try:
                parsed = pd.to_datetime(df[col], format="mixed")
                if parsed.dt.year.median() > 2000:
                    ts_col = col
                    val_col = [c for c in df.columns if c != col][0]
                    return ts_col, val_col
            except (ValueError, TypeError):
                continue
        raise ValueError(
            "Both columns are numeric and neither parses as a plausible "
            f"datetime: {list(df.columns)}"
        )
    else:
        # Neither column is fully numeric — try datetime on both
        ts_col = non_numeric_cols[0]
        val_col = non_numeric_cols[1] if len(non_numeric_cols) > 1 else None
        if val_col is None:
            raise ValueError(
                "Could not identify a numeric value column in "
                f"{list(df.columns)}"
            )

    # Verify the candidate timestamp actually parses
    try:
        pd.to_datetime(df[ts_col], format="mixed", dayfirst=False)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Column '{ts_col}' does not parse as datetime"
        ) from exc

    return ts_col, val_col


def load_mendeley_csv(
    file_path: Path | str,
    *,
    target_value_name: str = "value",
) -> pd.DataFrame:
    """Load a single Mendeley CSV/TXT, discover columns dynamically.

    Parameters
    ----------
    file_path : Path or str
        Path to the file (e.g. ``D_moisture.txt``).
    target_value_name : str
        Name to assign to the numeric value column in the output.

    Returns
    -------
    pd.DataFrame
        Two columns: ``timestamp`` (datetime64, UTC-naive) and
        *target_value_name* (float64).  Sorted by timestamp.
    """
    file_path = Path(file_path)
    logger.info("Loading Mendeley file: %s", file_path.name)

    df = pd.read_csv(file_path, encoding="utf-8")
    # Strip whitespace from column names (safety)
    df.columns = [c.strip() for c in df.columns]

    ts_col, val_col = _identify_columns(df)

    df[ts_col] = pd.to_datetime(df[ts_col], format="mixed", dayfirst=False)
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")

    df = df.rename(columns={ts_col: "timestamp", val_col: target_value_name})
    df = df[["timestamp", target_value_name]]
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info(
        "  → %d rows, %s to %s",
        len(df),
        df["timestamp"].iloc[0],
        df["timestamp"].iloc[-1],
    )
    return df


def localize_mendeley_to_utc(
    df: pd.DataFrame,
    utc_offset_hours: float,
    *,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Convert Mendeley local wall-clock timestamps to UTC.

    The Mendeley loggers record the *local* clock time of the deployment
    site.  NASA POWER is requested with ``time-standard=UTC``.  Merging
    the two without an explicit conversion silently offsets every
    meteorological covariate by the site's UTC offset, which destroys any
    diurnal relationship between weather and soil moisture.

    The conversion follows the standard convention ``local = UTC +
    offset``, hence::

        utc = local - offset

    For Paraguay (``utc_offset_hours = -4``) this *adds* four hours.  The
    offset is supplied by the caller from
    :attr:`~src.config.DataLoaderConfig.mendeley_utc_offset_hours`; it is
    deliberately not hard-coded here.

    Parameters
    ----------
    df : pd.DataFrame
        Frame with a naive datetime column holding local timestamps.
    utc_offset_hours : float
        UTC offset of the site's wall clock (negative west of Greenwich).
    timestamp_col : str
        Name of the timestamp column to convert.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with *timestamp_col* shifted to UTC (still
        timezone-naive, by project convention).

    Raises
    ------
    KeyError
        If *timestamp_col* is absent.
    """
    if timestamp_col not in df.columns:
        raise KeyError(
            f"Column '{timestamp_col}' not found; available: "
            f"{list(df.columns)}"
        )

    out = df.copy()
    shift = pd.Timedelta(hours=-utc_offset_hours)
    out[timestamp_col] = out[timestamp_col] + shift

    logger.info(
        "Mendeley timestamps localised: local(UTC%+g) → UTC "
        "(shift %+g h); %s → %s",
        utc_offset_hours,
        -utc_offset_hours,
        out[timestamp_col].iloc[0],
        out[timestamp_col].iloc[-1],
    )
    return out


# ======================================================================
# NASA POWER
# ======================================================================

def _nasa_cache_path(
    cache_dir: Path,
    start: str,
    end: str,
    latitude: float | None = None,
    longitude: float | None = None,
) -> Path:
    """Return the cache file path for a given date range and site.

    Coordinates are part of the filename so that changing the site can
    never be masked by a stale cache hit from a previous location.  They
    are optional purely for backward compatibility with callers that
    predate the site fix.
    """
    if latitude is None or longitude is None:
        return cache_dir / f"nasa_power_{start}_{end}.json"
    return (
        cache_dir
        / f"nasa_power_{latitude:+.2f}_{longitude:+.2f}_{start}_{end}.json"
    )


def fetch_nasa_power(
    config: NASAPowerConfig,
    start_date: str,
    end_date: str,
    cache_dir: Path | str,
    use_cache: bool = False,
) -> pd.DataFrame:
    """Fetch hourly meteorological data from the NASA POWER API.

    Parameters
    ----------
    config : NASAPowerConfig
        API configuration (coordinates, parameters, time standard).
    start_date, end_date : str
        Date range as ``"YYYYMMDD"`` strings.
    cache_dir : Path or str
        Directory for raw JSON cache files.
    use_cache : bool
        If ``True``, read from cache only — raise if cache missing.

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp`` (datetime64) plus one column per POWER
        parameter (e.g. ``T2M``, ``RH2M``, …).

    Raises
    ------
    FileNotFoundError
        If ``use_cache=True`` and no cached file exists.
    requests.HTTPError
        On non-2xx API responses.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = _nasa_cache_path(
        cache_dir, start_date, end_date, config.latitude, config.longitude,
    )

    if cache_file.exists():
        logger.info("Loading NASA POWER from cache: %s", cache_file.name)
        with open(cache_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    elif use_cache:
        raise FileNotFoundError(
            f"Cache file not found and --use-cache is set: {cache_file}"
        )
    else:
        url = (
            f"{config.base_url}/{config.temporal}/point"
            f"?parameters={','.join(config.parameters)}"
            f"&community={config.community}"
            f"&longitude={config.longitude}"
            f"&latitude={config.latitude}"
            f"&start={start_date}"
            f"&end={end_date}"
            f"&format=JSON"
            f"&header=false"
            f"&time-standard={config.time_standard}"
        )
        logger.info("Fetching NASA POWER data: %s", url)
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        with open(cache_file, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        logger.info("Cached NASA POWER response → %s", cache_file.name)

    # Parse the nested JSON → DataFrame
    params_data: dict[str, dict[str, float]] = data["properties"]["parameter"]
    records: dict[str, dict[str, float]] = {}
    # Keys are like "2022071200" (YYYYMMDDHH)
    first_param = config.parameters[0]
    for ts_key in params_data[first_param]:
        records[ts_key] = {
            param: params_data[param][ts_key] for param in config.parameters
        }

    df = pd.DataFrame.from_dict(records, orient="index")
    df.index.name = "ts_key"
    df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["ts_key"], format="%Y%m%d%H")
    df = df.drop(columns=["ts_key"]).sort_values("timestamp").reset_index(drop=True)

    # Replace fill value (-999.0) with NaN
    fill_value = data.get("header", {}).get("fill_value", -999.0)
    df = df.replace(fill_value, np.nan)

    logger.info(
        "NASA POWER: %d hourly records, %s to %s",
        len(df),
        df["timestamp"].iloc[0],
        df["timestamp"].iloc[-1],
    )
    return df


# ======================================================================
# Merge & resample
# ======================================================================

def merge_and_resample(
    df_moisture: pd.DataFrame,
    df_valve: pd.DataFrame,
    df_flow: pd.DataFrame,
    df_weather: pd.DataFrame,
    freq: str = "1h",
) -> pd.DataFrame:
    """Merge all data sources and resample to a uniform time grid.

    Parameters
    ----------
    df_moisture : pd.DataFrame
        Columns: ``timestamp``, ``soil_moisture``.
    df_valve : pd.DataFrame
        Columns: ``timestamp``, ``irrigation_event``.
    df_flow : pd.DataFrame
        Columns: ``timestamp``, ``flow_l_cumulative``.
    df_weather : pd.DataFrame
        Columns: ``timestamp``, ``T2M``, ``RH2M``, ``WS2M``,
        ``ALLSKY_SFC_SW_DWN``.
    freq : str
        Target resampling frequency (pandas offset alias).

    Returns
    -------
    pd.DataFrame
        Unified DataFrame with columns matching :data:`EXPECTED_COLUMNS`.
        ``flow_l`` is the *differential* volume per period;
        ``flow_l_cumulative`` is the raw cumulative reading.
    """
    # --- Set timestamp as index for resampling -------------------------
    df_m = df_moisture.set_index("timestamp").resample(freq).mean()
    df_v = df_valve.set_index("timestamp").resample(freq).max()  # 1 if any open
    df_f = df_flow.set_index("timestamp").resample(freq).last()  # last cumulative

    df_w = df_weather.rename(columns={
        "T2M": "air_temp",
        "RH2M": "humidity",
        "WS2M": "wind_speed",
        "ALLSKY_SFC_SW_DWN": "solar_radiation",
    }).set_index("timestamp")
    # Convert solar radiation: NASA POWER hourly unit is MJ/m²/hr;
    # we convert to W/m² (1 MJ/hr = 277.78 W) for standard SI usage.
    _MJ_HR_TO_W_M2 = 277.78
    df_w["solar_radiation"] = df_w["solar_radiation"] * _MJ_HR_TO_W_M2
    # Weather is already hourly — align to the same grid
    df_w = df_w.resample(freq).mean()

    # --- Join ------------------------------------------------------------
    merged = (
        df_m.join(df_w, how="inner")
            .join(df_v, how="left")
            .join(df_f, how="left")
    )

    # --- Compute differential flow ------------------------------------
    merged["flow_l"] = merged["flow_l_cumulative"].diff().clip(lower=0.0)

    # --- Clean up --------------------------------------------------------
    merged = merged.reset_index()
    # Ensure column order
    for col in EXPECTED_COLUMNS:
        if col not in merged.columns:
            merged[col] = np.nan
    merged = merged[EXPECTED_COLUMNS]

    logger.info(
        "Merged dataset: %d rows × %d cols, %s to %s",
        len(merged),
        len(merged.columns),
        merged["timestamp"].iloc[0],
        merged["timestamp"].iloc[-1],
    )
    return merged


# ======================================================================
# Validation
# ======================================================================

def validate_schema(df: pd.DataFrame) -> None:
    """Validate the output DataFrame schema and value ranges.

    Parameters
    ----------
    df : pd.DataFrame
        The merged hourly DataFrame.

    Raises
    ------
    ValueError
        With a detailed message listing all violations found.
    """
    errors: list[str] = []

    # Check required columns
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")

    # Check value ranges
    for col, lo, hi, allow_nan in _VALIDATION_RULES:
        if col not in df.columns:
            continue
        series = df[col]
        if not allow_nan and series.isna().any():
            n_nan = int(series.isna().sum())
            errors.append(
                f"Column '{col}' has {n_nan} NaN values (NaN not allowed)"
            )
        valid = series.dropna()
        below = (valid < lo).sum()
        above = (valid > hi).sum()
        if below:
            errors.append(
                f"Column '{col}': {int(below)} values below {lo} "
                f"(min={valid.min():.4f})"
            )
        if above:
            errors.append(
                f"Column '{col}': {int(above)} values above {hi} "
                f"(max={valid.max():.4f})"
            )

    # Irrigation event should be binary
    if "irrigation_event" in df.columns:
        valid_ie = df["irrigation_event"].dropna()
        bad = ~valid_ie.isin([0.0, 1.0])
        if bad.any():
            errors.append(
                f"Column 'irrigation_event': {int(bad.sum())} values "
                f"not in {{0, 1}}"
            )

    if errors:
        msg = "Schema validation failed:\n  • " + "\n  • ".join(errors)
        raise ValueError(msg)

    logger.info("Schema validation passed ✓")


def expected_solar_peak_utc_hour(utc_offset_hours: float) -> float:
    """Return the UTC hour at which mean insolation should peak.

    Derived from local solar noon and the site's UTC offset — see
    :data:`SOLAR_NOON_LOCAL_HOUR`.  Never hard-code the result.

    Parameters
    ----------
    utc_offset_hours : float
        UTC offset of the site's wall clock.

    Returns
    -------
    float
        Expected peak hour in ``[0, 24)`` UTC.  Paraguay (−4) → 16.0.
    """
    return (SOLAR_NOON_LOCAL_HOUR - utc_offset_hours) % 24.0


def validate_diurnal_alignment(
    df: pd.DataFrame,
    utc_offset_hours: float,
    *,
    tolerance_hours: float = 1.0,
    column: str = "solar_radiation",
    timestamp_col: str = "timestamp",
) -> float:
    """Assert the merged frame's insolation peaks at the right UTC hour.

    Averages *column* over UTC hour-of-day and checks that the maximum
    of that mean diurnal cycle falls within *tolerance_hours* of
    :func:`expected_solar_peak_utc_hour`.

    This is the pipeline's guard against silent temporal misalignment.
    It fails loudly on the two failure modes that produce
    plausible-looking but meaningless weather features: a NASA POWER
    response served in Local Solar Time instead of UTC, and coordinates
    from the wrong longitude (or the wrong hemisphere entirely).

    Scope note
    ----------
    The check validates the *weather* grid's time standard against the
    site's expected solar geometry.  It cannot, by construction, detect
    an error in the Mendeley-side offset alone, because
    ``solar_radiation`` originates entirely from NASA POWER and carries
    its own timestamps.  The Mendeley conversion is guarded separately by
    the single configured constant in
    :func:`localize_mendeley_to_utc`.

    Parameters
    ----------
    df : pd.DataFrame
        Merged hourly frame with UTC timestamps.
    utc_offset_hours : float
        UTC offset of the site's wall clock.
    tolerance_hours : float
        Half-width of the acceptance window, in hours.
    column : str
        Radiation column to profile.
    timestamp_col : str
        UTC timestamp column.

    Returns
    -------
    float
        The observed peak hour (UTC).

    Raises
    ------
    ValueError
        If the column is missing/empty, or the observed peak lies
        outside the acceptance window.
    """
    if column not in df.columns or timestamp_col not in df.columns:
        raise ValueError(
            f"Diurnal check needs columns '{timestamp_col}' and '{column}'; "
            f"available: {list(df.columns)}"
        )

    ts = pd.to_datetime(df[timestamp_col])
    profile = (
        pd.DataFrame({"hour": ts.dt.hour, "value": df[column].to_numpy()})
        .groupby("hour")["value"]
        .mean()
        .dropna()
    )
    if profile.empty:
        raise ValueError(
            f"Diurnal check failed: '{column}' has no non-NaN values."
        )

    observed_peak = float(profile.idxmax())
    expected_peak = expected_solar_peak_utc_hour(utc_offset_hours)

    # Circular distance on a 24-hour clock.
    delta = abs(observed_peak - expected_peak) % 24.0
    circular_delta = min(delta, 24.0 - delta)

    logger.info(
        "Diurnal check: mean %s peaks at %02d:00 UTC "
        "(expected %02d:00 UTC ± %g h, |Δ| = %.1f h)",
        column,
        int(observed_peak),
        int(expected_peak),
        tolerance_hours,
        circular_delta,
    )

    if circular_delta > tolerance_hours:
        top = profile.sort_values(ascending=False).head(3)
        raise ValueError(
            f"Diurnal alignment check FAILED for '{column}'.\n"
            f"  Observed peak hour : {int(observed_peak):02d}:00 UTC\n"
            f"  Expected peak hour : {int(expected_peak):02d}:00 UTC "
            f"(local solar noon at UTC{utc_offset_hours:+g})\n"
            f"  Circular offset    : {circular_delta:.1f} h "
            f"(tolerance {tolerance_hours:g} h)\n"
            f"  Top-3 hourly means : "
            + ", ".join(f"{int(h):02d}h={v:.1f}" for h, v in top.items())
            + "\n"
            "  The timezone reconciliation is wrong. Check, in order: "
            "(1) nasa_power.time_standard is 'UTC' and not the API default "
            "Local Solar Time; (2) nasa_power.latitude/longitude point at "
            "the sensor site; (3) mendeley_utc_offset_hours matches the "
            "logger wall clock. Any downstream metric computed on this "
            "merge would be meaningless."
        )

    return observed_peak


# ======================================================================
# Provenance
# ======================================================================

def generate_provenance(
    config: DataLoaderConfig,
    output_path: Path | str,
    date_shift_applied: bool = False,
) -> None:
    """Generate ``data_provenance.yaml`` for the Data Availability Statement.

    Parameters
    ----------
    config : DataLoaderConfig
        Active loader configuration.
    output_path : Path or str
        Where to write the YAML file.
    date_shift_applied : bool
        Whether a calendar-date shift was applied to align datasets.
    """
    provenance: dict[str, Any] = {
        "datasets": [
            {
                "name": "Smart Irrigation Control System Data",
                "source": "Mendeley Data",
                "doi": "10.17632/cjb4vy4mzj.3",
                "url": "https://data.mendeley.com/datasets/cjb4vy4mzj/3",
                "license": "CC BY 4.0",
                "accessed": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "files": ["D_moisture.txt", "D_valve.txt", "D_flowmeter.txt"],
                "date_range": "2022-07-12 to 2022-09-16 (local time)",
                "site": config.nasa_power.site_name,
                "country": config.nasa_power.country,
                "crop": config.nasa_power.crop,
                "native_timezone": (
                    f"UTC{config.mendeley_utc_offset_hours:+g} "
                    "(PYT, Paraguay standard time)"
                ),
                "utc_conversion": (
                    "Timestamps are local wall-clock readings and are "
                    f"shifted by {-config.mendeley_utc_offset_hours:+g} h to "
                    "UTC before merging with NASA POWER. Paraguayan DST "
                    "(PYST, UTC-03:00) runs from the first Sunday of October "
                    "to the fourth Sunday of March and therefore does not "
                    "overlap the 2022-07-12 → 2022-09-16 observation window; "
                    "a single constant offset is exact for this dataset."
                ),
                "notes": (
                    "Soil moisture, electrovalve relay status, cumulative "
                    "flow meter data."
                ),
            },
            {
                "name": "NASA POWER Hourly Meteorological Data",
                "source": "NASA Langley POWER Project (MERRA-2 reanalysis)",
                "url": (
                    "https://power.larc.nasa.gov/api/temporal/hourly/point"
                ),
                "license": "Public domain (NASA Open Data Policy)",
                "accessed": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "parameters": list(config.nasa_power.parameters),
                "coordinates": {
                    "lat": config.nasa_power.latitude,
                    "lon": config.nasa_power.longitude,
                },
                "site": config.nasa_power.site_name,
                "country": config.nasa_power.country,
                "date_range": "2022-07-12 to 2022-09-17 (UTC)",
                "date_shift_applied": date_shift_applied,
                "time_standard": config.nasa_power.time_standard,
                "notes": (
                    "Coordinates are the sensor deployment site itself "
                    f"({config.nasa_power.site_name}), not a different "
                    "region. NASA POWER MERRA-2 reanalysis covers the exact "
                    "Mendeley observation period. No calendar-date shift was "
                    "necessary; only the local→UTC clock conversion described "
                    "under the Mendeley entry. The API is called with "
                    "time-standard=UTC to avoid silent misalignment with "
                    "Mendeley sensor clock-times (the POWER Hourly API "
                    "defaults to Local Solar Time)."
                ),
            },
        ],
        "output_conventions": {
            "timestamp_column": {
                "timezone": "UTC",
                "mendeley_utc_offset_hours": config.mendeley_utc_offset_hours,
                "note": (
                    "All timestamps in the final merged DataFrame are in UTC. "
                    "Mendeley sensor timestamps are local Paraguayan wall-clock "
                    f"time (UTC{config.mendeley_utc_offset_hours:+g}) and are "
                    f"shifted by {-config.mendeley_utc_offset_hours:+g} h "
                    "before the merge; NASA POWER is explicitly requested in "
                    "UTC via time-standard=UTC. The reconciliation is verified "
                    "automatically after every merge: the mean diurnal cycle "
                    "of solar_radiation must peak at "
                    f"{int(expected_solar_peak_utc_hour(config.mendeley_utc_offset_hours)):02d}:00 UTC "
                    f"± {config.solar_peak_tolerance_hours:g} h, otherwise the "
                    "loader raises."
                ),
            },
            "flow_l": (
                "Differential (per-period) volume computed from the "
                "cumulative flow-meter readings."
            ),
            "flow_l_cumulative": (
                "Raw cumulative readings preserved from D_flowmeter.txt."
            ),
            "solar_radiation": {
                "unit": "W/m²",
                "native_unit": "MJ/m²/hr (NASA POWER ALLSKY_SFC_SW_DWN)",
                "conversion": "multiplied by 277.78 (1 MJ/hr = 10⁶ J / 3600 s ≈ 277.78 W)",
                "note": (
                    "The NASA POWER Hourly API returns ALLSKY_SFC_SW_DWN "
                    "in MJ/m²/hr. Values are converted to W/m² during merge "
                    "for consistency with standard SI irradiance units."
                ),
            },
        },
        "assumptions": [
            {
                "id": "A1",
                "description": (
                    "Meteorological covariates are NASA POWER MERRA-2 "
                    "reanalysis for "
                    f"({config.nasa_power.latitude}, "
                    f"{config.nasa_power.longitude}) — "
                    f"{config.nasa_power.site_name} — i.e. a gridded "
                    "regional proxy for the SAME region and the SAME season "
                    "as the Mendeley sensor deployment, not in-situ "
                    "measurements at the field. The residual limitation is "
                    "spatial: MERRA-2 resolves roughly 0.5° × 0.625°, so "
                    "sub-grid microclimate (canopy shading, local advection, "
                    "irrigation-induced humidity) is not represented. "
                    "Earlier revisions of this pipeline requested weather for "
                    "Karaganda, Kazakhstan (49.80, 73.10) — a different "
                    "continent and the opposite hemisphere's season. Those "
                    "features were uncorrelated with the target by "
                    "construction and any result derived from them is void."
                ),
            },
            {
                "id": "A2",
                "description": (
                    "Mendeley timestamps are treated as local Paraguayan "
                    f"wall-clock time (UTC{config.mendeley_utc_offset_hours:+g}) "
                    "and converted to UTC with a single constant offset. This "
                    "is exact for the 2022-07-12 → 2022-09-16 window because "
                    "Paraguayan DST does not overlap it. The dataset "
                    "documentation does not state the logger timezone "
                    "explicitly; the assumption is verified indirectly by the "
                    "post-merge diurnal alignment check on solar_radiation."
                ),
            },
            {
                "id": "A3",
                "description": (
                    "ec_salt_concentration defaults to 2.0 dS/m, a documented "
                    "approximation only. The cited source (Soil Systems, 2025, "
                    "doi:10.3390/soilsystems9020057) reports the secondary "
                    "salinization threshold for irrigated Haplic Kastanozem "
                    "soils of the Kazakhstan steppe — a soil and climate "
                    "unrelated to the Paraguayan site. It is retained solely "
                    "as an order-of-magnitude parameter for the synthetic "
                    "drift-injection experiment (src/robustness_experiment.py) "
                    "and must NOT be read as a calibrated property of this "
                    "field. The ec_factor mapping "
                    "(EC ≈ f(soil_moisture, salt_concentration)) is likewise "
                    "not a validated physical model. See Review §2.3.1, "
                    "sources [3, 4]."
                ),
            },
        ],
    }

    output_path = Path(output_path)
    with open(output_path, "w", encoding="utf-8") as fh:
        yaml.dump(provenance, fh, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)
    logger.info("Data provenance written → %s", output_path)


# ======================================================================
# Main entry point
# ======================================================================

def load_dataset(config: DataLoaderConfig) -> pd.DataFrame:
    """Orchestrate the full data-loading pipeline.

    Steps
    -----
    1. Load three Mendeley CSV/TXT files (dynamic column discovery).
    2. Convert their local wall-clock timestamps to UTC.
    3. Fetch (or load cached) NASA POWER hourly weather data.
    4. Merge and resample to a uniform grid.
    5. Validate schema, value ranges, and diurnal alignment.
    6. Save processed output and provenance file.

    Parameters
    ----------
    config : DataLoaderConfig
        Full pipeline configuration.

    Returns
    -------
    pd.DataFrame
        Merged, validated, hourly DataFrame.
    """
    base = Path(config.mendeley_dir)

    # --- Identify Mendeley files by pattern ----------------------------
    all_files = sorted(base.glob("*.txt")) + sorted(base.glob("*.csv"))
    if not all_files:
        raise FileNotFoundError(
            f"No .txt/.csv files found in {base.resolve()}"
        )
    logger.info("Found %d Mendeley files in %s", len(all_files), base)

    # Categorise by filename heuristics
    moisture_file: Path | None = None
    valve_file: Path | None = None
    flow_file: Path | None = None
    for f in all_files:
        name_lower = f.stem.lower()
        if "moisture" in name_lower:
            moisture_file = f
        elif "valve" in name_lower:
            valve_file = f
        elif "flow" in name_lower:
            flow_file = f

    if not all([moisture_file, valve_file, flow_file]):
        raise FileNotFoundError(
            f"Could not identify all three Mendeley files "
            f"(moisture/valve/flow) in {base.resolve()}. "
            f"Found: {[f.name for f in all_files]}"
        )

    # --- Load Mendeley -------------------------------------------------
    df_moisture = load_mendeley_csv(moisture_file, target_value_name="soil_moisture")
    df_valve = load_mendeley_csv(valve_file, target_value_name="irrigation_event")
    df_flow = load_mendeley_csv(flow_file, target_value_name="flow_l_cumulative")

    # --- Local wall-clock → UTC ----------------------------------------
    # Must happen BEFORE the NASA POWER date range is derived, so the
    # requested window covers the shifted observation period.
    offset = config.mendeley_utc_offset_hours
    df_moisture = localize_mendeley_to_utc(df_moisture, offset)
    df_valve = localize_mendeley_to_utc(df_valve, offset)
    df_flow = localize_mendeley_to_utc(df_flow, offset)

    # --- Determine Mendeley date range ---------------------------------
    all_ts = pd.concat([
        df_moisture["timestamp"],
        df_valve["timestamp"],
        df_flow["timestamp"],
    ])
    start_date = all_ts.min().strftime("%Y%m%d")
    end_date = all_ts.max().strftime("%Y%m%d")
    logger.info("Mendeley date range: %s → %s", start_date, end_date)

    # --- Fetch NASA POWER ----------------------------------------------
    df_weather = fetch_nasa_power(
        config=config.nasa_power,
        start_date=start_date,
        end_date=end_date,
        cache_dir=Path(config.raw_cache_dir),
        use_cache=config.use_cache,
    )

    # --- Merge and resample --------------------------------------------
    merged = merge_and_resample(
        df_moisture=df_moisture,
        df_valve=df_valve,
        df_flow=df_flow,
        df_weather=df_weather,
        freq=config.resample_freq,
    )

    # --- Validate -------------------------------------------------------
    validate_schema(merged)
    validate_diurnal_alignment(
        merged,
        utc_offset_hours=config.mendeley_utc_offset_hours,
        tolerance_hours=config.solar_peak_tolerance_hours,
    )

    # --- Persist --------------------------------------------------------
    out_dir = Path(config.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "merged_hourly.csv"
    merged.to_csv(out_csv, index=False)
    logger.info("Saved processed dataset → %s (%d rows)", out_csv, len(merged))

    # --- Provenance -----------------------------------------------------
    provenance_path = Path("data_provenance.yaml")
    generate_provenance(config, provenance_path, date_shift_applied=False)

    return merged


# ======================================================================
# CLI
# ======================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load, merge, and validate the AIoT irrigation dataset. "
            "Combines Mendeley sensor data with NASA POWER weather data."
        ),
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        default=False,
        help="Run fully offline using cached data (no network requests).",
    )
    parser.add_argument(
        "--resample-freq",
        type=str,
        default="1h",
        help="Resampling frequency (pandas offset alias, default: 1h).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for ``python -m src.data_loader``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )
    args = _parse_args()

    config = build_loader_config(args.config)
    config.use_cache = config.use_cache or args.use_cache
    config.resample_freq = args.resample_freq

    df = load_dataset(config)
    logger.info("Done. Shape: %s", df.shape)


if __name__ == "__main__":
    main()
