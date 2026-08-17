"""Causal feature engineering for irrigation-event forecasting.

The pipeline predicts ``irrigation_event(t)`` — will the electrovalve be
open during hour *t*? — from information available **strictly before**
hour *t*.  This module is the single place where that contract is
enforced.

The contract
------------
For every engineered column *f* and every row *t*::

    f(t) = g( x(t-1), x(t-2), … )

No feature may read ``x(t)``.  In particular ``soil_moisture(t)`` is
*excluded by construction*: soil moisture rises as a **consequence** of
the valve opening, so conditioning on it turns forecasting into
after-the-fact detection.  Earlier revisions of this project trained on
``soil_moisture(t)`` and reported the result as a prediction; that is the
defect this module exists to prevent.

The contract is machine-checked, not merely documented — see
``tests/test_features.py::TestNoLookAhead``, which overwrites the tail of
the input with NaN and asserts that no feature value before the cut
changes.

Feature blocks
--------------
``moisture``
    Lags, drying-rate first differences, and causal rolling
    mean/min/max/std of ``soil_moisture``.
``weather``
    Lags and causal rolling means of air temperature, humidity, wind
    speed and solar radiation.  Cumulative evaporative demand over the
    preceding day carries far more signal than any instantaneous
    reading.
``calendar``
    Hour-of-day as sine/cosine, plus days elapsed since the start of the
    record.  Deterministic functions of the clock, hence legitimately
    known at prediction time.
``irrigation``
    Autoregressive lags of the target and hours since the last observed
    irrigation.

Blocks are returned as a mapping so that the ablation study
(:mod:`src.evaluate_pipeline`) can compose feature sets from them
without re-deriving the column names by string matching.

Leakage guard
-------------
``flow_l`` and ``flow_l_cumulative`` are the metered volume delivered by
the valve.  They are consequences of the event being predicted, at any
lag, and :func:`assert_no_forbidden_features` raises if they ever appear
in a feature set.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config import FeatureConfig

logger = logging.getLogger(__name__)

# ── Block names ───────────────────────────────────────────────────────

BLOCK_MOISTURE: str = "moisture"
BLOCK_WEATHER: str = "weather"
BLOCK_CALENDAR: str = "calendar"
BLOCK_IRRIGATION: str = "irrigation"

BLOCK_ORDER: Tuple[str, ...] = (
    BLOCK_MOISTURE, BLOCK_WEATHER, BLOCK_CALENDAR, BLOCK_IRRIGATION,
)

#: Rolling statistics emitted for the soil-moisture block.
_ROLLING_STATS: Tuple[str, ...] = ("mean", "min", "max", "std")


# ======================================================================
# Episode structure
# ======================================================================

def find_episodes(
    target: pd.Series,
    timestamps: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Locate contiguous runs of irrigation in the hourly record.

    An *episode* is a maximal run of consecutive hours with the valve
    open.  Episodes, not individual hours, are what an irrigation
    schedule consists of, so their count and duration describe the target
    far better than a positive rate does: 23.6 % of hours positive is
    consistent both with continuous trickle irrigation and with a handful
    of multi-day floods, and the two call for different models.

    Missing relay readings are treated as **breaks**, not as
    continuations.  An episode that actually spanned a data gap is
    therefore counted as two, which understates duration and overstates
    count.  The alternative — bridging gaps — would invent irrigation
    that was never recorded.  Callers should report the number of unknown
    hours alongside these figures; :func:`describe_episodes` does.

    Args:
        target: Binary irrigation series on a uniform hourly grid, in
            time order.  NaN marks an unknown hour.
        timestamps: Optional aligned timestamps, used to report when each
            episode began and ended.

    Returns:
        One row per episode with ``start_index``, ``end_index``,
        ``length_hours`` and, when *timestamps* is given, ``start_time``
        and ``end_time``.  Empty if the record contains no irrigation.
    """
    values = pd.Series(target).to_numpy(dtype=float)
    is_on = values == 1.0

    if not is_on.any():
        return pd.DataFrame(
            columns=["start_index", "end_index", "length_hours"]
        )

    # Pad with False so runs starting at row 0 or ending at the last row
    # are detected by the same difference operation.
    padded = np.concatenate([[False], is_on, [False]])
    edges = np.diff(padded.astype(int))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1

    episodes = pd.DataFrame({
        "start_index": starts,
        "end_index": ends,
        "length_hours": ends - starts + 1,
    })

    if timestamps is not None:
        stamps = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True))
        episodes["start_time"] = stamps.iloc[starts].to_numpy()
        episodes["end_time"] = stamps.iloc[ends].to_numpy()

    return episodes


def describe_episodes(
    target: pd.Series,
    timestamps: Optional[pd.Series] = None,
    *,
    histogram_bins: Sequence[int] = (1, 2, 3, 6, 12, 24, 48, 96),
) -> Dict[str, object]:
    """Summarise the episode structure for the run's metadata.

    Args:
        target: Binary irrigation series on a uniform hourly grid.
        timestamps: Optional aligned timestamps.
        histogram_bins: Left edges of the duration histogram, in hours.
            The final bin is open-ended.

    Returns:
        JSON-serialisable mapping with the episode count, duration
        quantiles, a duration histogram, and the number of unknown hours
        that limit how precisely episodes can be delimited.
    """
    series = pd.Series(target)
    episodes = find_episodes(series, timestamps)
    lengths = episodes["length_hours"].to_numpy() if len(episodes) else np.array([])

    summary: Dict[str, object] = {
        "n_episodes": int(len(episodes)),
        "n_irrigating_hours": int((series == 1.0).sum()),
        "n_unknown_hours": int(series.isna().sum()),
        "note": (
            "An episode is a maximal run of consecutive irrigating hours. "
            "Unknown relay hours break a run rather than bridging it, so an "
            "episode spanning a data gap is counted as two: the count is an "
            "upper bound and the durations a lower bound."
        ),
    }

    if len(lengths) == 0:
        summary["duration_hours"] = None
        summary["duration_histogram_hours"] = {}
        return summary

    summary["duration_hours"] = {
        "min": int(lengths.min()),
        "q25": float(np.quantile(lengths, 0.25)),
        "median": float(np.median(lengths)),
        "q75": float(np.quantile(lengths, 0.75)),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "std": float(lengths.std(ddof=1)) if len(lengths) > 1 else 0.0,
        "total": int(lengths.sum()),
    }

    edges = list(histogram_bins)
    histogram: Dict[str, int] = {}
    for i, low in enumerate(edges):
        if i + 1 < len(edges):
            high = edges[i + 1]
            count = int(((lengths >= low) & (lengths < high)).sum())
            label = f"{low}" if high == low + 1 else f"{low}-{high - 1}"
        else:
            count = int((lengths >= low).sum())
            label = f"{low}+"
        histogram[label] = count
    summary["duration_histogram_hours"] = histogram

    return summary


def irrigation_onset(
    target: pd.Series,
    config: FeatureConfig | None = None,
) -> pd.Series:
    """Mark the first hour of each irrigation episode.

    ``onset(t) = 1`` when the valve is open at *t* and was closed at
    *t − 1*.  This is the decision an irrigation controller actually
    makes; the hours that follow are continuations of a decision already
    taken.

    The first row is NaN, not 0: whether the record opens mid-episode is
    unknowable, and calling it "not an onset" would assert something the
    data does not say.

    Args:
        target: Binary irrigation series on a uniform hourly grid.
        config: Supplies the causal shift; the previous state is read at
            ``t − causal_shift``.

    Returns:
        Series aligned to *target* with 1 at episode starts, 0 elsewhere,
        and NaN where the previous state is unknown.
    """
    cfg = config or FeatureConfig()
    series = pd.Series(target).astype(float)
    previous = series.shift(cfg.causal_shift)

    onset = ((series == 1.0) & (previous == 0.0)).astype(float)
    # Propagate genuine ignorance rather than defaulting to zero.
    onset[series.isna() | previous.isna()] = np.nan
    return onset


# ======================================================================
# Leakage guard
# ======================================================================

def assert_no_forbidden_features(
    feature_names: Iterable[str],
    config: FeatureConfig | None = None,
) -> None:
    """Raise if any flow-meter–derived column reached the feature set.

    ``flow_l`` and ``flow_l_cumulative`` measure water that flowed
    *because* the valve opened.  A model given them — directly or as a
    lag, difference or rolling statistic — is reading the answer off the
    water meter.  Because the column names of engineered features are
    prefixed with their source column, a prefix test catches every
    derived form.

    Parameters
    ----------
    feature_names : Iterable[str]
        Column names about to be handed to an estimator.
    config : FeatureConfig, optional
        Supplies :attr:`~src.config.FeatureConfig.forbidden_cols`.

    Raises
    ------
    ValueError
        Listing every offending column.
    """
    cfg = config or FeatureConfig()
    names = list(feature_names)

    offenders: List[str] = []
    for name in names:
        for banned in cfg.forbidden_cols:
            if name == banned or name.startswith(f"{banned}_"):
                offenders.append(name)
                break

    if offenders:
        raise ValueError(
            "Forbidden flow-meter features present in the feature set: "
            f"{sorted(set(offenders))}.\n"
            "flow_l and flow_l_cumulative are direct consequences of the "
            "electrovalve opening — the metered volume delivered by the very "
            "irrigation event being predicted. Including them at any lag is "
            "target leakage, not feature engineering. Remove them from the "
            "feature set (they may remain in the DataFrame for reporting)."
        )


# ======================================================================
# Block builders
# ======================================================================

def _causal(series: pd.Series, config: FeatureConfig) -> pd.Series:
    """Shift *series* so the newest usable observation is ``t - shift``."""
    return series.shift(config.causal_shift)


def _min_periods(window: int, config: FeatureConfig) -> int:
    """Return ``min_periods`` for a rolling window of *window* hours."""
    return window if config.require_full_window else 1


def _build_moisture_block(
    df: pd.DataFrame, config: FeatureConfig,
) -> Tuple[Dict[str, pd.Series], List[str]]:
    """Lags, drying rates and causal rolling statistics of soil moisture.

    The rolling windows are computed on the already-shifted series, so a
    ``W``-hour window on row *t* spans ``[t - W, t - 1]``.
    """
    out: Dict[str, pd.Series] = {}
    names: List[str] = []

    raw = df[config.moisture_col]
    base = _causal(raw, config)          # value at t-1
    col = config.moisture_col

    for lag in config.moisture_lags:
        name = f"{col}_lag{lag}h"
        out[name] = raw.shift(lag)
        names.append(name)

    # Drying rate: change over the H hours ending at t-1. Positive means
    # the soil got wetter, negative means it dried out.
    for horizon in config.moisture_diff_lags:
        name = f"{col}_diff{horizon}h"
        out[name] = base - raw.shift(config.causal_shift + horizon)
        names.append(name)

    for window in config.moisture_roll_windows:
        roller = base.rolling(
            window=window, min_periods=_min_periods(window, config),
        )
        for stat in _ROLLING_STATS:
            name = f"{col}_roll{window}h_{stat}"
            out[name] = getattr(roller, stat)()
            names.append(name)

    return out, names


def _build_weather_block(
    df: pd.DataFrame, config: FeatureConfig,
) -> Tuple[Dict[str, pd.Series], List[str]]:
    """Lags and causal rolling means of the meteorological covariates.

    Only **means** are emitted, not sums.  Under a fully populated window
    the two are related by ``sum = window × mean``, i.e. perfectly
    collinear: emitting both would add no information while splitting
    each variable's SHAP attribution across two identical columns and
    inflating the apparent feature count.  Multiply by the window width
    to recover the integral (e.g. daily insolation) if a physical unit is
    wanted for reporting.
    """
    out: Dict[str, pd.Series] = {}
    names: List[str] = []

    for col in config.weather_cols:
        if col not in df.columns:
            raise KeyError(
                f"Weather column '{col}' not found; available: "
                f"{list(df.columns)}"
            )
        raw = df[col]
        base = _causal(raw, config)

        for lag in config.weather_lags:
            name = f"{col}_lag{lag}h"
            out[name] = raw.shift(lag)
            names.append(name)

        for window in config.weather_roll_windows:
            name = f"{col}_roll{window}h_mean"
            out[name] = base.rolling(
                window=window, min_periods=_min_periods(window, config),
            ).mean()
            names.append(name)

    return out, names


def _build_calendar_block(
    df: pd.DataFrame, config: FeatureConfig,
) -> Tuple[Dict[str, pd.Series], List[str]]:
    """Hour-of-day sine/cosine and days elapsed since the first record.

    These are deterministic functions of the timestamp, not of any
    measurement, so they are known arbitrarily far in advance and do not
    violate the causality contract.  The sine/cosine pair keeps 23:00 and
    00:00 adjacent, which a raw integer hour would not.
    """
    ts = pd.to_datetime(df[config.timestamp_col])
    angle = 2.0 * np.pi * ts.dt.hour / config.hours_per_day

    out: Dict[str, pd.Series] = {
        "hour_sin": pd.Series(np.sin(angle), index=df.index),
        "hour_cos": pd.Series(np.cos(angle), index=df.index),
        "days_since_start": pd.Series(
            (ts - ts.iloc[0]).dt.total_seconds()
            / (config.hours_per_day * 3600.0),
            index=df.index,
        ),
    }
    return out, list(out)


def _hours_since_last_irrigation(
    target: pd.Series, config: FeatureConfig,
) -> pd.Series:
    """Hours elapsed since the most recent irrigation at or before t-1.

    Computed by carrying forward the positional index of the last
    observed event.  The forward fill runs over the *shifted* event
    series, so row *t* can only ever see events at ``t - 1`` or earlier;
    if irrigation ran during hour ``t - 1`` the feature equals 1.

    Rows preceding the first event in the record are NaN — there is no
    honest value to impute there — and are dropped downstream with the
    other warm-up rows.
    """
    positions = np.arange(len(target), dtype=float)
    # Treat a missing relay reading as "no event observed" rather than
    # silently carrying the previous event forward across the gap.
    is_event = target.fillna(0).to_numpy() == 1
    event_positions = np.where(is_event, positions, np.nan)

    last_event = (
        pd.Series(event_positions, index=target.index)
        .shift(config.causal_shift)
        .ffill()
    )
    return pd.Series(positions, index=target.index) - last_event


def _build_irrigation_block(
    df: pd.DataFrame, config: FeatureConfig,
) -> Tuple[Dict[str, pd.Series], List[str]]:
    """Autoregressive target lags and time since the last irrigation.

    Block placement note
    --------------------
    ``hours_since_last_irrigation`` is grouped here rather than with the
    soil-moisture features, even though it describes the irrigation
    *regime* around the moisture signal.  It is derived from the target
    series, so grouping it with the moisture block would quietly leak
    target history into ablation sets A–C and destroy the comparison
    those sets exist to make.
    """
    out: Dict[str, pd.Series] = {}
    names: List[str] = []

    target = df[config.target_col]
    col = config.target_col

    for lag in config.target_lags:
        name = f"{col}_lag{lag}h"
        out[name] = target.shift(lag)
        names.append(name)

    out["hours_since_last_irrigation"] = _hours_since_last_irrigation(
        target, config,
    )
    names.append("hours_since_last_irrigation")

    return out, names


# ======================================================================
# Public API
# ======================================================================

def build_features(
    df: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Build the full causal feature matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Merged hourly frame, sorted ascending by time, containing at
        least the timestamp, soil-moisture, weather and target columns.
        A uniform hourly grid is assumed: lags are positional, so a gap
        in the index would silently change their meaning.
    config : FeatureConfig, optional
        Feature definition; defaults to :class:`FeatureConfig`.

    Returns
    -------
    features : pd.DataFrame
        The timestamp and target columns followed by every engineered
        feature, on the same index as *df*.  Leading rows contain NaN
        until every window is populated; use :func:`prepare_supervised`
        to drop them.
    blocks : dict[str, list[str]]
        Feature names grouped by block (see :data:`BLOCK_ORDER`).

    Raises
    ------
    ValueError
        If required columns are missing, the frame is not sorted by
        time, or ``causal_shift < 1``.
    """
    cfg = config or FeatureConfig()

    if cfg.causal_shift < 1:
        raise ValueError(
            f"causal_shift must be >= 1 to keep features causal, "
            f"got {cfg.causal_shift}. A shift of 0 would let row t read "
            f"its own measurements."
        )

    required = [cfg.timestamp_col, cfg.moisture_col, cfg.target_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    ts = pd.to_datetime(df[cfg.timestamp_col])
    if not ts.is_monotonic_increasing:
        raise ValueError(
            "Input must be sorted ascending by timestamp: lags are "
            "positional and would otherwise reach forward in time."
        )

    builders = {
        BLOCK_MOISTURE: _build_moisture_block,
        BLOCK_WEATHER: _build_weather_block,
        BLOCK_CALENDAR: _build_calendar_block,
        BLOCK_IRRIGATION: _build_irrigation_block,
    }

    columns: Dict[str, pd.Series] = {}
    blocks: Dict[str, List[str]] = {}
    for block in BLOCK_ORDER:
        block_cols, block_names = builders[block](df, cfg)
        columns.update(block_cols)
        blocks[block] = block_names

    features = pd.DataFrame(columns, index=df.index)
    features.insert(0, cfg.target_col, df[cfg.target_col].to_numpy())
    features.insert(0, cfg.timestamp_col, ts.to_numpy())

    all_names = [n for block in BLOCK_ORDER for n in blocks[block]]
    assert_no_forbidden_features(all_names, cfg)

    logger.info(
        "Built %d causal features from %d rows (%s)",
        len(all_names),
        len(features),
        ", ".join(f"{b}={len(blocks[b])}" for b in BLOCK_ORDER),
    )
    return features, blocks


def prepare_supervised(
    features: pd.DataFrame,
    feature_names: Sequence[str],
    config: FeatureConfig | None = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Drop warm-up rows and split into ``(X, y, timestamps)``.

    Rows are dropped only where a *selected* feature or the target is
    NaN, so a narrow feature set (e.g. ablation set A) is not penalised
    by the warm-up period of features it does not use.

    Parameters
    ----------
    features : pd.DataFrame
        Output of :func:`build_features`.
    feature_names : Sequence[str]
        Columns to retain, in order.
    config : FeatureConfig, optional
        Supplies the target and timestamp column names, and the
        forbidden-column list used for the leakage guard.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with a fresh ``RangeIndex``.
    y : pd.Series
        Integer target aligned to *X*.
    timestamps : pd.Series
        UTC timestamps aligned to *X*, for chronological splitting and
        for reporting which period a fold covers.

    Raises
    ------
    ValueError
        If a requested column is absent, a forbidden column was
        requested, or no rows survive.
    """
    cfg = config or FeatureConfig()
    names = list(feature_names)

    absent = [c for c in names if c not in features.columns]
    if absent:
        raise ValueError(
            f"Requested features not present: {absent}. "
            f"Available: {list(features.columns)}"
        )

    assert_no_forbidden_features(names, cfg)

    subset = features[[cfg.timestamp_col, cfg.target_col] + names]
    complete = subset.dropna()

    if complete.empty:
        raise ValueError(
            "No complete rows remain after dropping the warm-up period. "
            f"The widest window among the {len(names)} selected features "
            f"exceeds the {len(features)} available rows."
        )

    n_dropped = len(features) - len(complete)
    logger.info(
        "Supervised matrix: %d rows × %d features "
        "(dropped %d warm-up/incomplete rows), positive rate %.4f",
        len(complete),
        len(names),
        n_dropped,
        complete[cfg.target_col].mean(),
    )

    X = complete[names].reset_index(drop=True)
    y = complete[cfg.target_col].astype(int).reset_index(drop=True)
    timestamps = complete[cfg.timestamp_col].reset_index(drop=True)
    return X, y, timestamps
