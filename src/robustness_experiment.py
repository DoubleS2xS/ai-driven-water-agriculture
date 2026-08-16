"""Robustness study: sensor degradation and attempted recovery.

Deliberately **outside** the main pipeline
------------------------------------------
:mod:`src.evaluate_pipeline` answers "can irrigation events be forecast
from past observations?".  This module answers a different and secondary
question: "what happens to that forecast when the moisture sensor
degrades, and does the healing stack recover it?".

Keeping the two separate is not organisational tidiness.  In the earlier
revision this experiment *was* the headline result, and the comparison
Clean → Corrupted → Healed was presented as the study's contribution.
Two problems make that untenable, both measured rather than suspected,
and both reported below without adjustment.

Finding 1 — the healing stack makes the signal worse
----------------------------------------------------
Correlation of the reconstructed soil-moisture series with ground truth:

===========================  ===========
Variant                      Correlation
===========================  ===========
Corrupted, no healing              0.877
MICE imputation                    0.787
MICE + DriftCompensator            0.520
===========================  ===========

Each healing stage moves the signal *further* from the truth.  The
compensator is the worse offender: it subtracts a rolling-minimum
baseline, which removes genuine low-frequency variation in soil moisture
along with the injected drift — and low-frequency variation is precisely
the part that predicts irrigation.

This is recorded as a measured result.  It is not tuned away.  Adjusting
the compensator's window until the number improves would be fitting the
recovery method to the outcome, and the honest report of a method that
does not work is more useful than a tuned number that flatters it.

Finding 2 — the injected drift is a clock, so "corrupted" can win
-----------------------------------------------------------------
The drift model is ``drift(t) = a·(1 − e^{−b·t})`` with *t* reset to zero
every 35–40 hours.  Resetting a monotone ramp at a near-fixed interval
produces a **regular sawtooth**: a periodic, almost noise-free signal
superimposed on the sensor channel.

A gradient-boosted tree can read that sawtooth as a clock.  Because the
irrigation schedule itself has temporal structure, knowing the phase of a
~37-hour cycle is genuinely informative — so a model trained on
*corrupted* data sometimes scores **above** one trained on clean data.

That is an artifact of the injection model, not a discovery about sensor
noise, and any result of the form "degradation improved performance" from
this experiment must be read that way.  A study that wants to make real
claims about drift robustness needs an **aperiodic** drift model — a
random walk, or resets drawn from a heavy-tailed interval distribution —
so that no phase information is available to be exploited.

Protocol
--------
The forecasting protocol is identical to the main pipeline's: causal
features only, rolling-origin cross-validation, preprocessing fitted
inside each training fold.  Only the soil-moisture channel differs
between variants.  Every variant is trained on its own degraded data and
evaluated against the **clean** ground-truth labels, since the valve
record is not what degrades.

Usage
-----
::

    python -m src.robustness_experiment
    python -m src.robustness_experiment --missing-rate 0.2 --model lightgbm
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import (
    FeatureConfig,
    MissingDataConfig,
    SensorDriftConfig,
    ValidationConfig,
)
from src.data_corruption import inject_missing, inject_sensor_drift
from src.data_healing import DataImputer, DriftCompensator
from src.evaluate_pipeline import (
    DEFAULT_SEED,
    OUTPUT_DIR,
    PROCESSED_CSV,
    build_design_matrix,
    evaluate_on_splits,
    load_modeling_frame,
    make_model_factory,
)
from src.metrics import PRIMARY_METRICS
from src.statistics import summarize_metric
from src.validation import rolling_origin_splits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

TARGET_COLUMN: str = "soil_moisture"

#: Channels MICE may use to reconstruct the degraded one.  The target and
#: the flow meter are excluded: imputing a sensor from the label it is
#: meant to predict would leak, and the flow meter is a consequence of
#: the valve opening.
IMPUTATION_COLUMNS: Tuple[str, ...] = (
    "soil_moisture", "air_temp", "humidity", "wind_speed", "solar_radiation",
)

ROBUSTNESS_SIGNAL_CSV: str = "robustness_signal_quality.csv"
ROBUSTNESS_FORECAST_CSV: str = "robustness_forecast.csv"


# ======================================================================
# Variant construction
# ======================================================================

def build_variants(
    df_clean: pd.DataFrame,
    *,
    missing_config: MissingDataConfig,
    drift_config: SensorDriftConfig,
    mice_max_iter: int = 10,
    mice_random_state: int = 42,
    compensator_window_hours: int = 24,
) -> Dict[str, pd.DataFrame]:
    """Build the clean, corrupted and two healed variants.

    Args:
        df_clean: Merged hourly frame, unmodified.
        missing_config: Missingness injection settings.
        drift_config: Drift injection settings.
        mice_max_iter: IterativeImputer iterations.
        mice_random_state: Seed for MICE.
        compensator_window_hours: Rolling window for drift compensation.

    Returns:
        Ordered mapping of variant name to frame.  The two healing stages
        are kept apart so their individual contributions are visible;
        collapsing them into one "healed" variant is what hid the
        compensator's damage in the earlier revision.
    """
    variants: Dict[str, pd.DataFrame] = {"clean": df_clean.copy()}

    df_corrupted, _ = inject_missing(df_clean, TARGET_COLUMN, missing_config)
    df_corrupted, _ = inject_sensor_drift(
        df_corrupted, TARGET_COLUMN, drift_config,
    )
    variants["corrupted"] = df_corrupted

    imputer = DataImputer(
        mice_max_iter=mice_max_iter, mice_random_state=mice_random_state,
    )
    df_mice = imputer.impute_mice(df_corrupted, list(IMPUTATION_COLUMNS))
    variants["healed_mice"] = df_mice

    df_compensated = DriftCompensator().compensate_exponential_drift(
        df_mice, TARGET_COLUMN, window_hours=compensator_window_hours,
    )
    variants["healed_mice_plus_compensator"] = df_compensated

    logger.info(
        "Built %d variants: %s", len(variants), ", ".join(variants),
    )
    return variants


# ======================================================================
# Signal quality
# ======================================================================

def measure_signal_quality(
    variants: Dict[str, pd.DataFrame],
    *,
    reference: str = "clean",
    column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Compare each variant's sensor channel against the ground truth.

    Correlation is computed on rows where both series are present, so a
    variant is not penalised for the NaN it was given; the comparison is
    about the *shape* of the reconstructed signal.

    Args:
        variants: Output of :func:`build_variants`.
        reference: Variant treated as ground truth.
        column: Sensor channel to compare.

    Returns:
        One row per variant with correlation, MAE, RMSE and coverage.
    """
    truth = variants[reference][column]

    rows = []
    for name, frame in variants.items():
        series = frame[column]
        both = truth.notna() & series.notna()
        n_common = int(both.sum())

        if n_common < 2:
            correlation = mae = rmse = float("nan")
        else:
            correlation = float(truth[both].corr(series[both]))
            residual = (series[both] - truth[both]).to_numpy()
            mae = float(np.abs(residual).mean())
            rmse = float(np.sqrt((residual ** 2).mean()))

        rows.append({
            "variant": name,
            "correlation_with_truth": correlation,
            "mae": mae,
            "rmse": rmse,
            "n_missing": int(series.isna().sum()),
            "n_compared": n_common,
        })

    table = pd.DataFrame(rows)

    corrupted = table.loc[table["variant"] == "corrupted"]
    healed = table.loc[
        table["variant"] == "healed_mice_plus_compensator"
    ]
    if not corrupted.empty and not healed.empty:
        before = float(corrupted["correlation_with_truth"].iloc[0])
        after = float(healed["correlation_with_truth"].iloc[0])
        if after < before:
            logger.warning(
                "Healing REDUCED signal fidelity: correlation with ground "
                "truth fell from %.3f (corrupted) to %.3f (healed). This is "
                "the measured outcome and is reported unadjusted — see the "
                "module docstring. Do not tune the compensator to remove it.",
                before, after,
            )
    return table


# ======================================================================
# Forecasting performance
# ======================================================================

def evaluate_variants(
    variants: Dict[str, pd.DataFrame],
    *,
    model_name: str = "xgboost",
    seed: int = DEFAULT_SEED,
    validation_config: ValidationConfig | None = None,
    feature_config: FeatureConfig | None = None,
    restrict_to_common_rows: bool = False,
) -> pd.DataFrame:
    """Run the forecasting protocol on each variant.

    Corruption harms a forecast in two separable ways, and conflating
    them makes the results uninterpretable:

    1. **Lost observations.** NaN in the soil-moisture channel propagate
       through every lag and rolling window, so entire rows drop out of
       the design matrix. At a 20 % missingness rate the usable sample
       falls from 1 313 rows to 299 — a 77 % loss.

       The loss depends sharply on *how* the gaps are distributed, not
       just how many there are. A design row needs all 24 preceding
       hours, so under MCAR its survival probability is
       ``(1 − rate)^25`` — 0.4 % at a 20 % rate, which would leave the
       matrix empty. Heat-dependent missingness clusters into runs and
       destroys far fewer rows at the same nominal rate, which is why
       299 survive here. Any comparison of missingness rates must
       therefore hold the *mechanism* fixed.
    2. **Distorted values.** The readings that survive are shifted by the
       injected drift.

    ``restrict_to_common_rows=False`` measures both together, which is
    what a deployed system would experience. ``True`` restricts every
    variant to the hours all of them retain, isolating effect 2 — the
    only setting in which a difference between variants can be
    attributed to the *values* rather than to the sample size.

    Comparing 299 training rows against 1 313 and calling the difference
    a drift effect would be a sample-size artifact, so both views are
    reported.

    Args:
        variants: Output of :func:`build_variants`.
        model_name: Model to train on every variant.
        seed: Random seed.
        validation_config: Fold settings.
        feature_config: Feature settings.
        restrict_to_common_rows: Restrict every variant to shared hours.

    Returns:
        Long-format per-fold results with ``variant`` and ``n_rows``.
    """
    validation_config = validation_config or ValidationConfig()

    designs = {
        name: build_design_matrix(frame, feature_config)
        for name, frame in variants.items()
    }

    common: set | None = None
    if restrict_to_common_rows:
        for _X, _y, timestamps, _blocks in designs.values():
            stamps = set(pd.to_datetime(timestamps))
            common = stamps if common is None else (common & stamps)
        logger.info(
            "Restricting every variant to the %d hours all of them retain "
            "(from %s)",
            len(common or ()),
            ", ".join(f"{n}={len(d[0])}" for n, d in designs.items()),
        )

    frames = []
    for name, (X, y, timestamps, _blocks) in designs.items():
        if common is not None:
            mask = pd.to_datetime(timestamps).isin(common).to_numpy()
            X = X.loc[mask].reset_index(drop=True)
            y = y.loc[mask].reset_index(drop=True)

        splits = rolling_origin_splits(len(X), validation_config)

        logger.info(
            "── Variant '%s': %d rows × %d features, %d folds ──",
            name, len(X), X.shape[1], len(splits),
        )
        result = evaluate_on_splits(
            make_model_factory(model_name), X, y, splits,
            seed=seed, label=model_name,
        )
        result.insert(0, "n_rows", len(X))
        result.insert(0, "variant", name)
        frames.append(result)

    output = pd.concat(frames, ignore_index=True)
    output.insert(0, "comparison", (
        "common_rows" if restrict_to_common_rows else "as_available"
    ))
    return output


def summarise_variants(
    results: pd.DataFrame,
    metrics: Tuple[str, ...] = PRIMARY_METRICS,
) -> pd.DataFrame:
    """Aggregate per-fold variant results into mean ± SD with intervals."""
    rows = []
    for (comparison, variant, n_rows), group in results.groupby(
        ["comparison", "variant", "n_rows"], sort=False,
    ):
        row: Dict[str, object] = {
            "comparison": comparison,
            "variant": variant,
            "n_rows": int(n_rows),
        }
        for metric in metrics:
            stats_ = summarize_metric(
                group[metric].to_numpy(), label=f"{variant}/{metric}",
            )
            row[f"{metric}_mean"] = stats_["mean"]
            row[f"{metric}_std"] = stats_["std"]
            row[f"{metric}_ci_low"] = stats_["ci_low"]
            row[f"{metric}_ci_high"] = stats_["ci_high"]
        rows.append(row)
    return pd.DataFrame(rows)


# ======================================================================
# Reporting
# ======================================================================

def print_signal_quality(table: pd.DataFrame) -> None:
    """Print the signal-fidelity table."""
    print()
    print("## Signal fidelity of the soil-moisture channel")
    print()
    print("| Variant | corr. with truth | MAE | RMSE | NaN | n compared |")
    print("|:--------|-----------------:|----:|-----:|----:|-----------:|")
    for _, r in table.iterrows():
        print(
            f"| {r['variant']} | {r['correlation_with_truth']:.4f} "
            f"| {r['mae']:.4f} | {r['rmse']:.4f} "
            f"| {int(r['n_missing'])} | {int(r['n_compared'])} |"
        )
    print()
    print("_Each healing stage moves the reconstruction further from the "
          "ground truth. The compensator subtracts a rolling-minimum "
          "baseline, which removes genuine low-frequency variation in soil "
          "moisture along with the injected drift — and that low-frequency "
          "component is what predicts irrigation. Reported unadjusted._")
    print()


_COMPARISON_TITLES = {
    "as_available": (
        "as available — each variant on the rows it retains "
        "(lost observations + distorted values together)"
    ),
    "common_rows": (
        "common rows — every variant on the hours all of them retain "
        "(distorted values only)"
    ),
}


def print_variant_results(
    summary: pd.DataFrame, metric: str = "pr_auc",
) -> None:
    """Print forecasting performance per variant, with the caveats."""
    for comparison, block in summary.groupby("comparison", sort=False):
        print()
        print(f"## Forecasting performance — {metric.upper()}")
        print()
        print(f"**{_COMPARISON_TITLES.get(comparison, comparison)}**")
        print()
        print("| Variant | n rows | Mean | SD | 95% CI |")
        print("|:--------|-------:|-----:|---:|:-------|")
        for _, r in block.iterrows():
            print(
                f"| {r['variant']} | {int(r['n_rows'])} "
                f"| {r[f'{metric}_mean']:.4f} | {r[f'{metric}_std']:.4f} "
                f"| [{r[f'{metric}_ci_low']:.4f}, "
                f"{r[f'{metric}_ci_high']:.4f}] |"
            )
        print()

        if comparison == "as_available":
            sizes = block.set_index("variant")["n_rows"]
            if sizes.nunique() > 1:
                print(
                    f"_Sample sizes differ by construction here "
                    f"({sizes.min()}–{sizes.max()} rows): missingness "
                    f"propagates through the lag features and removes whole "
                    f"rows. Differences in this table therefore mix the cost "
                    f"of lost observations with the effect of distorted "
                    f"values, and cannot be attributed to drift alone. The "
                    f"common-rows table below separates them._"
                )
                print()
            continue

        # Say plainly when nothing here separates. Restricting to common
        # rows costs sample size — five folds over ~300 rows leave ~60
        # rows per test block — and the intervals widen accordingly.
        lo = block[f"{metric}_ci_low"].max()
        hi = block[f"{metric}_ci_high"].min()
        if lo <= hi:
            print(
                f"_**No variant is distinguishable from any other here.** "
                f"Every 95 % interval overlaps the range "
                f"[{lo:.4f}, {hi:.4f}]. Restricting to common rows costs "
                f"sample size — five folds over "
                f"{int(block['n_rows'].max())} rows leave roughly "
                f"{int(block['n_rows'].max()) // 6} rows per test block — so "
                f"this comparison can only rule out large effects. Read the "
                f"ordering below as a direction to investigate, not as a "
                f"result._"
            )
            print()

        clean = block.loc[block["variant"] == "clean", f"{metric}_mean"]
        corrupted = block.loc[
            block["variant"] == "corrupted", f"{metric}_mean"
        ]
        if not clean.empty and not corrupted.empty:
            if float(corrupted.iloc[0]) >= float(clean.iloc[0]):
                print(
                    "> **Corrupted data scored at or above clean data on "
                    "identical rows.** With sample size held fixed, this "
                    "cannot be a data-volume effect and points to the "
                    "injection model itself. The drift `a(1 − e^{−bt})` "
                    "resets every 35–40 h, forming a regular sawtooth that a "
                    "tree can read as a clock; since the irrigation schedule "
                    "has temporal structure, that phase information is "
                    "genuinely useful. It is an artifact, not a finding. A "
                    "study of drift robustness needs an aperiodic drift "
                    "model — a random walk, or resets drawn from a "
                    "heavy-tailed interval distribution."
                )
                print()


# ======================================================================
# Main
# ======================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Robustness study: sensor degradation and attempted recovery. "
            "Secondary to the main forecasting experiment."
        ),
    )
    parser.add_argument("--csv", default=PROCESSED_CSV)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--model", default="xgboost",
                        choices=("xgboost", "lightgbm"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--missing-rate", type=float, default=0.20,
        help="Fraction of soil-moisture readings removed (default: 0.20).",
    )
    parser.add_argument(
        "--mechanism", default="heat_dependent",
        choices=("mcar", "heat_dependent"),
    )
    parser.add_argument(
        "--metric", default="pr_auc", choices=PRIMARY_METRICS,
    )
    return parser.parse_args()


def main() -> None:
    """Run the robustness study and export its tables."""
    args = _parse_args()

    df_clean = load_modeling_frame(args.csv)

    variants = build_variants(
        df_clean,
        missing_config=MissingDataConfig(
            rate=args.missing_rate,
            mechanism=args.mechanism,
            seed=args.seed,
        ),
        drift_config=SensorDriftConfig(seed=args.seed),
    )

    signal_quality = measure_signal_quality(variants)
    print_signal_quality(signal_quality)

    # Both views: the deployed-system cost, and the isolated value effect.
    results = pd.concat(
        [
            evaluate_variants(
                variants, model_name=args.model, seed=args.seed,
                restrict_to_common_rows=restrict,
            )
            for restrict in (False, True)
        ],
        ignore_index=True,
    )
    summary = summarise_variants(results)
    print_variant_results(summary, metric=args.metric)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    signal_quality.to_csv(output_dir / ROBUSTNESS_SIGNAL_CSV, index=False)
    summary.to_csv(output_dir / ROBUSTNESS_FORECAST_CSV, index=False)

    print("## Exported")
    print()
    print(f"* `{output_dir / ROBUSTNESS_SIGNAL_CSV}`")
    print(f"* `{output_dir / ROBUSTNESS_FORECAST_CSV}`")
    print()

    logger.info("Robustness study complete. ✓")


if __name__ == "__main__":
    main()
