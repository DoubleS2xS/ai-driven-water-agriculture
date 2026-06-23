"""Phase 4 — End-to-end evaluation pipeline for the Q1 journal paper.

Runs the definitive comparative experiment on the real processed dataset
(``data/processed/merged_hourly.csv``) and produces:

1. A Markdown-formatted results table (Precision / Recall / F1 / ROC-AUC)
   for three scenarios: **Clean baseline**, **Corrupted**, and **Healed**.
2. Publication-ready SHAP plots (beeswarm summary + waterfall for a
   single edge-case decision) saved to ``data/outputs/``.

Usage
-----
::

    python -m src.evaluate_pipeline          # from project root
    python src/evaluate_pipeline.py          # direct invocation

Design notes
------------
* The train / test split is **chronological** (first 80 % → train, last
  20 % → test) to respect the temporal structure of the hourly sensor
  data and avoid look-ahead bias.
* Corruption parameters match the "harsh Karaganda climate degradation"
  profile: 20 % heat-dependent missingness + exponential sensor drift
  (a = 5.545, b = 0.08, recalibration cadence 35–40 h).
* All random seeds are fixed for reproducibility.

References
----------
* Review §2.3.1, sources [1–5]: empirical benchmarks for drift and
  imputation that ground the corruption / healing parameters.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import MissingDataConfig, SensorDriftConfig
from src.data_corruption import inject_missing, inject_sensor_drift
from src.data_healing import DataImputer, DriftCompensator, HealingEvaluator
from src.models.irrigation_ml import IrrigationPredictor
from src.models.explanation import SHAPExplainer

# ── Logging ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────

PROCESSED_CSV: str = "data/processed/merged_hourly.csv"
OUTPUT_DIR: str = "data/outputs"

FEATURE_COLS: List[str] = [
    "soil_moisture",
    "air_temp",
    "humidity",
    "wind_speed",
    "solar_radiation",
]
TARGET_COL: str = "irrigation_event"

TRAIN_FRACTION: float = 0.80


# ======================================================================
# Step 1 — Load and prepare the clean ground truth
# ======================================================================


def load_clean_dataset(csv_path: str) -> pd.DataFrame:
    """Read the merged hourly CSV and interpolate any native NaN.

    Args:
        csv_path: Path to ``merged_hourly.csv``.

    Returns:
        DataFrame with columns :data:`FEATURE_COLS` + :data:`TARGET_COL`,
        free of NaN values, ready to serve as clean ground truth.
    """
    logger.info("Loading processed dataset from %s", csv_path)
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    required = FEATURE_COLS + [TARGET_COL]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Available: {list(df.columns)}"
        )

    n_nan_before = int(df[required].isna().sum().sum())
    if n_nan_before > 0:
        logger.info(
            "Interpolating %d native NaN to create clean ground truth",
            n_nan_before,
        )
        for col in FEATURE_COLS:
            df[col] = df[col].interpolate(method="linear").ffill().bfill()
        # Target: fill with 0 (no irrigation) for any rare NaN
        df[TARGET_COL] = df[TARGET_COL].fillna(0).astype(int)

    logger.info(
        "Clean dataset ready: %d rows × %d cols, "
        "target distribution: %s",
        len(df),
        len(required),
        dict(df[TARGET_COL].value_counts().sort_index()),
    )
    return df


# ======================================================================
# Step 2 — Create experimental scenarios
# ======================================================================


def create_scenarios(
    df_clean: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create the three experimental DataFrames.

    Args:
        df_clean: NaN-free ground truth.

    Returns:
        Tuple of (df_clean, df_corrupted, df_healed).
    """
    # ── Scenario B: Corrupted ─────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("Scenario B — Applying corruption (20 %% heat-dependent "
                "missing + sensor drift)")

    missing_cfg = MissingDataConfig(
        rate=0.20,
        mechanism="heat_dependent",
        seed=42,
    )
    drift_cfg = SensorDriftConfig(
        a=5.545,
        b=0.08,
        recalibration_interval_hours=(35, 40),
        seed=42,
    )

    # Inject missing values into soil_moisture
    df_corrupted, _ = inject_missing(df_clean, "soil_moisture", missing_cfg)
    # Inject sensor drift on top of the missing-corrupted data
    df_corrupted, _ = inject_sensor_drift(
        df_corrupted, "soil_moisture", drift_cfg,
    )

    n_nan = int(df_corrupted["soil_moisture"].isna().sum())
    logger.info(
        "Corruption complete: %d NaN in soil_moisture, drift applied",
        n_nan,
    )

    # ── Scenario C: Healed ────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("Scenario C — Healing (MICE imputation + drift compensation)")

    imputer = DataImputer(mice_max_iter=10, mice_random_state=42)
    compensator = DriftCompensator()

    # Step 1: impute NaN using MICE (multivariate)
    df_healed = imputer.impute_mice(df_corrupted, FEATURE_COLS)
    # Step 2: compensate drift on soil_moisture
    df_healed = compensator.compensate_exponential_drift(
        df_healed, "soil_moisture", window_hours=24,
    )

    logger.info("Healing complete.")

    # ── Healing quality metrics (informational) ───────────────────────
    evaluator = HealingEvaluator()
    try:
        healing_metrics = evaluator.calculate_metrics(
            df_clean, df_corrupted, df_healed, "soil_moisture",
        )
        logger.info(
            "Healing quality on soil_moisture: MAE=%.4f, RMSE=%.4f, "
            "R²=%.4f (n=%d corrupted indices)",
            healing_metrics["mae"],
            healing_metrics["rmse"],
            healing_metrics["r2"],
            healing_metrics["n_corrupted"],
        )
    except ValueError as e:
        logger.warning("Could not compute healing metrics: %s", e)

    return df_clean, df_corrupted, df_healed


# ======================================================================
# Step 3 — Train and evaluate
# ======================================================================


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRACTION,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split data chronologically (no shuffle).

    Args:
        df: Full dataset with features and target.
        train_frac: Fraction of rows for training.

    Returns:
        (X_train, y_train, X_test, y_test).
    """
    split_idx = int(len(df) * train_frac)

    X_train = df[FEATURE_COLS].iloc[:split_idx].reset_index(drop=True)
    y_train = df[TARGET_COL].iloc[:split_idx].reset_index(drop=True)
    X_test = df[FEATURE_COLS].iloc[split_idx:].reset_index(drop=True)
    y_test = df[TARGET_COL].iloc[split_idx:].reset_index(drop=True)

    return X_train, y_train, X_test, y_test


def train_and_evaluate_scenario(
    label: str,
    df_train_source: pd.DataFrame,
    df_test_source: pd.DataFrame,
) -> Tuple[Dict[str, float], IrrigationPredictor]:
    """Train XGBoost on a scenario and return metrics + fitted model.

    The **test set is always the same** (from the clean ground-truth) to
    ensure a fair comparison.  Only the training data differs across
    scenarios.

    Args:
        label: Human-readable scenario name.
        df_train_source: Training DataFrame (varies per scenario).
        df_test_source: Test DataFrame (always clean ground truth).

    Returns:
        Tuple of (metrics_dict, fitted_predictor).
    """
    logger.info("─" * 50)
    logger.info("Training: %s", label)

    X_train = df_train_source[FEATURE_COLS]
    y_train = df_train_source[TARGET_COL]
    X_test = df_test_source[FEATURE_COLS]
    y_test = df_test_source[TARGET_COL]

    predictor = IrrigationPredictor(model_type="xgboost", random_state=42)
    predictor.train(X_train, y_train)
    metrics = predictor.evaluate(X_test, y_test)

    return metrics, predictor


# ======================================================================
# Step 4 — SHAP explanations
# ======================================================================


def generate_shap_plots(
    predictor: IrrigationPredictor,
    X_test: pd.DataFrame,
    output_dir: str,
) -> None:
    """Generate and save SHAP summary + waterfall plots.

    Args:
        predictor: Fitted IrrigationPredictor (Scenario C — Healed).
        X_test: Test feature matrix.
        output_dir: Directory for output PNGs.
    """
    logger.info("═" * 60)
    logger.info("Generating SHAP explanations (Healed model)")

    explainer = SHAPExplainer(predictor.model)

    summary_path = str(Path(output_dir) / "shap_summary.png")
    waterfall_path = str(Path(output_dir) / "shap_waterfall.png")

    explainer.plot_summary(X_test, save_path=summary_path, dpi=300)
    explainer.plot_local_decision(
        X_test, index=0, save_path=waterfall_path, dpi=300,
    )

    logger.info("SHAP plots saved to %s/", output_dir)


# ======================================================================
# Step 5 — Formatted results output
# ======================================================================


def print_results_table(
    results: Dict[str, Dict[str, float]],
) -> None:
    """Print a Markdown-formatted results table to stdout.

    Args:
        results: Mapping of scenario labels to metric dicts.
    """
    header = (
        "| Scenario | Precision | Recall | F1-Score | ROC-AUC |"
    )
    separator = (
        "|:---------|----------:|-------:|---------:|--------:|"
    )

    print()
    print("## Table — Comparative Classification Performance")
    print()
    print(header)
    print(separator)

    for scenario, metrics in results.items():
        row = (
            f"| {scenario:<25s} "
            f"| {metrics['precision']:>9.4f} "
            f"| {metrics['recall']:>6.4f} "
            f"| {metrics['f1']:>8.4f} "
            f"| {metrics['roc_auc']:>7.4f} |"
        )
        print(row)

    print()


# ======================================================================
# Main orchestrator
# ======================================================================


def main() -> None:
    """Run the full comparative evaluation pipeline."""
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║  Phase 4 — Comparative Evaluation Pipeline               ║")
    logger.info("╚" + "═" * 58 + "╝")

    # ── Step 1: Load clean data ───────────────────────────────────────
    df_clean = load_clean_dataset(PROCESSED_CSV)

    # ── Step 2: Create scenarios ──────────────────────────────────────
    df_clean, df_corrupted, df_healed = create_scenarios(df_clean)

    # ── Step 3: Chronological split ───────────────────────────────────
    split_idx = int(len(df_clean) * TRAIN_FRACTION)
    logger.info(
        "Chronological split: train=%d rows, test=%d rows (%.0f%% / %.0f%%)",
        split_idx,
        len(df_clean) - split_idx,
        TRAIN_FRACTION * 100,
        (1 - TRAIN_FRACTION) * 100,
    )

    # The test set is ALWAYS from the clean ground truth so that all
    # three models are evaluated on identical, unperturbed data.
    df_test_clean = df_clean.iloc[split_idx:].reset_index(drop=True)

    df_train_clean = df_clean.iloc[:split_idx].reset_index(drop=True)
    df_train_corrupted = df_corrupted.iloc[:split_idx].reset_index(drop=True)
    df_train_healed = df_healed.iloc[:split_idx].reset_index(drop=True)

    # ── Step 3b: Train and evaluate all scenarios ─────────────────────
    results: Dict[str, Dict[str, float]] = {}

    metrics_a, predictor_a = train_and_evaluate_scenario(
        "A — Clean Baseline", df_train_clean, df_test_clean,
    )
    results["A — Clean Baseline"] = metrics_a

    metrics_b, predictor_b = train_and_evaluate_scenario(
        "B — Corrupted (20% NaN + drift)", df_train_corrupted, df_test_clean,
    )
    results["B — Corrupted (20% NaN + drift)"] = metrics_b

    metrics_c, predictor_c = train_and_evaluate_scenario(
        "C — Healed (MICE + drift comp.)", df_train_healed, df_test_clean,
    )
    results["C — Healed (MICE + drift comp.)"] = metrics_c

    # ── Step 4: SHAP plots for Scenario C ─────────────────────────────
    X_test = df_test_clean[FEATURE_COLS]
    generate_shap_plots(predictor_c, X_test, OUTPUT_DIR)

    # ── Step 5: Print results table ───────────────────────────────────
    logger.info("═" * 60)
    logger.info("Final Results")
    print_results_table(results)

    # ── Summary deltas ────────────────────────────────────────────────
    f1_a = metrics_a["f1"]
    f1_b = metrics_b["f1"]
    f1_c = metrics_c["f1"]
    degradation = f1_a - f1_b
    recovery = f1_c - f1_b

    logger.info(
        "F1 degradation (Clean → Corrupted): %.4f → %.4f (Δ = −%.4f)",
        f1_a, f1_b, degradation,
    )
    logger.info(
        "F1 recovery    (Corrupted → Healed): %.4f → %.4f (Δ = +%.4f)",
        f1_b, f1_c, recovery,
    )
    if f1_a > 0:
        recovery_pct = (recovery / degradation * 100) if degradation > 0 else 0.0
        logger.info(
            "Healing recovered %.1f%% of the corruption-induced F1 loss.",
            recovery_pct,
        )

    logger.info("Pipeline complete. ✓")


if __name__ == "__main__":
    main()
