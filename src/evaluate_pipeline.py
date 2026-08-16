"""End-to-end evaluation: forecasting irrigation events from past data.

What this pipeline claims
-------------------------
Given everything observable up to and including hour ``t − 1``, predict
whether the electrovalve will be open during hour *t*.  That is a
forecast.  It is deliberately a weaker claim than the one the earlier
revision of this repository made.

What changed and why
--------------------
The previous version predicted ``irrigation_event(t)`` from
``soil_moisture(t)``.  Soil moisture rises *because* the valve opened, so
the model was reading the consequence of the event it claimed to
anticipate — detection after the fact dressed up as prediction.  Every
number produced under that setup is void, including the comparison
between the clean, corrupted and healed scenarios.

The corruption/healing experiment has moved out of this pipeline
entirely; it lives in :mod:`src.robustness_experiment`, where it is
reported as a robustness study rather than as the headline result.

Protocol
--------
* **Target** — ``irrigation_event(t)``, binary.
* **Features** — the causal set from :mod:`src.features`; every column
  is a function of ``t − 1`` or earlier.  ``soil_moisture(t)`` and the
  flow-meter channels are excluded and the exclusion is enforced, not
  merely intended.
* **Validation** — rolling-origin cross-validation with five
  expanding-window folds (primary), plus a single chronological 80/20
  split (secondary, for comparability).  Nothing is shuffled.
* **Preprocessing** — fitted inside each training fold and applied to
  that fold's test block, never on the full frame.

Usage
-----
::

    python -m src.evaluate_pipeline
    python -m src.evaluate_pipeline --model lightgbm --folds 5
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.baselines import (
    BASELINE_REGISTRY,
    available_baselines,
    make_baseline,
)
from src.config import FeatureConfig, ValidationConfig, build_loader_config
from src.export import export_all
from src.features import (
    BLOCK_CALENDAR,
    BLOCK_IRRIGATION,
    BLOCK_MOISTURE,
    BLOCK_ORDER,
    BLOCK_WEATHER,
    assert_no_forbidden_features,
    build_features,
    prepare_supervised,
)
from src.metrics import PRIMARY_METRICS, compute_classification_metrics
from src.statistics import (
    DEFAULT_N_BOOTSTRAP,
    aggregate_runs,
    bootstrap_auc_difference,
    format_p_value,
    summarize_metric,
)
from src.models.explanation import explain_model
from src.models.irrigation_ml import IrrigationPredictor
from src.validation import (
    assert_splits_are_ordered,
    chronological_holdout_split,
    describe_folds,
    rolling_origin_splits,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────

PROCESSED_CSV: str = "data/processed/merged_hourly.csv"
OUTPUT_DIR: str = "data/outputs"

DEFAULT_SEED: int = 42

#: Gradient-boosted models compared against the baselines.  LightGBM is
#: evaluated rather than merely mentioned: the earlier revision listed it
#: in the README and in the class docstring but never ran it, so no
#: reported number was ever attributable to it.
MAIN_MODELS: Tuple[str, ...] = ("xgboost", "lightgbm")


# ======================================================================
# Step 1 — Design matrix
# ======================================================================

def load_modeling_frame(csv_path: str = PROCESSED_CSV) -> pd.DataFrame:
    """Read the merged hourly dataset without imputing anything.

    Deliberately does **not** interpolate gaps or fill the target.

    The relay channel has 137 missing hours (8.5 %).  The earlier
    revision replaced them with ``0``, which manufactures negative
    examples out of "we don't know" and biases the positive rate
    downward.  Rows whose target is unknown are simply not supervised
    examples, and are dropped later by
    :func:`src.features.prepare_supervised`.

    Soil-moisture gaps (139 hours across 7 runs, the longest 62 h) are
    likewise left as NaN and propagate into the lag features, which
    causes the affected rows to be dropped.  Interpolating across a
    62-hour hole would invent the very dynamics the model is asked to
    learn.

    Args:
        csv_path: Path to ``merged_hourly.csv``.

    Returns:
        The raw merged frame, sorted by timestamp.
    """
    logger.info("Loading processed dataset from %s", csv_path)
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    n_missing_target = int(df["irrigation_event"].isna().sum())
    logger.info(
        "Loaded %d rows; %d (%.1f%%) have an unknown relay state and will "
        "not be used as supervised examples",
        len(df),
        n_missing_target,
        n_missing_target / len(df) * 100,
    )
    return df


def build_design_matrix(
    df: pd.DataFrame,
    feature_config: FeatureConfig | None = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, Dict[str, List[str]]]:
    """Build the causal feature matrix and the aligned target.

    Args:
        df: Merged hourly frame from :func:`load_modeling_frame`.
        feature_config: Feature definition; defaults apply if omitted.

    Returns:
        ``(X, y, timestamps, blocks)`` where *blocks* maps each feature
        block name to its column names, for the ablation study.
    """
    features, blocks = build_features(df, feature_config)
    all_names = [name for block in BLOCK_ORDER for name in blocks[block]]
    X, y, timestamps = prepare_supervised(features, all_names, feature_config)

    logger.info(
        "Design matrix: %d rows × %d features, %s → %s, positive rate %.4f",
        len(X),
        X.shape[1],
        timestamps.iloc[0],
        timestamps.iloc[-1],
        y.mean(),
    )
    return X, y, timestamps, blocks


# ======================================================================
# Step 2 — Fold evaluation
# ======================================================================

ModelFactory = Callable[[int], object]


def make_model_factory(
    model_type: str = "xgboost",
    **model_kwargs: object,
) -> ModelFactory:
    """Return a factory producing a fresh estimator for a given seed.

    A factory rather than an instance, so that every fold trains a model
    built from scratch.  Reusing a fitted estimator across folds would
    carry information from one fold's training data into the next fold's
    test block.

    Recognises both the gradient-boosted main models and the names in
    :data:`src.baselines.BASELINE_REGISTRY`, so a single call site covers
    every row of the results table.

    Args:
        model_type: ``"xgboost"``, ``"lightgbm"``, or a baseline name.
        **model_kwargs: Forwarded to the estimator constructor.

    Returns:
        Callable mapping a random seed to an unfitted estimator.
    """
    if model_type in BASELINE_REGISTRY:
        def baseline_factory(seed: int) -> object:
            return make_baseline(model_type, seed=seed)

        return baseline_factory

    def factory(seed: int) -> IrrigationPredictor:
        return IrrigationPredictor(
            model_type=model_type, random_state=seed, **model_kwargs,
        )

    return factory


def evaluate_on_splits(
    factory: ModelFactory,
    X: pd.DataFrame,
    y: pd.Series,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    seed: int = DEFAULT_SEED,
    label: str = "model",
) -> pd.DataFrame:
    """Train and score one estimator per fold.

    Each fold gets a freshly constructed estimator whose pipeline —
    preprocessing included — is fitted on that fold's training rows only.
    Baselines and main models take the identical path, so any difference
    between their scores comes from the models and not from the
    evaluation code.

    Args:
        factory: Produces an unfitted estimator for a seed.
        X: Causal feature matrix, time-ordered.
        y: Binary target aligned to *X*.
        splits: ``(train_idx, test_idx)`` pairs.
        seed: Random seed passed to the factory.
        label: Name recorded in the output table.

    Returns:
        One row per fold with the fold's metrics.
    """
    assert_splits_are_ordered(splits)
    # Last line of defence. The guard already runs at feature
    # construction and when composing ablation sets, but this is the
    # boundary that matters: nothing reaches an estimator without passing
    # here, including a matrix assembled by hand in a notebook.
    assert_no_forbidden_features(X.columns)

    rows: List[Dict[str, object]] = []
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        model = factory(seed)
        model.fit(X_train, y_train)

        y_pred = np.asarray(model.predict(X_test))
        y_proba = np.asarray(model.predict_proba(X_test))[:, 1]
        metrics = compute_classification_metrics(y_test, y_pred, y_proba)

        rows.append({
            "model": label,
            "seed": seed,
            "fold": fold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            **metrics,
        })

    return pd.DataFrame(rows)


def evaluate_all_models(
    X: pd.DataFrame,
    y: pd.Series,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    seed: int = DEFAULT_SEED,
    model_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Evaluate every baseline and main model on the same folds.

    Baselines requiring a feature absent from *X* are skipped with a
    warning rather than silently substituted — see
    :func:`src.baselines.available_baselines`.

    Args:
        X: Causal feature matrix.
        y: Binary target.
        splits: Fold definitions.
        seed: Random seed.
        model_names: Explicit model list; defaults to every usable
            baseline followed by the main models.

    Returns:
        Concatenated per-fold results for all models.
    """
    if model_names is None:
        usable = available_baselines(X.columns)
        skipped = set(BASELINE_REGISTRY) - set(usable)
        if skipped:
            logger.warning(
                "Baselines skipped on this feature set (required column "
                "absent): %s", sorted(skipped),
            )
        model_names = list(usable) + list(MAIN_MODELS)

    frames = []
    for name in model_names:
        logger.info("── Evaluating %s ──", name)
        frames.append(
            evaluate_on_splits(
                make_model_factory(name), X, y, splits,
                seed=seed, label=name,
            )
        )
    return pd.concat(frames, ignore_index=True)


def summarise(results: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    """Aggregate per-fold results into mean ± standard deviation.

    Args:
        results: Output of :func:`evaluate_on_splits`.
        metrics: Metric column names to aggregate.

    Returns:
        One row per model with ``<metric>_mean`` and ``<metric>_std``.
    """
    agg = (
        results.groupby("model")[list(metrics)]
        .agg(["mean", "std"])
    )
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    return agg.reset_index()


def evaluate_across_seeds(
    X: pd.DataFrame,
    y: pd.Series,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    seeds: Sequence[int],
    model_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Repeat the whole fold evaluation under each seed.

    Ten independent runs are what turns a point estimate into an
    estimate with a stated interval.  The deterministic baselines are
    still run for every seed so that the protocol is identical across
    models; their variance is structurally zero and is reported as such
    rather than hidden.

    Args:
        X: Causal feature matrix.
        y: Binary target.
        splits: Fold definitions, identical across seeds.
        seeds: Random seeds, ``0…9`` by protocol.
        model_names: Explicit model list, or ``None`` for all applicable.

    Returns:
        Long-format results, one row per (model, seed, fold).
    """
    frames = []
    for seed in seeds:
        logger.info("══ Seed %d ══", seed)
        frames.append(
            evaluate_all_models(
                X, y, splits, seed=seed, model_names=model_names,
            )
        )
    return pd.concat(frames, ignore_index=True)


def collect_out_of_fold_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    model_name: str,
    *,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Return each model's prediction for every row, from the fold that
    held that row out.

    Pooling out-of-fold predictions gives one prediction per row, none of
    them made by a model that saw the row during training, which is what
    the bootstrap comparison needs: a single aligned sample on which two
    models can be scored pairwise.

    Args:
        X: Causal feature matrix.
        y: Binary target.
        splits: Fold definitions.
        model_name: Model to run.
        seed: Random seed.

    Returns:
        Frame with ``row``, ``fold``, ``y_true`` and ``y_proba``, ordered
        by row index so that adjacent rows are adjacent hours — a
        prerequisite for the moving-block bootstrap.
    """
    assert_splits_are_ordered(splits)
    factory = make_model_factory(model_name)

    frames = []
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        model = factory(seed)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = np.asarray(model.predict_proba(X.iloc[test_idx]))[:, 1]
        frames.append(pd.DataFrame({
            "row": test_idx,
            "fold": fold,
            "y_true": y.iloc[test_idx].to_numpy(),
            "y_proba": proba,
        }))

    return pd.concat(frames, ignore_index=True).sort_values(
        "row"
    ).reset_index(drop=True)


def compare_best_model_to_best_baseline(
    X: pd.DataFrame,
    y: pd.Series,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    summary: pd.DataFrame,
    *,
    metric: str = "roc_auc",
    seed: int = DEFAULT_SEED,
    n_iterations: int = DEFAULT_N_BOOTSTRAP,
) -> Dict[str, object] | None:
    """Bootstrap-test the leading model against the leading baseline.

    "Best" is decided on the aggregated *metric*, so the comparison is
    not cherry-picked after seeing the test outcome.

    Args:
        X: Causal feature matrix.
        y: Binary target.
        splits: Fold definitions.
        summary: Output of :func:`src.statistics.aggregate_runs`.
        metric: Metric used both to rank and to compare.
        seed: Random seed for model fitting and resampling.
        n_iterations: Bootstrap replicates.

    Returns:
        The test result augmented with the two model names, or ``None``
        if either side has no candidate.
    """
    ranked = summary[summary["metric"] == metric].sort_values(
        "mean", ascending=False,
    )
    baseline_names = set(BASELINE_REGISTRY)

    main = ranked[~ranked["model"].isin(baseline_names)]
    base = ranked[ranked["model"].isin(baseline_names)]
    if main.empty or base.empty:
        logger.warning("Cannot compare: one side has no candidate model.")
        return None

    best_model = str(main.iloc[0]["model"])
    best_baseline = str(base.iloc[0]["model"])

    logger.info(
        "Comparing best main model '%s' (%s=%.4f) against best baseline "
        "'%s' (%s=%.4f)",
        best_model, metric, main.iloc[0]["mean"],
        best_baseline, metric, base.iloc[0]["mean"],
    )

    oof_model = collect_out_of_fold_predictions(
        X, y, splits, best_model, seed=seed,
    )
    oof_baseline = collect_out_of_fold_predictions(
        X, y, splits, best_baseline, seed=seed,
    )
    assert (oof_model["row"].to_numpy() == oof_baseline["row"].to_numpy()).all()

    result = bootstrap_auc_difference(
        oof_model["y_true"].to_numpy(),
        oof_model["y_proba"].to_numpy(),
        oof_baseline["y_proba"].to_numpy(),
        n_iterations=n_iterations,
        seed=seed,
        metric=metric,
    )
    # The i.i.d. variant is reported alongside so the effect of honouring
    # temporal dependence is visible rather than merely asserted.
    iid = bootstrap_auc_difference(
        oof_model["y_true"].to_numpy(),
        oof_model["y_proba"].to_numpy(),
        oof_baseline["y_proba"].to_numpy(),
        n_iterations=n_iterations,
        block_size=1,
        seed=seed,
        metric=metric,
    )

    result["model_a"] = best_model
    result["model_b"] = best_baseline
    result["p_value_iid"] = iid["p_value"]
    return result


# ======================================================================
# Step 3 — Reporting
# ======================================================================

def print_fold_table(table: pd.DataFrame) -> None:
    """Print the fold structure as a Markdown table."""
    print()
    print("## Rolling-origin fold structure")
    print()
    print("| Fold | test period (UTC) | n_train | n_test | train pos. | "
          "test pos. | test pos. rate | usable |")
    print("|-----:|:------------------|--------:|-------:|-----------:|"
          "----------:|---------------:|:-------|")
    for _, r in table.iterrows():
        period = (
            f"{r['test_start']:%Y-%m-%d} → {r['test_end']:%Y-%m-%d}"
            if "test_start" in table.columns else "—"
        )
        flag = "yes" if r["sufficient_positives"] else "**too few pos.**"
        print(
            f"| {int(r['fold'])} | {period} | {int(r['n_train'])} "
            f"| {int(r['n_test'])} | {int(r['train_positives'])} "
            f"| {int(r['test_positives'])} | {r['test_positive_rate']:.4f} "
            f"| {flag} |"
        )
    print()


def print_per_fold_results(
    results: pd.DataFrame,
    metrics: Sequence[str],
    metric: str = "pr_auc",
) -> None:
    """Print one metric per model per fold.

    Mandatory reporting, not a debugging aid: when the positive rate
    drifts across folds the aggregate hides which regimes each model
    actually handles.

    Args:
        results: Concatenated per-fold results for all models.
        metrics: Unused placeholder retained for call-site symmetry.
        metric: Which metric to break out by fold.
    """
    print()
    print(f"## Per-fold {metric.upper()} by model")
    print()
    pivot = results.pivot_table(
        index="model", columns="fold", values=metric, sort=False,
    )
    folds = list(pivot.columns)
    print("| Model | " + " | ".join(f"Fold {f}" for f in folds) + " |")
    print("|:------|" + "|".join(["------:"] * len(folds)) + "|")
    for model, row in pivot.iterrows():
        cells = " | ".join(
            f"{row[f]:.4f}" if pd.notna(row[f]) else "—" for f in folds
        )
        print(f"| {model} | {cells} |")
    print()


def print_results_table(
    summary: pd.DataFrame,
    metrics: Sequence[str],
    title: str,
) -> None:
    """Print aggregated results as a Markdown table."""
    print()
    print(f"## {title}")
    print()
    header = "| Model | " + " | ".join(m.upper() for m in metrics) + " |"
    sep = "|:------|" + "|".join(["------:"] * len(metrics)) + "|"
    print(header)
    print(sep)
    for _, r in summary.iterrows():
        cells = []
        for m in metrics:
            mean = r[f"{m}_mean"]
            std = r.get(f"{m}_std", float("nan"))
            cells.append(
                f"{mean:.4f} ± {std:.4f}" if pd.notna(std) else f"{mean:.4f}"
            )
        print(f"| {r['model']} | " + " | ".join(cells) + " |")
    print()


def print_summary_with_intervals(
    summary: pd.DataFrame,
    metrics: Sequence[str],
    title: str,
) -> None:
    """Print the aggregated table with both dispersion components.

    Every figure carries an interval.  The two are kept in separate
    columns because they measure different things: the seed interval
    reflects training stochasticity (zero for the deterministic
    baselines), the fold interval reflects variation across periods and
    is the one that speaks to generalisation.
    """
    print()
    print(f"## {title}")
    print()
    all_deterministic = bool(summary["deterministic_across_seeds"].all())
    if all_deterministic:
        n_seeds = int(summary["n_seeds"].max())
        print(
            f"> **Seed replication carries no information here.** All "
            f"{n_seeds} seeds produced byte-identical results for every "
            f"model: the tree ensembles run without row or column "
            f"subsampling, and the remaining models have no stochastic "
            f"component. The seed-level SD is therefore exactly 0 and is "
            f"omitted below. The interval reported is the **fold-level** "
            f"one, which measures the variation that actually exists — "
            f"performance across periods."
        )
        print()

    print("| Model | Metric | Mean | SD (fold) | 95% CI (fold) |")
    print("|:------|:-------|-----:|----------:|:--------------|")
    for metric in metrics:
        block = summary[summary["metric"] == metric].sort_values(
            "mean", ascending=(metric == "brier"),
        )
        for _, r in block.iterrows():
            ci = (
                f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
                if pd.notna(r["ci_low"]) else "—"
            )
            print(
                f"| {r['model']} | {metric} | {r['mean']:.4f} "
                f"| {r['std']:.4f} | {ci} |"
            )
    print()
    print("_Intervals are Student-t over the 5 rolling-origin folds, "
          "clipped to [0, 1]. They are wide because the irrigation regime "
          "is non-stationary, not because the estimates are noisy._")
    print()


def print_bootstrap_comparison(result: Dict[str, object] | None) -> None:
    """Print the model-versus-baseline bootstrap test."""
    print()
    print("## Best model vs. best baseline — paired bootstrap")
    print()
    if result is None:
        print("_Not available: one side had no candidate model._")
        print()
        return

    n_iter = int(result["n_iterations"])
    print(f"**{result['model_a']}** (a) versus **{result['model_b']}** (b), "
          f"on pooled out-of-fold predictions.")
    print()
    print("| Quantity | Value |")
    print("|:---------|------:|")
    print(f"| Metric | {result['metric']} |")
    print(f"| Observed difference (a − b) | {result['observed_diff']:+.4f} |")
    print(f"| 95% CI of the difference | "
          f"[{result['ci_low']:+.4f}, {result['ci_high']:+.4f}] |")
    print(f"| p-value (moving-block, {result['block_size']} h) | "
          f"{format_p_value(float(result['p_value']), n_iter)} |")
    print(f"| p-value (i.i.d. bootstrap, for contrast) | "
          f"{format_p_value(float(result['p_value_iid']), n_iter)} |")
    print(f"| Bootstrap iterations | "
          f"{result['n_valid']} valid of {n_iter} |")
    print()

    crosses_zero = float(result["ci_low"]) < 0.0 < float(result["ci_high"])
    if crosses_zero:
        print("The interval spans zero: the difference is **not** "
              "statistically distinguishable from no difference.")
    else:
        direction = (
            "outranks" if float(result["observed_diff"]) > 0 else
            "is outranked by"
        )
        print(f"The interval excludes zero: **{result['model_a']}** "
              f"{direction} **{result['model_b']}** on this sample.")
    print()
    print("_The i.i.d. p-value is shown only for contrast, and the "
          "moving-block value is the one to quote. Consecutive hours are "
          "strongly dependent (irrigation episodes run 2–117 h), so "
          "resampling individual hours treats each as fresh evidence and "
          "understates the variance. On this sample the effect is "
          "substantial: widening the block from 1 h to 72 h widens the "
          "interval of the difference from 0.019 to 0.045 and moves p from "
          "below 1e-3 to 0.024. Part of the dependence cancels because the "
          "comparison is paired, so the gap is smaller than it would be for "
          "either model's AUC on its own._")
    print()


# ======================================================================
# SHAP explanations
# ======================================================================

#: Models SHAP can explain here.  ``shap.TreeExplainer`` is exact for
#: gradient-boosted ensembles; the linear and rule-based baselines need a
#: different explainer and are out of scope for these figures.
EXPLAINABLE_MODELS: Tuple[str, ...] = MAIN_MODELS


def select_fold_for_explanation(
    results: pd.DataFrame,
    fold_table: pd.DataFrame,
    *,
    model_name: str,
    metric: str = "pr_auc",
    strategy: str = "best",
) -> int:
    """Choose which fold's held-out block to explain.

    Args:
        results: Per-fold results for the model.
        fold_table: Output of :func:`src.validation.describe_folds`.
        model_name: Model whose scores decide "best".
        metric: Metric used to rank folds.
        strategy: ``"best"`` — highest-scoring fold, as the protocol
            specifies.  ``"last"`` — the final fold, which has the most
            training history and the balance closest to deployment.

    Returns:
        The chosen fold number.

    Raises:
        ValueError: On an unknown strategy or an empty result set.
    """
    subset = results[results["model"] == model_name]
    if subset.empty:
        raise ValueError(f"No results for model '{model_name}'.")

    if strategy == "last":
        fold = int(subset["fold"].max())
    elif strategy == "best":
        fold = int(subset.loc[subset[metric].idxmax(), "fold"])
    else:
        raise ValueError(
            f"strategy must be 'best' or 'last', got '{strategy}'."
        )

    row = fold_table[fold_table["fold"] == fold]
    if not row.empty:
        rate = float(row.iloc[0]["test_positive_rate"])
        # An extreme balance makes the highest-scoring fold the least
        # instructive one to explain: near-perfect separation leaves few
        # errors, and a false-positive waterfall needs errors.
        if rate > 0.6 or rate < 0.1:
            logger.warning(
                "Fold %d was selected by '%s' but its test block is %.1f%% "
                "positive — an extreme regime. Its near-perfect separation "
                "leaves few errors to explain. Consider --shap-fold last "
                "for a block whose class balance resembles deployment.",
                fold, strategy, rate * 100,
            )
    return fold


def generate_shap_explanations(
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    fold: int,
    *,
    model_name: str,
    output_dir: str = OUTPUT_DIR,
    seed: int = DEFAULT_SEED,
) -> Dict[str, object]:
    """Fit the model on one fold and explain that fold's held-out block.

    The explanation is produced from a model that never saw the rows it
    explains, so the attributions describe generalisation behaviour
    rather than memorised training rows.

    Args:
        X: Causal feature matrix.
        y: Binary target.
        timestamps: UTC timestamps aligned to *X*.
        splits: Fold definitions.
        fold: 1-based fold number to explain.
        model_name: Tree model to explain.
        output_dir: Destination for figures and the JSON manifest.
        seed: Random seed.

    Returns:
        The SHAP manifest.

    Raises:
        ValueError: If the model is not tree-based or the fold is absent.
    """
    if model_name not in EXPLAINABLE_MODELS:
        raise ValueError(
            f"SHAP TreeExplainer needs a gradient-boosted model; "
            f"'{model_name}' is not one of {EXPLAINABLE_MODELS}."
        )
    if not 1 <= fold <= len(splits):
        raise ValueError(
            f"Fold {fold} out of range 1…{len(splits)}."
        )

    train_idx, test_idx = splits[fold - 1]

    predictor = make_model_factory(model_name)(seed)
    predictor.fit(X.iloc[train_idx], y.iloc[train_idx])

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    y_proba = predictor.predict_proba(X_test)[:, 1]

    logger.info(
        "Explaining %s on fold %d: %d held-out rows, %d positive",
        model_name, fold, len(X_test), int(y_test.sum()),
    )

    return explain_model(
        predictor.model,
        X_test.reset_index(drop=True),
        y_test.to_numpy(),
        y_proba,
        output_dir,
        timestamps=timestamps.iloc[test_idx].to_numpy(),
        row_indices=test_idx,
        metadata={
            "model": model_name,
            "fold": fold,
            "seed": seed,
            "fold_test_start": str(timestamps.iloc[test_idx[0]]),
            "fold_test_end": str(timestamps.iloc[test_idx[-1]]),
            "fold_test_positive_rate": float(y_test.mean()),
        },
    )


def print_shap_manifest(manifest: Dict[str, object]) -> None:
    """Print which hours the SHAP figures depict."""
    print()
    print("## SHAP explanations")
    print()
    print(f"Model **{manifest['model']}**, fold {manifest['fold']} "
          f"({manifest['fold_test_start']} → {manifest['fold_test_end']}), "
          f"{manifest['n_explained_rows']} held-out rows, "
          f"{float(manifest['fold_test_positive_rate']):.1%} positive.")
    print()
    print("| Case | Row | Timestamp (UTC) | y_true | p(irrigate) |")
    print("|:-----|----:|:----------------|-------:|------------:|")
    for case, inst in manifest["instances"].items():
        if inst is None:
            print(f"| {case} | — | _no candidate in this fold_ | — | — |")
            continue
        print(
            f"| {case} | {inst['row_index']} | {inst['timestamp']} "
            f"| {inst['y_true']} | {inst['y_proba']:.4f} |"
        )
    print()
    print(f"**Top features by mean |SHAP|:** "
          f"{', '.join(manifest['top_features'])}")
    print()


# ======================================================================
# Feature ablation
# ======================================================================

#: Which feature blocks make up each ablation set.
#:
#: Set A is the whole soil-moisture block — lags, drying-rate
#: differences and causal rolling statistics — since all of them are
#: functions of past soil moisture alone.
#:
#: Set E is the control.  It excludes every channel that could carry the
#: target's own history, so a high score there would mean information is
#: reaching the model by a route the design does not intend.
ABLATION_SETS: Dict[str, Tuple[str, ...]] = {
    "A": (BLOCK_MOISTURE,),
    "B": (BLOCK_MOISTURE, BLOCK_WEATHER),
    "C": (BLOCK_MOISTURE, BLOCK_WEATHER, BLOCK_CALENDAR),
    "D": (BLOCK_MOISTURE, BLOCK_WEATHER, BLOCK_CALENDAR, BLOCK_IRRIGATION),
    "E": (BLOCK_WEATHER,),
}

ABLATION_DESCRIPTIONS: Dict[str, str] = {
    "A": "soil-moisture lags only",
    "B": "+ weather",
    "C": "+ calendar",
    "D": "+ irrigation autoregressive lags (full set)",
    "E": "weather only (leakage control)",
}

#: Models used for the ablation.  The protocol calls for one model; a
#: second is included because set E is a *control*, and a control that
#: rests on a single estimator is only as trustworthy as that estimator.
#: XGBoost collapses on the small early folds (see the per-fold table),
#: so a low set-E score from XGBoost alone could reflect a weak learner
#: rather than an absence of leakage.  Logistic regression, the strongest
#: model on this data, provides the corroborating read.
ABLATION_MODELS: Tuple[str, ...] = ("xgboost", "logistic")


def build_ablation_feature_sets(
    blocks: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """Expand :data:`ABLATION_SETS` into concrete column names.

    Args:
        blocks: Block-to-columns mapping from :func:`build_design_matrix`.

    Returns:
        Ablation-set name to ordered feature-column names.
    """
    return {
        name: [col for block in block_names for col in blocks[block]]
        for name, block_names in ABLATION_SETS.items()
    }


def run_ablation(
    X: pd.DataFrame,
    y: pd.Series,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    feature_sets: Dict[str, List[str]],
    *,
    model_names: Sequence[str] = ABLATION_MODELS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Evaluate one model per ablation set, on identical rows.

    Every set is a **column subset of the same matrix** *X*, never a
    fresh call to :func:`src.features.prepare_supervised`.  This matters:
    rebuilding per set would drop a different warm-up period for each
    one — set A survives 24 h of history while set D needs all of it —
    and the sets would then differ in sample size as well as in
    features.  Any gap between them would confound the two, which is
    precisely the comparison the ablation exists to isolate.

    Args:
        X: Full causal feature matrix, rows already filtered.
        y: Binary target aligned to *X*.
        splits: Fold definitions, shared by every set.
        feature_sets: Output of :func:`build_ablation_feature_sets`.
        model_names: Models to run on each set.
        seed: Random seed.  A single seed suffices because every
            estimator here is deterministic; see
            :func:`src.statistics.aggregate_runs`.

    Returns:
        Long-format results with a ``feature_set`` column added.

    Raises:
        ValueError: If a set references a column absent from *X*.
    """
    frames = []
    for set_name, columns in feature_sets.items():
        missing = [c for c in columns if c not in X.columns]
        if missing:
            raise ValueError(
                f"Ablation set {set_name} references columns absent from the "
                f"design matrix: {missing}."
            )
        # Guard the guard: a flow-meter column reaching an ablation set
        # would make its score meaningless in a way that looks like a
        # discovery rather than a bug.
        assert_no_forbidden_features(columns)

        X_subset = X[columns]
        for model_name in model_names:
            logger.info(
                "── Ablation %s (%s), %d features, model=%s ──",
                set_name, ABLATION_DESCRIPTIONS[set_name],
                len(columns), model_name,
            )
            result = evaluate_on_splits(
                make_model_factory(model_name), X_subset, y, splits,
                seed=seed, label=model_name,
            )
            result.insert(0, "n_features", len(columns))
            result.insert(0, "feature_set", set_name)
            frames.append(result)

    return pd.concat(frames, ignore_index=True)


def summarise_ablation(
    results: pd.DataFrame,
    metrics: Sequence[str] = PRIMARY_METRICS,
) -> pd.DataFrame:
    """Aggregate ablation folds into mean ± SD per (set, model, metric)."""
    rows = []
    grouped = results.groupby(
        ["feature_set", "model", "n_features"], sort=False,
    )
    for (set_name, model, n_features), group in grouped:
        row: Dict[str, object] = {
            "feature_set": set_name,
            "description": ABLATION_DESCRIPTIONS[set_name],
            "model": model,
            "n_features": int(n_features),
            "n_rows": int(group["n_test"].sum()),
        }
        for metric in metrics:
            stats_ = summarize_metric(
                group[metric].to_numpy(), label=f"{set_name}/{model}/{metric}",
            )
            row[f"{metric}_mean"] = stats_["mean"]
            row[f"{metric}_std"] = stats_["std"]
            row[f"{metric}_ci_low"] = stats_["ci_low"]
            row[f"{metric}_ci_high"] = stats_["ci_high"]
        rows.append(row)
    return pd.DataFrame(rows)


def print_ablation_table(
    summary: pd.DataFrame, y: pd.Series, metric: str = "pr_auc",
) -> None:
    """Print the ablation table and interpret the set-E control."""
    print()
    print(f"## Feature ablation — {metric.upper()}")
    print()
    print("| Set | Features | n | Model | Mean | SD | 95% CI |")
    print("|:----|:---------|--:|:------|-----:|---:|:-------|")
    for _, r in summary.sort_values(["feature_set", "model"]).iterrows():
        print(
            f"| {r['feature_set']} | {r['description']} | {r['n_features']} "
            f"| {r['model']} | {r[f'{metric}_mean']:.4f} "
            f"| {r[f'{metric}_std']:.4f} "
            f"| [{r[f'{metric}_ci_low']:.4f}, {r[f'{metric}_ci_high']:.4f}] |"
        )
    print()

    prevalence = float(y.mean())
    no_skill = prevalence if metric == "pr_auc" else 0.5
    n_rows = int(summary["n_rows"].max())
    print(
        f"_All sets are scored on the identical {n_rows} held-out rows, so "
        f"differences reflect features alone. No-skill {metric.upper()} = "
        f"{no_skill:.4f}._"
    )
    print()


def interpret_leakage_control(
    summary: pd.DataFrame, y: pd.Series, metric: str = "pr_auc",
) -> None:
    """Report what set E implies about leakage.

    Set E is the diagnostic that would expose target information
    arriving through the weather channel.  Its reading needs care in both
    directions, so the thresholds and the caveat are stated rather than
    left to the reader.
    """
    print()
    print("## Set E — leakage control")
    print()
    e_rows = summary[summary["feature_set"] == "E"]
    if e_rows.empty:
        print("_Set E was not evaluated._")
        return

    prevalence = float(y.mean())
    no_skill = prevalence if metric == "pr_auc" else 0.5

    for _, r in e_rows.iterrows():
        value = r[f"{metric}_mean"]
        print(f"* **{r['model']}** — {metric.upper()} = {value:.4f} "
              f"(no-skill {no_skill:.4f}, full set D = "
              f"{summary[(summary['feature_set'] == 'D') & (summary['model'] == r['model'])][f'{metric}_mean'].iloc[0]:.4f})")

    worst = e_rows[f"{metric}_mean"].max()
    print()
    if worst > 0.90:
        print("**Leakage indicated.** Weather alone should not come close to "
              "the full model. A score this high means target information is "
              "reaching the model through the meteorological channel — "
              "investigate before reporting any other result.")
    else:
        print(
            "**No leakage indicated.** Weather alone stays far below the "
            "full feature set.\n\n"
            "The score is nevertheless above no-skill, and that is expected "
            "rather than suspicious: the record runs from July to September, "
            "over which both temperature and irrigation frequency rise "
            "together. Weather therefore proxies *season*, and season "
            "correlates with the irrigation regime. This is confounding by a "
            "shared trend, not leakage of the target — the distinction being "
            "that no weather feature is computed from the valve state. It "
            "does mean set E's margin over no-skill should not be read as "
            "evidence that weather drives irrigation decisions."
        )
    print()


def print_exported_artifacts(paths: Dict[str, Path]) -> None:
    """List the files written, so the manuscript can cite them by name."""
    print()
    print("## Exported artefacts")
    print()
    print("Every figure quoted in the manuscript is generated by this run "
          "and written below. None is transcribed by hand.")
    print()
    print("| Artefact | File |")
    print("|:---------|:-----|")
    for name, path in paths.items():
        print(f"| {name} | `{path}` |")
    print()


def print_reference_points(y: pd.Series) -> None:
    """Print the values a useless classifier would score.

    PR-AUC in particular is routinely misread against 0.5.  Its no-skill
    value is the positive rate, so stating both next to the results table
    removes the most common way of overstating a result.
    """
    prevalence = float(y.mean())
    print("**Reference points for a no-skill classifier** — "
          f"ROC-AUC = 0.5000, PR-AUC = {prevalence:.4f} (the positive "
          f"rate), Brier = {prevalence * (1 - prevalence):.4f}. "
          "A PR-AUC must be compared against the positive rate, never "
          "against 0.5.")
    print()


# ======================================================================
# Main
# ======================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Forecast irrigation events from strictly causal features "
            "using rolling-origin cross-validation."
        ),
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help=(
            "Models to evaluate. Defaults to every applicable baseline "
            "followed by the main models."
        ),
    )
    parser.add_argument(
        "--folds", type=int, default=ValidationConfig().n_folds,
        help="Number of rolling-origin folds (default: 5).",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Seed for the bootstrap and holdout run (default: 42).",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=len(ValidationConfig().random_seeds),
        help="Number of repeated runs, seeds 0…n-1 (default: 10).",
    )
    parser.add_argument(
        "--bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP,
        help="Bootstrap iterations for the model comparison (default: 10000).",
    )
    parser.add_argument(
        "--compare-metric", default="roc_auc", choices=("roc_auc", "pr_auc"),
        help="Metric used to rank models and to compare them (default: roc_auc).",
    )
    parser.add_argument(
        "--shap-model", default="xgboost", choices=EXPLAINABLE_MODELS,
        help="Tree model to explain with SHAP (default: xgboost).",
    )
    parser.add_argument(
        "--shap-fold", default="best", choices=("best", "last"),
        help=(
            "Which fold's held-out block to explain: 'best' by the "
            "comparison metric (default), or 'last' for the block whose "
            "class balance is closest to deployment."
        ),
    )
    parser.add_argument(
        "--csv", default=PROCESSED_CSV,
        help="Path to the merged hourly dataset.",
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help=(
            "Loader YAML, recorded in run_metadata.json so the site and "
            "timezone assumptions travel with the results."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the forecasting evaluation."""
    args = _parse_args()

    validation_config = ValidationConfig(n_folds=args.folds)

    # ── Design matrix ─────────────────────────────────────────────────
    df = load_modeling_frame(args.csv)
    X, y, timestamps, _blocks = build_design_matrix(df)

    metrics = PRIMARY_METRICS
    seeds = tuple(range(args.n_seeds))

    # ── Primary: rolling-origin CV, repeated across seeds ─────────────
    splits = rolling_origin_splits(len(X), validation_config)
    fold_table = describe_folds(splits, y, timestamps, validation_config)
    print_fold_table(fold_table)

    cv_results = evaluate_across_seeds(
        X, y, splits, seeds=seeds, model_names=args.models,
    )
    summary = aggregate_runs(cv_results, metrics)

    print_per_fold_results(
        cv_results[cv_results["seed"] == seeds[0]], metrics, metric="pr_auc",
    )
    print_summary_with_intervals(
        summary,
        metrics,
        f"Rolling-origin CV ({args.folds} folds × {len(seeds)} seeds)",
    )
    print_reference_points(y)

    # ── Feature ablation ──────────────────────────────────────────────
    feature_sets = build_ablation_feature_sets(_blocks)
    ablation_results = run_ablation(
        X, y, splits, feature_sets, seed=args.seed,
    )
    ablation_summary = summarise_ablation(ablation_results)
    print_ablation_table(ablation_summary, y, metric=args.compare_metric)
    interpret_leakage_control(ablation_summary, y, metric=args.compare_metric)

    # ── Model versus baseline ─────────────────────────────────────────
    comparison = compare_best_model_to_best_baseline(
        X, y, splits, summary,
        metric=args.compare_metric,
        seed=args.seed,
        n_iterations=args.bootstrap,
    )
    print_bootstrap_comparison(comparison)

    # ── SHAP explanations ─────────────────────────────────────────────
    shap_model = args.shap_model
    first_seed_results = cv_results[cv_results["seed"] == seeds[0]]
    shap_fold = select_fold_for_explanation(
        first_seed_results, fold_table,
        model_name=shap_model,
        metric=args.compare_metric,
        strategy=args.shap_fold,
    )
    manifest = generate_shap_explanations(
        X, y, timestamps, splits, shap_fold,
        model_name=shap_model, output_dir=OUTPUT_DIR, seed=args.seed,
    )
    print_shap_manifest(manifest)

    # ── Secondary: single chronological 80/20 split ────────────────────
    holdout = [chronological_holdout_split(len(X), validation_config)]
    holdout_results = evaluate_all_models(
        X, y, holdout, seed=args.seed, model_names=args.models,
    )
    print_results_table(
        holdout_results.assign(
            **{f"{m}_mean": holdout_results[m] for m in metrics}
        ),
        metrics,
        "Chronological 80/20 holdout (single split, secondary result)",
    )

    # ── Export every number to a file ─────────────────────────────────
    paths = export_all(
        OUTPUT_DIR,
        summary=summary,
        ablation_summary=ablation_summary,
        fold_table=fold_table,
        shap_manifest=manifest,
        comparison=comparison,
        blocks=_blocks,
        baseline_names=list(BASELINE_REGISTRY),
        main_model_names=list(MAIN_MODELS),
        n_rows=len(X),
        n_features=X.shape[1],
        positive_rate=float(y.mean()),
        seeds=seeds,
        n_folds=args.folds,
        dataset_start=str(timestamps.iloc[0]),
        dataset_end=str(timestamps.iloc[-1]),
        loader_config=build_loader_config(args.config),
    )
    print_exported_artifacts(paths)

    logger.info("Evaluation complete. ✓")


if __name__ == "__main__":
    main()
