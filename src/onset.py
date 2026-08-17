"""Forecasting the *start* of an irrigation episode.

Why this is a separate protocol
-------------------------------
The main experiment predicts whether the valve is open during hour *t*.
Most of that signal is continuation: irrigation runs in episodes, and
``irrigation_event(t-1)`` alone correlates with the target at r = 0.81.
A model can score well on it while having learned only "what was
happening a moment ago is probably still happening".

The decision an irrigation controller actually makes is *when to start*.
Once the valve is open, the following hours are consequences of a choice
already taken, not new choices. This protocol therefore targets

``onset(t) = 1`` iff the valve is open at *t* and was closed at *t − 1*,

which is a strictly harder and operationally more meaningful problem.

Conditioning on a closed valve
------------------------------
Evaluation is restricted to hours where the valve **was closed** at
*t − 1*. An onset is impossible otherwise, so keeping those rows would
pad the negative class with examples any model classifies correctly for
free — inflating every metric while telling nobody anything. Restricting
makes the comparison a genuine hazard question: *given that irrigation is
not currently running, will it start this hour?*

Two consequences follow, and both are handled explicitly rather than left
to surprise the reader:

* ``irrigation_event_lag1h`` becomes constant zero by construction and is
  dropped. Keeping a zero-variance column would clutter the SHAP ranking
  with a feature that cannot contribute.
* The **persistence baseline degenerates**. It predicts
  ``irrigation_event(t-1)``, which is always 0 here, making it identical
  to the majority baseline. It is excluded rather than reported twice
  under two names.

Expect weak numbers
-------------------
Roughly 4 % of eligible hours are onsets, against 23.6 % for the
continuation task. Folds hold a handful of positive examples each, so
intervals are wide and some folds may contain no onset at all — in which
case AUC is undefined and reported as NaN rather than zero. A weak result
here is informative: it says the *timing* of irrigation decisions is not
recoverable from soil-moisture history alone, which is a more useful
finding for practitioners than a strong score on the continuation task.

Usage
-----
Invoked from :mod:`src.evaluate_pipeline`; results land in
``data/outputs/onset_results.csv``.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.baselines import BASELINE_REGISTRY
from src.config import FeatureConfig, ValidationConfig
from src.features import (
    BLOCK_ORDER,
    build_features,
    irrigation_onset,
    prepare_supervised,
)
from src.metrics import PRIMARY_METRICS
from src.statistics import summarize_metric
from src.validation import describe_folds, rolling_origin_splits

logger = logging.getLogger(__name__)

ONSET_TARGET_COLUMN: str = "irrigation_onset"

#: Feature that becomes constant once evaluation is restricted to hours
#: with the valve closed.
CONSTANT_UNDER_RESTRICTION: str = "irrigation_event_lag1h"

#: Baselines that remain meaningful for onset.  Persistence is excluded:
#: with the valve closed at t-1 by construction it always predicts 0,
#: which is the majority baseline under a different name.
ONSET_BASELINES: Tuple[str, ...] = (
    "majority", "moisture_threshold", "logistic",
)

ONSET_RESULTS_CSV: str = "onset_results.csv"


# ======================================================================
# Design matrix
# ======================================================================

def build_onset_design_matrix(
    df: pd.DataFrame,
    feature_config: FeatureConfig | None = None,
    *,
    restrict_to_closed_valve: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, Dict[str, List[str]]]:
    """Build the causal feature matrix against the onset target.

    Args:
        df: Merged hourly frame.
        feature_config: Feature definition; defaults apply if omitted.
        restrict_to_closed_valve: Keep only hours where the valve was
            closed at *t − 1*.  Leaving this on is what makes the metrics
            interpretable; see the module docstring.

    Returns:
        ``(X, y, timestamps, blocks)`` with *y* the onset indicator.

    Raises:
        ValueError: If no eligible rows remain.
    """
    cfg = feature_config or FeatureConfig()

    features, blocks = build_features(df, cfg)
    features[ONSET_TARGET_COLUMN] = irrigation_onset(
        df[cfg.target_col], cfg,
    ).to_numpy()

    onset_cfg = replace(cfg, target_col=ONSET_TARGET_COLUMN)
    names = [name for block in BLOCK_ORDER for name in blocks[block]]
    X, y, timestamps = prepare_supervised(features, names, onset_cfg)

    if restrict_to_closed_valve:
        if CONSTANT_UNDER_RESTRICTION not in X.columns:
            raise ValueError(
                f"Cannot restrict to a closed valve without "
                f"'{CONSTANT_UNDER_RESTRICTION}' in the feature set."
            )
        eligible = X[CONSTANT_UNDER_RESTRICTION].to_numpy() == 0
        logger.info(
            "Restricting to hours with the valve closed at t-1: "
            "%d of %d rows eligible (%d onsets retained of %d)",
            int(eligible.sum()), len(X),
            int(y[eligible].sum()), int(y.sum()),
        )
        X = X.loc[eligible].reset_index(drop=True)
        y = y.loc[eligible].reset_index(drop=True)
        timestamps = timestamps.loc[eligible].reset_index(drop=True)

        # Constant by construction now: drop rather than feed a
        # zero-variance column to the models and the SHAP ranking.
        X = X.drop(columns=[CONSTANT_UNDER_RESTRICTION])
        blocks = {
            block: [n for n in names_ if n != CONSTANT_UNDER_RESTRICTION]
            for block, names_ in blocks.items()
        }

    if len(X) == 0:
        raise ValueError("No eligible rows remain for the onset protocol.")

    logger.info(
        "Onset design matrix: %d rows × %d features, %s → %s, "
        "onset rate %.4f (%d positives)",
        len(X), X.shape[1], timestamps.iloc[0], timestamps.iloc[-1],
        y.mean(), int(y.sum()),
    )
    return X, y, timestamps, blocks


# ======================================================================
# Evaluation
# ======================================================================

def evaluate_onset(
    X: pd.DataFrame,
    y: pd.Series,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    model_names: Optional[Sequence[str]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate models on the onset target over the given folds.

    Args:
        X: Causal feature matrix, restricted to eligible hours.
        y: Onset indicator.
        splits: Fold definitions.
        model_names: Models to run; defaults to
            :data:`ONSET_BASELINES` plus the gradient-boosted models.
        seed: Random seed.

    Returns:
        Long-format per-fold results.
    """
    # Imported here to avoid a circular import: evaluate_pipeline imports
    # this module to run the protocol.
    from src.evaluate_pipeline import (
        MAIN_MODELS,
        evaluate_on_splits,
        make_model_factory,
    )

    if model_names is None:
        model_names = list(ONSET_BASELINES) + list(MAIN_MODELS)

    excluded = set(BASELINE_REGISTRY) - set(ONSET_BASELINES)
    if excluded:
        logger.info(
            "Baselines excluded from the onset protocol: %s — with the "
            "valve closed at t-1 by construction, persistence always "
            "predicts 0 and duplicates the majority baseline.",
            sorted(excluded),
        )

    frames = []
    for name in model_names:
        logger.info("── Onset: %s ──", name)
        frames.append(
            evaluate_on_splits(
                make_model_factory(name), X, y, splits,
                seed=seed, label=name,
            )
        )
    return pd.concat(frames, ignore_index=True)


def summarise_onset(
    results: pd.DataFrame,
    metrics: Sequence[str] = PRIMARY_METRICS,
) -> pd.DataFrame:
    """Aggregate onset folds into mean ± SD with intervals.

    Folds where the metric is undefined — a test block containing no
    onset at all — are dropped from the aggregate by
    :func:`src.statistics.summarize_metric`, and the surviving count is
    reported in ``n_folds_scored`` so the reader can see how thin the
    evidence is.
    """
    rows = []
    for model, group in results.groupby("model", sort=False):
        row: Dict[str, object] = {
            "model": model,
            "n_rows": int(group["n_test"].sum()),
            "n_folds": int(group["fold"].nunique()),
            "n_onsets": int(group["n_positive"].sum()),
        }
        for metric in metrics:
            stats_ = summarize_metric(
                group[metric].to_numpy(), label=f"onset/{model}/{metric}",
            )
            row[f"{metric}_mean"] = stats_["mean"]
            row[f"{metric}_std"] = stats_["std"]
            row[f"{metric}_ci_low"] = stats_["ci_low"]
            row[f"{metric}_ci_high"] = stats_["ci_high"]
            row[f"{metric}_n_folds_scored"] = stats_["n"]
        rows.append(row)
    return pd.DataFrame(rows)


# ======================================================================
# Reporting and export
# ======================================================================

def print_onset_results(
    summary: pd.DataFrame,
    fold_table: pd.DataFrame,
    y: pd.Series,
    metric: str = "pr_auc",
) -> None:
    """Print the onset fold structure and results."""
    print()
    print("## Onset protocol — fold structure")
    print()
    print("| Fold | n_train | n_test | onsets in train | onsets in test |")
    print("|-----:|--------:|-------:|----------------:|---------------:|")
    for _, r in fold_table.iterrows():
        print(
            f"| {int(r['fold'])} | {int(r['n_train'])} | {int(r['n_test'])} "
            f"| {int(r['train_positives'])} | {int(r['test_positives'])} |"
        )
    print()

    no_skill = float(y.mean()) if metric == "pr_auc" else 0.5
    print(f"## Onset protocol — {metric.upper()}")
    print()
    print("| Model | Mean | SD | 95% CI | folds scored |")
    print("|:------|-----:|---:|:-------|-------------:|")
    for _, r in summary.sort_values(f"{metric}_mean", ascending=False).iterrows():
        mean, std = r[f"{metric}_mean"], r[f"{metric}_std"]
        ci = (
            f"[{r[f'{metric}_ci_low']:.4f}, {r[f'{metric}_ci_high']:.4f}]"
            if pd.notna(r[f"{metric}_ci_low"]) else "—"
        )
        mean_text = f"{mean:.4f}" if pd.notna(mean) else "—"
        std_text = f"{std:.4f}" if pd.notna(std) else "—"
        print(
            f"| {r['model']} | {mean_text} | {std_text} | {ci} "
            f"| {int(r[f'{metric}_n_folds_scored'])} of {int(r['n_folds'])} |"
        )
    print()
    print(f"_Onset rate {float(y.mean()):.4f}, so the no-skill "
          f"{metric.upper()} is {no_skill:.4f}. Evaluation covers only hours "
          f"with the valve closed at t-1, where an onset is possible at all. "
          f"The persistence baseline is excluded: it always predicts 0 here "
          f"and duplicates the majority baseline._")
    print()


def write_onset_results(
    summary: pd.DataFrame,
    output_dir: str | Path,
    *,
    filename: str = ONSET_RESULTS_CSV,
) -> Path:
    """Write the onset results table.

    Args:
        summary: Output of :func:`summarise_onset`.
        output_dir: Destination directory.
        filename: Output file name.

    Returns:
        The path written.
    """
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    logger.info("Wrote %s (%d rows)", path, len(summary))
    return path


def run_onset_protocol(
    df: pd.DataFrame,
    *,
    output_dir: str | Path,
    feature_config: FeatureConfig | None = None,
    validation_config: ValidationConfig | None = None,
    seed: int = 42,
    metric: str = "pr_auc",
) -> Tuple[pd.DataFrame, Path]:
    """Run the whole onset protocol and export its table.

    Args:
        df: Merged hourly frame.
        output_dir: Destination for ``onset_results.csv``.
        feature_config: Feature definition.
        validation_config: Fold settings.
        seed: Random seed.
        metric: Metric highlighted in the printed report.

    Returns:
        ``(summary, path)``.
    """
    validation_config = validation_config or ValidationConfig()

    X, y, timestamps, _blocks = build_onset_design_matrix(df, feature_config)
    splits = rolling_origin_splits(len(X), validation_config)
    fold_table = describe_folds(splits, y, timestamps, validation_config)

    results = evaluate_onset(X, y, splits, seed=seed)
    summary = summarise_onset(results)

    print_onset_results(summary, fold_table, y, metric=metric)
    path = write_onset_results(summary, output_dir)
    return summary, path
