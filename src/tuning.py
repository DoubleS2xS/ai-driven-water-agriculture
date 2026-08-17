"""Nested cross-validation for honest hyperparameter selection.

The problem this closes
-----------------------
The main comparison runs the gradient-boosted models at library defaults
(``n_estimators=200``, ``max_depth=6``) inherited from an earlier
revision, while the moisture-threshold baseline *is* tuned — its
cut-point is fitted on each training fold. That asymmetry is a fair
objection to the headline finding: it compares an untuned learner against
a fitted rule, so "trees do not beat the threshold" could in principle be
an artifact of the defaults rather than a property of the data.

Nested cross-validation removes the objection without introducing a worse
one. Tuning on the outer test block would be straightforward and
completely invalid; instead each outer training fold is split again, the
grid is scored on those **inner** folds only, and the winner is refitted
on the whole outer training fold before touching the outer test block.
The outer test data influence nothing except the final score.

Structure
---------
For outer fold *k* with training rows ``T_k``:

1. Build inner rolling-origin folds **within** ``T_k``.
2. Score every grid point by the mean metric across inner folds.
3. Refit the winner on all of ``T_k``.
4. Score once on the outer test block.

Selected hyperparameters are recorded per fold. They usually differ
between folds, and that is information rather than noise: an early fold
with 7 positive examples needs a far smaller model than a late fold with
252, so a single "best" configuration for the whole series does not
exist.

Cost
----
The grid is deliberately small — 12 points over depth, ensemble size and
minimum child weight — because the inner folds of early outer folds
contain single-digit positive counts. A larger grid searched against that
little signal would select on noise, and the selection variance would
exceed any gain. Learning rate is held fixed for the same reason.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config import ValidationConfig
from src.metrics import PRIMARY_METRICS, compute_classification_metrics
from src.statistics import summarize_metric
from src.validation import rolling_origin_splits

logger = logging.getLogger(__name__)

#: Inner folds used to score the grid inside each outer training fold.
DEFAULT_INNER_FOLDS: int = 3

#: Metric optimised during selection.  PR-AUC rather than ROC-AUC: at a
#: 4–24 % positive rate it responds to false alarms, which is the failure
#: mode an irrigation controller cares about.
DEFAULT_SELECTION_METRIC: str = "pr_auc"

#: Search space.  Small on purpose — see the module docstring.
PARAM_GRID: Dict[str, Sequence[Any]] = {
    "max_depth": (2, 4, 6),
    "n_estimators": (100, 200),
    "min_child_weight": (1, 5),
}

NESTED_CV_CSV: str = "nested_cv.csv"


def _grid_points(grid: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    """Expand a parameter grid into a list of concrete settings."""
    keys = list(grid)
    return [
        dict(zip(keys, values))
        for values in itertools.product(*(grid[k] for k in keys))
    ]


def _translate_params(model_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Map the shared grid onto backend-specific argument names.

    LightGBM calls the minimum-child-weight control
    ``min_child_samples``; passing XGBoost's name would be silently
    ignored, leaving that axis of the grid untuned for one of the two
    models and making the comparison between them unfair.
    """
    if model_name != "lightgbm":
        return dict(params)

    translated = dict(params)
    if "min_child_weight" in translated:
        translated["min_child_samples"] = translated.pop("min_child_weight")
    return translated


def _score_on_splits(
    factory_kwargs: Dict[str, Any],
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    metric: str,
    seed: int,
) -> float:
    """Mean *metric* of one configuration across the given folds.

    Folds where the metric is undefined — no positive example in the test
    block — are skipped rather than counted as zero, which would penalise
    a configuration for a property of the fold.
    """
    from src.evaluate_pipeline import make_model_factory

    factory = make_model_factory(model_name, **factory_kwargs)

    scores: List[float] = []
    for train_idx, test_idx in splits:
        model = factory(seed)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = np.asarray(model.predict(X.iloc[test_idx]))
        y_proba = np.asarray(model.predict_proba(X.iloc[test_idx]))[:, 1]
        value = compute_classification_metrics(
            y.iloc[test_idx], y_pred, y_proba,
        )[metric]
        if np.isfinite(value):
            scores.append(float(value))

    return float(np.mean(scores)) if scores else float("nan")


def select_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    *,
    grid: Optional[Dict[str, Sequence[Any]]] = None,
    inner_folds: int = DEFAULT_INNER_FOLDS,
    metric: str = DEFAULT_SELECTION_METRIC,
    seed: int = 42,
) -> Tuple[Dict[str, Any], float, int]:
    """Choose hyperparameters using rolling-origin folds inside *X*.

    *X* and *y* must be an outer fold's **training** rows only.  Nothing
    here may see the outer test block.

    Args:
        X: Training-fold feature matrix.
        y: Training-fold target.
        model_name: ``"xgboost"`` or ``"lightgbm"``.
        grid: Search space; defaults to :data:`PARAM_GRID`.
        inner_folds: Inner rolling-origin fold count.
        metric: Metric to maximise.
        seed: Random seed.

    Returns:
        ``(best_params, best_score, n_candidates_scored)``.  When the
        training fold is too small or too sparse for any candidate to be
        scored, returns empty params so the caller falls back to library
        defaults — the honest outcome when there is nothing to select on.
    """
    grid = grid or PARAM_GRID
    candidates = _grid_points(grid)

    inner_config = ValidationConfig(n_folds=inner_folds)
    try:
        inner_splits = rolling_origin_splits(len(X), inner_config)
    except ValueError as exc:
        logger.warning(
            "Cannot build %d inner folds from %d training rows (%s); "
            "falling back to library defaults for this outer fold.",
            inner_folds, len(X), exc,
        )
        return {}, float("nan"), 0

    best_params: Dict[str, Any] = {}
    best_score = -np.inf
    n_scored = 0

    for params in candidates:
        score = _score_on_splits(
            _translate_params(model_name, params),
            model_name, X, y, inner_splits, metric, seed,
        )
        if not np.isfinite(score):
            continue
        n_scored += 1
        if score > best_score:
            best_score = score
            best_params = params

    if n_scored == 0:
        logger.warning(
            "No candidate could be scored on the inner folds (%d rows, "
            "%d positives): the inner test blocks contain no positive "
            "examples. Falling back to library defaults.",
            len(X), int(y.sum()),
        )
        return {}, float("nan"), 0

    logger.info(
        "Selected %s (inner %s = %.4f, %d of %d candidates scorable)",
        best_params, metric, best_score, n_scored, len(candidates),
    )
    return best_params, float(best_score), n_scored


def nested_cv_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    outer_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    model_names: Sequence[str] = ("xgboost", "lightgbm"),
    grid: Optional[Dict[str, Sequence[Any]]] = None,
    inner_folds: int = DEFAULT_INNER_FOLDS,
    metric: str = DEFAULT_SELECTION_METRIC,
    seed: int = 42,
) -> pd.DataFrame:
    """Run nested cross-validation and score each outer fold.

    Args:
        X: Full causal feature matrix, time-ordered.
        y: Binary target.
        outer_splits: Outer fold definitions.
        model_names: Models to tune.
        grid: Search space.
        inner_folds: Inner fold count.
        metric: Metric optimised during selection.
        seed: Random seed.

    Returns:
        One row per (model, outer fold) with the selected parameters, the
        inner score they achieved, and the outer-test metrics.
    """
    from src.evaluate_pipeline import make_model_factory

    rows: List[Dict[str, Any]] = []

    for model_name in model_names:
        for fold, (train_idx, test_idx) in enumerate(outer_splits, start=1):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

            logger.info(
                "── Nested CV: %s, outer fold %d (%d train rows, "
                "%d positives) ──",
                model_name, fold, len(X_train), int(y_train.sum()),
            )

            params, inner_score, n_scored = select_hyperparameters(
                X_train, y_train, model_name,
                grid=grid, inner_folds=inner_folds,
                metric=metric, seed=seed,
            )

            # Refit the winner on the whole outer training fold, then
            # score once. This is the only point the outer test is used.
            model = make_model_factory(
                model_name, **_translate_params(model_name, params),
            )(seed)
            model.fit(X_train, y_train)
            y_pred = np.asarray(model.predict(X_test))
            y_proba = np.asarray(model.predict_proba(X_test))[:, 1]
            outer_metrics = compute_classification_metrics(
                y_test, y_pred, y_proba,
            )

            rows.append({
                "model": model_name,
                "fold": fold,
                "n_train": len(train_idx),
                "n_train_positives": int(y_train.sum()),
                "n_test": len(test_idx),
                "selected_params": (
                    ", ".join(f"{k}={v}" for k, v in sorted(params.items()))
                    if params else "library defaults (nothing selectable)"
                ),
                **{f"selected_{k}": v for k, v in params.items()},
                "inner_score": inner_score,
                "n_candidates_scored": n_scored,
                **outer_metrics,
            })

    return pd.DataFrame(rows)


def summarise_nested_cv(
    results: pd.DataFrame,
    metrics: Sequence[str] = PRIMARY_METRICS,
) -> pd.DataFrame:
    """Aggregate nested-CV outer scores into mean ± SD with intervals."""
    rows = []
    for model, group in results.groupby("model", sort=False):
        row: Dict[str, Any] = {
            "model": model,
            "n_folds": int(group["fold"].nunique()),
            "params_per_fold": " | ".join(
                f"f{int(r.fold)}: {r.selected_params}"
                for r in group.itertuples()
            ),
        }
        for metric in metrics:
            stats_ = summarize_metric(
                group[metric].to_numpy(), label=f"nested/{model}/{metric}",
            )
            row[f"{metric}_mean"] = stats_["mean"]
            row[f"{metric}_std"] = stats_["std"]
            row[f"{metric}_ci_low"] = stats_["ci_low"]
            row[f"{metric}_ci_high"] = stats_["ci_high"]
        rows.append(row)
    return pd.DataFrame(rows)


def print_nested_cv(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    untuned: Optional[pd.DataFrame] = None,
    metric: str = DEFAULT_SELECTION_METRIC,
) -> None:
    """Print selected parameters per fold and the tuned-vs-untuned result.

    Args:
        results: Output of :func:`nested_cv_evaluate`.
        summary: Output of :func:`summarise_nested_cv`.
        untuned: Optional aggregated table from the untuned run, used to
            state whether tuning changed the conclusion.
        metric: Metric to display.
    """
    print()
    print("## Nested CV — hyperparameters selected per outer fold")
    print()
    print("| Model | Fold | Train rows | Train pos. | Selected | "
          f"inner {metric.upper()} | outer {metric.upper()} |")
    print("|:------|-----:|-----------:|-----------:|:---------|"
          "------------:|------------:|")
    for r in results.itertuples():
        inner = f"{r.inner_score:.4f}" if np.isfinite(r.inner_score) else "—"
        outer = getattr(r, metric)
        outer_text = f"{outer:.4f}" if np.isfinite(outer) else "—"
        print(
            f"| {r.model} | {r.fold} | {r.n_train} | {r.n_train_positives} "
            f"| {r.selected_params} | {inner} | {outer_text} |"
        )
    print()

    print(f"## Nested CV — outer-fold {metric.upper()}")
    print()
    print("| Model | Mean | SD | 95% CI |")
    print("|:------|-----:|---:|:-------|")
    for _, r in summary.iterrows():
        print(
            f"| {r['model']} | {r[f'{metric}_mean']:.4f} "
            f"| {r[f'{metric}_std']:.4f} "
            f"| [{r[f'{metric}_ci_low']:.4f}, {r[f'{metric}_ci_high']:.4f}] |"
        )
    print()

    if untuned is not None:
        print(f"**Tuned versus untuned ({metric.upper()}):**")
        print()
        print("| Model | Untuned | Nested CV | Δ |")
        print("|:------|--------:|----------:|--:|")
        for _, r in summary.iterrows():
            match = untuned[
                (untuned["model"] == r["model"])
                & (untuned["metric"] == metric)
            ]
            if match.empty:
                continue
            before = float(match.iloc[0]["mean"])
            after = float(r[f"{metric}_mean"])
            print(
                f"| {r['model']} | {before:.4f} | {after:.4f} "
                f"| {after - before:+.4f} |"
            )
        print()
        print("_Selection uses inner folds of each outer training block "
              "only; the outer test block is touched once, to produce the "
              "score. Parameters differ between folds because an early "
              "fold with single-digit positive counts supports a much "
              "smaller model than a late one._")
        print()
