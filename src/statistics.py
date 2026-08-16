"""Dispersion, confidence intervals, and model-comparison tests.

No metric leaves this pipeline as a bare number.  A single figure invites
the reader to treat a 0.02 gap as real when the fold-to-fold spread is
0.30, and the earlier revision of this repository reported exactly such
bare numbers.

Two sources of variability, kept separate
-----------------------------------------
They answer different questions and must not be pooled into one interval:

``seed``
    Re-running the same protocol with a different random seed.  Captures
    training stochasticity only.  For the deterministic baselines
    (majority, persistence, moisture threshold) it is **exactly zero** by
    construction — those models have no random component — so a
    seed-based interval on them has zero width.  That is not precision,
    and :func:`summarize_metric` flags it rather than letting a
    ``± 0.0000`` slip into a table.

``fold``
    Performance across rolling-origin folds.  Captures how the model
    holds up across periods, and on this dataset it is an order of
    magnitude larger than the seed component, because the irrigation
    regime is non-stationary (test-block positive rate ranges from 3 % to
    80 %).  This is the dispersion that actually matters for the claim
    "the model generalises".

Both are reported.  Quoting only the seed interval would understate the
uncertainty by roughly a factor of ten.

Comparing two models
--------------------
:func:`bootstrap_auc_difference` resamples the pooled out-of-fold
predictions to test whether one model's AUC really exceeds another's.
It defaults to a **moving-block** bootstrap rather than the i.i.d. kind:
consecutive hours are strongly dependent — irrigation runs in episodes of
median length 2 h and up to 117 h — and resampling individual hours
independently would treat each hour as fresh evidence, understating the
variance and returning p-values that are too small.  Both variants are
available so the difference can be shown rather than asserted.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger(__name__)

DEFAULT_CONFIDENCE: float = 0.95
DEFAULT_N_BOOTSTRAP: int = 10_000

#: Block length (hours) for the moving-block bootstrap.  One day, chosen
#: to span the dominant diurnal cycle of the weather covariates while
#: staying far below the length of the record.
DEFAULT_BLOCK_HOURS: int = 24


# ======================================================================
# Confidence intervals
# ======================================================================

#: All reported metrics are proportions or Brier scores, hence bounded.
METRIC_BOUNDS: tuple[float, float] = (0.0, 1.0)

#: Below this, a sample is treated as having no dispersion at all.
#:
#: An exact ``std == 0`` test is not safe: ten bit-identical replicates of
#: 0.775 yield ``std(ddof=1) = 1.17e-16`` because the two-pass formula
#: subtracts a rounded mean.  Judging determinism on that would silently
#: report a 1e-16-wide interval as though it were a real measurement.
DETERMINISM_TOLERANCE: float = 1e-12


def summarize_metric(
    values: Sequence[float],
    confidence: float = DEFAULT_CONFIDENCE,
    *,
    label: str = "",
    bounds: Optional[tuple[float, float]] = METRIC_BOUNDS,
) -> Dict[str, float]:
    """Return mean, SD and a Student-*t* confidence interval.

    The *t* distribution rather than the normal: with 5 folds or 10 seeds
    the sample is far too small for the normal approximation, which would
    give intervals about 25 % too narrow at n = 5.

    The interval is clipped to *bounds*.  Every metric here lives in
    [0, 1], yet a symmetric *t* interval around a mean near an edge
    happily runs past it — an unclipped fold-level interval on recall
    reached ``[-0.083, 1.039]`` on this dataset.  Clipping trades exact
    nominal coverage for a reportable number; the raw half-width is kept
    in ``half_width`` for anyone who needs the unclipped form.

    Args:
        values: Replicate measurements of one metric.
        confidence: Coverage of the interval (default 0.95).
        label: Name used in diagnostic messages.
        bounds: ``(low, high)`` clipping range, or ``None`` to disable.

    Returns:
        Mapping with ``mean``, ``std``, ``ci_low``, ``ci_high``,
        ``half_width``, ``n`` and ``degenerate``.  NaN values are dropped
        first; ``ci_*`` are NaN when fewer than two finite replicates
        remain, since dispersion is undefined there.

    Raises:
        ValueError: If *confidence* is not in (0, 1).
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}.")

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)

    # ``degenerate`` is None rather than False when it cannot be
    # determined: a single replicate says nothing about whether the
    # estimator is stochastic, and reporting False would assert that it
    # is on no evidence.
    nan_result: Dict[str, Any] = {
        "mean": float("nan"), "std": float("nan"),
        "ci_low": float("nan"), "ci_high": float("nan"),
        "half_width": float("nan"), "n": n, "degenerate": None,
    }
    if n == 0:
        return nan_result

    mean = float(arr.mean())
    if n == 1:
        return {**nan_result, "mean": mean, "n": 1}

    std = float(arr.std(ddof=1))
    degenerate = std <= DETERMINISM_TOLERANCE
    if degenerate:
        # Collapse floating-point residue so the table reads 0.0000
        # rather than 1.17e-16.
        std = 0.0

    if degenerate:
        # A deterministic estimator replicated across seeds. Report the
        # degenerate interval honestly instead of implying certainty.
        logger.debug(
            "Zero dispersion for %s across %d replicates: the estimator is "
            "deterministic, so this interval reflects training "
            "stochasticity (none) and NOT generalisation uncertainty.",
            label or "metric", n,
        )

    half_width = float(
        stats.t.ppf(0.5 + confidence / 2.0, df=n - 1) * std / np.sqrt(n)
    )
    ci_low, ci_high = mean - half_width, mean + half_width

    if bounds is not None:
        ci_low = float(np.clip(ci_low, *bounds))
        ci_high = float(np.clip(ci_high, *bounds))

    return {
        "mean": mean,
        "std": std,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "half_width": half_width,
        "n": n,
        "degenerate": degenerate,
    }


def aggregate_runs(
    results: pd.DataFrame,
    metrics: Sequence[str],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    model_key: str = "model",
    seed_key: str = "seed",
    fold_key: str = "fold",
) -> pd.DataFrame:
    """Aggregate per-(seed, fold) results into a publication table.

    Both dispersion components are computed: the seed-level interval over
    the per-seed means (one replicate per seed, as the protocol
    specifies) and the fold-level interval over the per-fold means.

    The **fold-level interval is the primary one**, carried in
    ``ci_low``/``ci_high``.  On this project every model turns out to be
    deterministic — XGBoost and LightGBM are run without row or column
    subsampling, and the remaining models have no random component at
    all — so ten seeds produce ten identical numbers and the seed-level
    interval has exactly zero width.  Publishing that as *the* confidence
    interval would claim perfect precision for a model whose score varies
    by ±0.30 between periods.  The seed columns are retained under
    ``seed_*`` names, and ``deterministic_across_seeds`` records the
    finding rather than burying it.

    Args:
        results: Long-format results, one row per (model, seed, fold).
        metrics: Metric column names to aggregate.
        confidence: Interval coverage.
        model_key, seed_key, fold_key: Column names.

    Returns:
        One row per (model, metric) with ``mean``, ``std``, ``ci_low``,
        ``ci_high`` (fold-level, primary), plus ``seed_std``,
        ``seed_ci_low``, ``seed_ci_high``, ``n_seeds``, ``n_folds`` and
        ``deterministic_across_seeds``.
    """
    rows = []
    for model, group in results.groupby(model_key, sort=False):
        for metric in metrics:
            per_seed = group.groupby(seed_key)[metric].mean()
            per_fold = group.groupby(fold_key)[metric].mean()

            seed_stats = summarize_metric(
                per_seed.to_numpy(), confidence,
                label=f"{model}/{metric} (seed)",
            )
            fold_stats = summarize_metric(
                per_fold.to_numpy(), confidence,
                label=f"{model}/{metric} (fold)",
            )

            rows.append({
                "model": model,
                "metric": metric,
                "mean": fold_stats["mean"],
                "std": fold_stats["std"],
                "ci_low": fold_stats["ci_low"],
                "ci_high": fold_stats["ci_high"],
                "seed_std": seed_stats["std"],
                "seed_ci_low": seed_stats["ci_low"],
                "seed_ci_high": seed_stats["ci_high"],
                "n_seeds": seed_stats["n"],
                "n_folds": fold_stats["n"],
                "deterministic_across_seeds": seed_stats["degenerate"],
            })

    table = pd.DataFrame(rows)

    determinism = table["deterministic_across_seeds"]
    if not table.empty and determinism.notna().all() and determinism.all():
        logger.warning(
            "Every model is deterministic: all %d seeds produced identical "
            "results, so the seed-level interval has zero width and carries "
            "no information. The reported CI is the fold-level one. To make "
            "the seed dimension informative, the tree models would need row "
            "or column subsampling enabled (subsample < 1, "
            "colsample_bytree < 1).",
            int(table["n_seeds"].max()),
        )
    return table


# ======================================================================
# Bootstrap comparison
# ======================================================================

def _moving_block_indices(
    n: int, block_size: int, rng: np.random.Generator,
) -> np.ndarray:
    """Draw a resample of length *n* as concatenated contiguous blocks."""
    if block_size <= 1:
        return rng.integers(0, n, size=n)

    block_size = min(block_size, n)
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n - block_size + 1, size=n_blocks)
    offsets = np.arange(block_size)
    return (starts[:, None] + offsets[None, :]).ravel()[:n]


def bootstrap_auc_difference(
    y_true: Sequence[int],
    proba_a: Sequence[float],
    proba_b: Sequence[float],
    *,
    n_iterations: int = DEFAULT_N_BOOTSTRAP,
    block_size: int = DEFAULT_BLOCK_HOURS,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
    metric: str = "roc_auc",
) -> Dict[str, float]:
    """Test whether model *a* outranks model *b*, by paired bootstrap.

    Both models are scored on the **same** resampled rows at every
    iteration, so the paired structure is preserved and the shared
    difficulty of each row cancels out.

    Args:
        y_true: Ground-truth labels of the pooled out-of-fold sample.
        proba_a: Positive-class probabilities from the first model.
        proba_b: Positive-class probabilities from the second model.
        n_iterations: Bootstrap replicates (10 000 by protocol).
        block_size: Moving-block length in hours; ``1`` gives the
            i.i.d. bootstrap, which is anti-conservative on this
            autocorrelated series.
        confidence: Coverage of the reported interval.
        seed: Seed for the resampling generator.
        metric: ``"roc_auc"`` or ``"pr_auc"``.

    Returns:
        Mapping with ``observed_diff`` (a − b), ``p_value`` (two-sided),
        ``ci_low``/``ci_high`` for the difference, ``n_valid``
        iterations, plus the settings used.

    Raises:
        ValueError: On length mismatch, unknown *metric*, or a target
            containing a single class.
    """
    scorer = {
        "roc_auc": roc_auc_score,
        "pr_auc": average_precision_score,
    }.get(metric)
    if scorer is None:
        raise ValueError(
            f"metric must be 'roc_auc' or 'pr_auc', got '{metric}'."
        )

    y_true = np.asarray(y_true)
    proba_a = np.asarray(proba_a, dtype=float)
    proba_b = np.asarray(proba_b, dtype=float)

    if not (len(y_true) == len(proba_a) == len(proba_b)):
        raise ValueError(
            f"Length mismatch: y={len(y_true)}, a={len(proba_a)}, "
            f"b={len(proba_b)}."
        )
    if len(np.unique(y_true)) < 2:
        raise ValueError(
            "Cannot compare AUCs on a sample containing a single class."
        )

    observed = float(scorer(y_true, proba_a) - scorer(y_true, proba_b))

    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_iterations, dtype=float)
    n_valid = 0

    for i in range(n_iterations):
        idx = _moving_block_indices(n, block_size, rng)
        y_boot = y_true[idx]
        # A resample can miss the minority class entirely; such draws
        # carry no information about ranking and are discarded rather
        # than scored as zero.
        if len(np.unique(y_boot)) < 2:
            continue
        diffs[n_valid] = (
            scorer(y_boot, proba_a[idx]) - scorer(y_boot, proba_b[idx])
        )
        n_valid += 1

    if n_valid < 2:
        raise ValueError(
            f"Only {n_valid} of {n_iterations} bootstrap resamples contained "
            "both classes; the sample is too imbalanced for this test."
        )

    diffs = diffs[:n_valid]

    # Two-sided percentile p-value: how much of the bootstrap
    # distribution sits on the far side of zero, doubled.
    p_value = 2.0 * min(
        float(np.mean(diffs <= 0.0)), float(np.mean(diffs >= 0.0)),
    )
    p_value = float(np.clip(p_value, 0.0, 1.0))

    alpha = 1.0 - confidence
    ci_low, ci_high = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])

    logger.info(
        "Bootstrap %s difference: %+.4f (95%% CI %+.4f…%+.4f), "
        "p=%.4f, %d/%d valid resamples, block=%d h",
        metric, observed, ci_low, ci_high, p_value,
        n_valid, n_iterations, block_size,
    )

    return {
        "metric": metric,
        "observed_diff": observed,
        "p_value": p_value,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_valid": n_valid,
        "n_iterations": n_iterations,
        "block_size": block_size,
        "seed": seed,
    }


def format_p_value(p: float, n_iterations: int) -> str:
    """Render a bootstrap p-value without implying unavailable resolution.

    A percentile bootstrap cannot resolve below ``1 / n_iterations``, so
    an empirical zero is reported as ``< 1e-4`` rather than ``p = 0``.
    """
    floor = 1.0 / n_iterations
    if p < floor:
        return f"< {floor:.0e}"
    return f"{p:.4f}"
