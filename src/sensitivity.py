"""Per-fold metric export and robustness to an episode-dominated fold.

The problem
-----------
One irrigation episode runs for 117 consecutive hours and supplies 37.7 %
of all positive hours in the design matrix. It falls almost entirely
inside a single rolling-origin fold, whose test block is consequently
79.8 % positive. Both gradient-boosted models reach a PR-AUC of 1.000
there, which lifts their five-fold means by roughly 0.10.

Predicting the interior of one long episode is close to predicting
persistence, so a near-perfect score on such a fold is not evidence of
generalisation. The aggregate needs to be reported with and without it.

Why the exclusion rule is data-driven
-------------------------------------
Dropping a fold *because its metric looked too high* would be selection
after the fact — the reader could not tell it from cherry-picking, and the
rule would not transfer to another dataset. Instead, a fold is excluded
when a stated share of its irrigating hours comes from one episode
(:attr:`~src.config.ValidationConfig.episode_dominance_threshold`,
default 0.5). That criterion is a property of the data, computable before
any model is fitted, and it happens to select exactly one fold here —
dominance 0.67 against 0.26–0.36 elsewhere.

Both views are exported. Neither is presented as the "true" number: the
all-folds mean covers the record as observed, the reduced mean covers it
minus its easiest stretch, and the gap between them is itself the result
worth reporting.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.metrics import PRIMARY_METRICS
from src.statistics import summarize_metric

logger = logging.getLogger(__name__)

PER_FOLD_METRICS_CSV: str = "per_fold_metrics.csv"
SENSITIVITY_SUMMARY_CSV: str = "sensitivity_summary.csv"

SUBSET_ALL: str = "all_folds"
SUBSET_REDUCED: str = "excluding_episode_dominated"

#: Columns carried through to the per-fold export, beyond the metrics.
_CONTEXT_COLUMNS: Sequence[str] = (
    "n_test", "n_positive", "positive_rate", "n_train",
)


def collect_per_fold_metrics(
    protocols: Dict[str, pd.DataFrame],
    *,
    metrics: Sequence[str] = PRIMARY_METRICS,
) -> pd.DataFrame:
    """Flatten every protocol's per-fold results into one long table.

    Per-fold numbers existed only in printed output before this; a
    reviewer asking "which fold produced that mean?" had nothing to open.

    Args:
        protocols: Protocol name → long-format results carrying at least
            ``model``, ``fold`` and the metric columns.
        metrics: Metric columns to export.

    Returns:
        One row per (protocol, model, fold) with the metrics and the
        context needed to read them, renaming ``positive_rate`` to
        ``test_positive_rate`` for clarity.

    Raises:
        ValueError: If a protocol's frame lacks ``model`` or ``fold``.
    """
    frames: List[pd.DataFrame] = []

    for name, results in protocols.items():
        if results is None or results.empty:
            logger.info("Protocol '%s' produced no rows; skipped.", name)
            continue

        missing = [c for c in ("model", "fold") if c not in results.columns]
        if missing:
            raise ValueError(
                f"Protocol '{name}' is missing required columns: {missing}."
            )

        columns = ["model", "fold"]
        columns += [c for c in metrics if c in results.columns]
        columns += [c for c in _CONTEXT_COLUMNS if c in results.columns]

        frame = results[columns].copy()
        frame.insert(0, "protocol", name)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    table = pd.concat(frames, ignore_index=True)
    return table.rename(columns={"positive_rate": "test_positive_rate"})


def _subset_rows(
    results: pd.DataFrame, excluded_folds: Sequence[int],
) -> pd.DataFrame:
    """Drop the excluded folds from a protocol's per-fold results."""
    if not excluded_folds:
        return results
    return results[~results["fold"].isin(list(excluded_folds))]


def sensitivity_summary(
    protocols: Dict[str, pd.DataFrame],
    excluded_folds: Sequence[int] | Dict[str, Sequence[int]],
    *,
    metrics: Sequence[str] = PRIMARY_METRICS,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Aggregate each protocol twice: over all folds, and without the
    episode-dominated ones.

    Args:
        protocols: Protocol name → long-format per-fold results.
        excluded_folds: Either one fold list applied to every protocol,
            or a per-protocol mapping.

            **Prefer the mapping.** Fold numbers are not comparable
            across protocols that split different row sets: the onset
            protocol runs over 1 003 eligible hours rather than 1 313, so
            its fold 4 covers a different period from the main task's
            fold 4. Applying one list to both would exclude an
            arbitrary period from the onset results. In practice the
            onset protocol has *no* dominated fold, because restricting
            to hours with the valve closed already removes the interior
            of the long episode.
        metrics: Metrics to aggregate.
        confidence: Interval coverage.

    Returns:
        One row per (protocol, model, metric, subset) with ``mean``,
        ``std``, ``ci_low``, ``ci_high``, ``n_folds`` and a
        ``delta_vs_all_folds`` column giving the shift caused by the
        exclusion.  ``excluded_folds`` records which folds the subset
        dropped, so the table is self-describing.
    """
    rows: List[Dict[str, object]] = []

    for name, results in protocols.items():
        if results is None or results.empty:
            continue

        if isinstance(excluded_folds, dict):
            protocol_excluded = excluded_folds.get(name, [])
        else:
            protocol_excluded = excluded_folds
        excluded = sorted(int(f) for f in protocol_excluded)

        subsets = {
            SUBSET_ALL: (results, []),
            SUBSET_REDUCED: (_subset_rows(results, excluded), excluded),
        }

        for model, group in results.groupby("model", sort=False):
            for metric in metrics:
                if metric not in results.columns:
                    continue

                baseline = float("nan")
                for subset_name, (subset_frame, dropped) in subsets.items():
                    subset_group = subset_frame[subset_frame["model"] == model]
                    stats_ = summarize_metric(
                        subset_group[metric].to_numpy(),
                        confidence,
                        label=f"{name}/{model}/{metric}/{subset_name}",
                    )
                    if subset_name == SUBSET_ALL:
                        baseline = stats_["mean"]

                    rows.append({
                        "protocol": name,
                        "model": model,
                        "metric": metric,
                        "subset": subset_name,
                        "excluded_folds": (
                            "none" if not dropped
                            else ", ".join(str(f) for f in dropped)
                        ),
                        "n_folds": stats_["n"],
                        "mean": stats_["mean"],
                        "std": stats_["std"],
                        "ci_low": stats_["ci_low"],
                        "ci_high": stats_["ci_high"],
                        "delta_vs_all_folds": (
                            0.0 if subset_name == SUBSET_ALL
                            else stats_["mean"] - baseline
                        ),
                    })

    table = pd.DataFrame(rows)
    if not table.empty:
        per_protocol = (
            table[table["subset"] == SUBSET_REDUCED]
            .groupby("protocol")["excluded_folds"].first().to_dict()
        )
        logger.info(
            "Sensitivity table: %d rows over %d protocols; excluded folds "
            "per protocol = %s",
            len(table), table["protocol"].nunique(), per_protocol,
        )
    return table


def ranking_is_preserved(
    summary: pd.DataFrame, protocol: str, metric: str,
) -> bool:
    """Whether excluding the dominated folds reorders the models.

    The question that decides how much the exclusion matters: if the
    ranking survives, the dominated fold inflated absolute values without
    changing which model wins, and the paper can report the all-folds
    numbers with a caveat rather than restructuring around the reduced
    set.

    Args:
        summary: Output of :func:`sensitivity_summary`.
        protocol: Protocol to inspect.
        metric: Metric to rank by.

    Returns:
        ``True`` when both subsets produce the same model order.
    """
    block = summary[
        (summary["protocol"] == protocol) & (summary["metric"] == metric)
    ]
    orders = {}
    for subset in (SUBSET_ALL, SUBSET_REDUCED):
        subset_block = block[block["subset"] == subset]
        orders[subset] = list(
            subset_block.sort_values("mean", ascending=False)["model"]
        )
    return orders[SUBSET_ALL] == orders[SUBSET_REDUCED]


# ======================================================================
# Reporting and export
# ======================================================================

def print_sensitivity(
    summary: pd.DataFrame,
    fold_table: pd.DataFrame,
    excluded_folds: Sequence[int],
    *,
    protocol: str = "main",
    metric: str = "pr_auc",
    threshold: float = 0.5,
) -> None:
    """Print the dominance diagnostic and the two aggregates."""
    print()
    print("## Episode dominance by fold")
    print()
    print("| Fold | Irrigating hours | Episodes | Largest episode | "
          "Dominance | Excluded |")
    print("|-----:|-----------------:|---------:|----------------:|"
          "----------:|:---------|")
    for _, r in fold_table.iterrows():
        dominance = r.get("episode_dominance", float("nan"))
        largest = r.get("largest_episode_hours", float("nan"))
        n_eps = r.get("n_test_episodes", 0)
        total = (
            int(largest / dominance)
            if np.isfinite(dominance) and dominance > 0 else 0
        )
        print(
            f"| {int(r['fold'])} | {total} | {int(n_eps)} "
            f"| {largest:.0f} h | {dominance:.4f} "
            f"| {'**yes**' if r.get('episode_dominated') else 'no'} |"
        )
    print()
    print(f"_A fold is excluded when more than {threshold:.0%} of its "
          f"irrigating hours come from one episode. The rule is stated in "
          f"advance and computed from the data, not chosen after seeing "
          f"which fold scored highest._")
    print()

    block = summary[
        (summary["protocol"] == protocol) & (summary["metric"] == metric)
    ]
    if block.empty:
        return

    print(f"## Robustness to the episode-dominated fold — {metric.upper()}")
    print()
    print("| Model | All folds | Excluding dominated | Δ |")
    print("|:------|----------:|--------------------:|--:|")
    all_rows = block[block["subset"] == SUBSET_ALL].set_index("model")
    reduced = block[block["subset"] == SUBSET_REDUCED].set_index("model")
    for model in all_rows.sort_values("mean", ascending=False).index:
        a = all_rows.loc[model]
        r = reduced.loc[model]
        print(
            f"| {model} | {a['mean']:.4f} "
            f"[{a['ci_low']:.4f}, {a['ci_high']:.4f}] "
            f"| {r['mean']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}] "
            f"| {r['delta_vs_all_folds']:+.4f} |"
        )
    print()

    preserved = ranking_is_preserved(summary, protocol, metric)
    excluded_text = ", ".join(str(f) for f in excluded_folds) or "none"
    print(
        f"_Excluded fold(s) for the `{protocol}` protocol: {excluded_text}. "
        f"Model ranking is "
        f"**{'preserved' if preserved else 'NOT preserved'}** under the "
        f"exclusion._"
    )
    if preserved:
        print()
        print("Because the ranking survives, the dominated fold inflates "
              "absolute values without changing which model wins. The "
              "all-folds figures may be quoted, provided the inflation is "
              "stated alongside them.")
    print()

    other = sorted(set(summary["protocol"]) - {protocol})
    if other:
        print("Exclusions applied per protocol, since fold numbers index "
              "different periods when the row sets differ:")
        print()
        for name in other:
            block_other = summary[
                (summary["protocol"] == name)
                & (summary["subset"] == SUBSET_REDUCED)
            ]
            dropped = (
                block_other["excluded_folds"].iloc[0]
                if not block_other.empty else "none"
            )
            print(f"* `{name}` — excluded: {dropped}")
        print()


def write_per_fold_metrics(
    table: pd.DataFrame,
    output_dir: str | Path,
    *,
    filename: str = PER_FOLD_METRICS_CSV,
) -> Path:
    """Write the per-fold metric table."""
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    logger.info("Wrote %s (%d rows)", path, len(table))
    return path


def write_sensitivity_summary(
    table: pd.DataFrame,
    output_dir: str | Path,
    *,
    filename: str = SENSITIVITY_SUMMARY_CSV,
) -> Path:
    """Write the sensitivity table."""
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    logger.info("Wrote %s (%d rows)", path, len(table))
    return path


def dominance_metadata(
    fold_table: pd.DataFrame,
    excluded_folds: Sequence[int],
    *,
    threshold: float,
    largest_episode_hours: int,
    n_positive_design_matrix: int,
    n_positive_full_record: int,
) -> Dict[str, object]:
    """Build the run-metadata block describing the exclusion rule.

    Args:
        fold_table: Output of :func:`src.validation.describe_folds`.
        excluded_folds: Folds flagged as dominated.
        threshold: Dominance threshold applied.
        largest_episode_hours: Duration of the longest episode.
        n_positive_design_matrix: Positive hours in the design matrix.
        n_positive_full_record: Positive hours in the hourly record.

    Returns:
        JSON-serialisable mapping.
    """
    return {
        "threshold": threshold,
        "rule": (
            "A fold is episode-dominated when more than the threshold share "
            "of its irrigating test hours comes from a single episode. "
            "Computed from the episode labelling, so it describes the "
            "period rather than any particular target and applies "
            "unchanged to the onset protocol. Stated in advance, not "
            "chosen after inspecting the metrics."
        ),
        "excluded_folds": [int(f) for f in excluded_folds],
        "dominance_by_fold": {
            str(int(r["fold"])): (
                None if not np.isfinite(r.get("episode_dominance", np.nan))
                else round(float(r["episode_dominance"]), 4)
            )
            for _, r in fold_table.iterrows()
        },
        "largest_episode": {
            "duration_hours": int(largest_episode_hours),
            "share_of_design_matrix_positives": (
                round(largest_episode_hours / n_positive_design_matrix, 4)
                if n_positive_design_matrix else None
            ),
            "share_of_full_record_positives": (
                round(largest_episode_hours / n_positive_full_record, 4)
                if n_positive_full_record else None
            ),
            "note": (
                "One episode supplies this share of all positive hours. It "
                "lies almost entirely within a single rolling-origin fold, "
                "whose test block is consequently dominated by it."
            ),
        },
    }
