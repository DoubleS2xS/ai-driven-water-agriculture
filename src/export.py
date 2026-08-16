"""Write every number destined for the manuscript to a file.

Nothing in the paper should be transcribed from a console.  A figure
copied by hand is a figure nobody can re-derive, and the earlier revision
of this repository carried a README performance table whose numbers the
code never produced — the failure mode this module exists to make
impossible.

Every artefact is regenerated from scratch on each run, so a stale file
cannot survive a change in the pipeline.  Each one records the git commit
it came from, and whether the working tree was dirty at the time: a
result produced from uncommitted code is not reproducible, and saying so
in the file is more useful than discovering it later.

Artefacts
---------
``results_summary.csv``
    Master table: model × metric × mean × SD × 95 % CI.
``baselines.csv``
    The baseline subset, with each figure's distance from the best main
    model and from the no-skill reference.
``ablation.csv``
    Feature set × model × metric.
``feature_importance.csv``
    Mean |SHAP| per feature, ranked, tagged with its feature block.
``folds.csv``
    Rolling-origin fold structure — periods, sizes, class balance.
``model_comparison.json``
    The paired bootstrap test: difference, CI, p-value.
``run_metadata.json``
    Library versions, git commit, dataset shape, class balance, site
    configuration.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from src.metrics import PRIMARY_METRICS

logger = logging.getLogger(__name__)

#: Packages whose versions materially affect the numbers.
TRACKED_PACKAGES: tuple[str, ...] = (
    "numpy", "pandas", "scipy", "sklearn", "xgboost", "lightgbm", "shap",
)

RESULTS_SUMMARY_CSV: str = "results_summary.csv"
BASELINES_CSV: str = "baselines.csv"
ABLATION_CSV: str = "ablation.csv"
FEATURE_IMPORTANCE_CSV: str = "feature_importance.csv"
FOLDS_CSV: str = "folds.csv"
MODEL_COMPARISON_JSON: str = "model_comparison.json"
RUN_METADATA_JSON: str = "run_metadata.json"


# ======================================================================
# Provenance helpers
# ======================================================================

def _git(*args: str) -> Optional[str]:
    """Run a git command, returning ``None`` outside a repository."""
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_provenance() -> Dict[str, Any]:
    """Return the commit the run came from and whether the tree was dirty.

    A dirty tree means the artefacts do not correspond to any commit, so
    the flag is recorded rather than silently omitted.
    """
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")

    if commit is None:
        return {
            "commit": None,
            "branch": None,
            "dirty": None,
            "note": "Not a git repository, or git is unavailable.",
        }

    dirty = bool(status)
    provenance = {"commit": commit, "branch": branch, "dirty": dirty}
    if dirty:
        provenance["note"] = (
            "The working tree had uncommitted changes when these results "
            "were produced, so they do not correspond to this commit alone "
            "and are not reproducible from it. Commit before quoting them."
        )
        logger.warning(
            "Results generated from a dirty working tree (commit %s). "
            "They cannot be reproduced from that commit alone.",
            commit[:8],
        )
    return provenance


def package_versions() -> Dict[str, Optional[str]]:
    """Return the installed version of each tracked package."""
    versions: Dict[str, Optional[str]] = {
        "python": platform.python_version(),
    }
    for name in TRACKED_PACKAGES:
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", None)
        except ImportError:
            versions[name] = None
    return versions


# ======================================================================
# Writers
# ======================================================================

def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    logger.info("Wrote %s (%d rows)", path, len(frame))
    return path


def write_results_summary(
    summary: pd.DataFrame,
    output_dir: str | Path,
    *,
    filename: str = RESULTS_SUMMARY_CSV,
) -> Path:
    """Write the master model × metric table.

    ``ci_low``/``ci_high`` are the **fold-level** interval.  On this
    project every estimator is deterministic, so the seed-level interval
    has zero width and would misrepresent the precision if quoted as the
    confidence interval; it is preserved in the ``seed_*`` columns and
    flagged by ``deterministic_across_seeds``.  See
    :func:`src.statistics.aggregate_runs`.

    Args:
        summary: Output of :func:`src.statistics.aggregate_runs`.
        output_dir: Destination directory.
        filename: Output file name.

    Returns:
        The path written.
    """
    columns = [
        "model", "metric", "mean", "std", "ci_low", "ci_high",
        "seed_std", "seed_ci_low", "seed_ci_high",
        "n_folds", "n_seeds", "deterministic_across_seeds",
    ]
    frame = summary[[c for c in columns if c in summary.columns]].copy()
    frame = frame.sort_values(["metric", "mean"], ascending=[True, False])
    return _write_csv(frame, Path(output_dir) / filename)


def write_baselines(
    summary: pd.DataFrame,
    output_dir: str | Path,
    *,
    baseline_names: Sequence[str],
    main_model_names: Sequence[str],
    positive_rate: float,
    filename: str = BASELINES_CSV,
) -> Path:
    """Write the baseline table with its two reference points.

    A baseline figure is only meaningful next to what it is being
    compared against, so each row carries the no-skill value for its
    metric and the gap to the best main model.  A positive
    ``delta_vs_best_main`` means the baseline *beat* the best main model,
    which on this dataset happens for several metrics.

    Args:
        summary: Output of :func:`src.statistics.aggregate_runs`.
        output_dir: Destination directory.
        baseline_names: Models to treat as baselines.
        main_model_names: Models to treat as the main comparison.
        positive_rate: Class prevalence, the no-skill PR-AUC.
        filename: Output file name.

    Returns:
        The path written.
    """
    baselines = summary[summary["model"].isin(baseline_names)].copy()

    no_skill = {
        "roc_auc": 0.5,
        "pr_auc": positive_rate,
        "brier": positive_rate * (1.0 - positive_rate),
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }
    baselines["no_skill_reference"] = baselines["metric"].map(no_skill)

    main = summary[summary["model"].isin(main_model_names)]
    best_main = (
        main.loc[main.groupby("metric")["mean"].idxmax(), ["metric", "mean"]]
        .set_index("metric")["mean"]
    )
    baselines["best_main_model_mean"] = baselines["metric"].map(best_main)
    baselines["delta_vs_best_main"] = (
        baselines["mean"] - baselines["best_main_model_mean"]
    )

    columns = [
        "model", "metric", "mean", "std", "ci_low", "ci_high",
        "no_skill_reference", "best_main_model_mean", "delta_vs_best_main",
    ]
    frame = baselines[columns].sort_values(
        ["metric", "mean"], ascending=[True, False],
    )
    return _write_csv(frame, Path(output_dir) / filename)


def write_ablation(
    ablation_summary: pd.DataFrame,
    output_dir: str | Path,
    *,
    filename: str = ABLATION_CSV,
) -> Path:
    """Write the feature-set × model × metric table.

    ``n_rows`` is identical across sets by construction — every set is a
    column subset of one matrix — and is exported so a reader can verify
    that rather than take it on trust.
    """
    return _write_csv(
        ablation_summary.sort_values(["feature_set", "model"]),
        Path(output_dir) / filename,
    )


def write_feature_importance(
    manifest: Dict[str, Any],
    output_dir: str | Path,
    *,
    blocks: Optional[Dict[str, Sequence[str]]] = None,
    filename: str = FEATURE_IMPORTANCE_CSV,
) -> Path:
    """Write mean |SHAP| per feature, ranked.

    Each feature is tagged with its block, so the ablation table and the
    importance table can be read against one another — a block that
    dominates the importance ranking but adds nothing in the ablation is
    a finding, not a contradiction.

    Args:
        manifest: Output of :func:`src.models.explanation.explain_model`.
        output_dir: Destination directory.
        blocks: Block-to-columns mapping used to tag each feature.
        filename: Output file name.

    Returns:
        The path written.
    """
    importance = manifest["mean_abs_shap"]
    frame = pd.DataFrame(
        {"feature": list(importance), "mean_abs_shap": list(importance.values())}
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    frame.insert(0, "rank", frame.index + 1)

    if blocks:
        lookup = {
            column: block
            for block, columns in blocks.items()
            for column in columns
        }
        frame["block"] = frame["feature"].map(lookup)

    frame["model"] = manifest.get("model")
    frame["fold"] = manifest.get("fold")
    return _write_csv(frame, Path(output_dir) / filename)


def write_folds(
    fold_table: pd.DataFrame,
    output_dir: str | Path,
    *,
    filename: str = FOLDS_CSV,
) -> Path:
    """Write the rolling-origin fold structure."""
    return _write_csv(fold_table, Path(output_dir) / filename)


def write_model_comparison(
    comparison: Optional[Dict[str, Any]],
    output_dir: str | Path,
    *,
    filename: str = MODEL_COMPARISON_JSON,
) -> Path:
    """Write the paired bootstrap test result.

    Args:
        comparison: Output of
            :func:`src.evaluate_pipeline.compare_best_model_to_best_baseline`,
            or ``None`` if the test could not be run.
        output_dir: Destination directory.
        filename: Output file name.

    Returns:
        The path written.
    """
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = (
        {"available": False, "reason": "No candidate on one side."}
        if comparison is None
        else {
            "available": True,
            **comparison,
            "interpretation": (
                "The 95% CI of the difference excludes zero; the ranking is "
                "distinguishable from chance on this sample."
                if not (
                    float(comparison["ci_low"]) < 0.0 < float(comparison["ci_high"])
                )
                else "The 95% CI of the difference spans zero; the two models "
                     "are not statistically distinguishable on this sample."
            ),
            "p_value_note": (
                "p_value uses a moving-block bootstrap and is the one to "
                "quote. p_value_iid ignores temporal dependence and is "
                "reported only to show the size of that effect."
            ),
        }
    )

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
    logger.info("Wrote %s", path)
    return path


def write_run_metadata(
    output_dir: str | Path,
    *,
    n_rows: int,
    n_features: int,
    positive_rate: float,
    seeds: Sequence[int],
    n_folds: int,
    dataset_start: str,
    dataset_end: str,
    loader_config: Any = None,
    extra: Optional[Dict[str, Any]] = None,
    filename: str = RUN_METADATA_JSON,
) -> Path:
    """Write the run's provenance and environment.

    Args:
        output_dir: Destination directory.
        n_rows: Rows in the design matrix.
        n_features: Feature count.
        positive_rate: Class prevalence.
        seeds: Seeds used for the repeated runs.
        n_folds: Rolling-origin fold count.
        dataset_start, dataset_end: Period covered, UTC.
        loader_config: Optional :class:`~src.config.DataLoaderConfig`,
            recorded so the site and timezone assumptions travel with the
            results.
        extra: Additional fields to merge in.
        filename: Output file name.

    Returns:
        The path written.
    """
    metadata: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_provenance(),
        "environment": {
            "platform": platform.platform(),
            "python_executable": sys.executable,
            **package_versions(),
        },
        "dataset": {
            "n_rows": int(n_rows),
            "n_features": int(n_features),
            "positive_rate": float(positive_rate),
            "n_positive": int(round(positive_rate * n_rows)),
            "start_utc": str(dataset_start),
            "end_utc": str(dataset_end),
        },
        "protocol": {
            "seeds": list(seeds),
            "n_seeds": len(seeds),
            "n_folds": int(n_folds),
            "validation": "rolling-origin, expanding window, no shuffle",
        },
    }

    if loader_config is not None:
        metadata["site"] = {
            "latitude": loader_config.nasa_power.latitude,
            "longitude": loader_config.nasa_power.longitude,
            "site_name": loader_config.nasa_power.site_name,
            "country": loader_config.nasa_power.country,
            "crop": loader_config.nasa_power.crop,
            "mendeley_utc_offset_hours": (
                loader_config.mendeley_utc_offset_hours
            ),
        }

    if extra:
        metadata.update(extra)

    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False, default=str)
    logger.info("Wrote %s", path)
    return path


def export_all(
    output_dir: str | Path,
    *,
    summary: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    fold_table: pd.DataFrame,
    shap_manifest: Dict[str, Any],
    comparison: Optional[Dict[str, Any]],
    blocks: Dict[str, Sequence[str]],
    baseline_names: Sequence[str],
    main_model_names: Sequence[str],
    n_rows: int,
    n_features: int,
    positive_rate: float,
    seeds: Sequence[int],
    n_folds: int,
    dataset_start: str,
    dataset_end: str,
    loader_config: Any = None,
) -> Dict[str, Path]:
    """Write every artefact and return the paths.

    Returns:
        Mapping of artefact name to the path written.
    """
    output_dir = Path(output_dir)
    paths = {
        "results_summary": write_results_summary(summary, output_dir),
        "baselines": write_baselines(
            summary, output_dir,
            baseline_names=baseline_names,
            main_model_names=main_model_names,
            positive_rate=positive_rate,
        ),
        "ablation": write_ablation(ablation_summary, output_dir),
        "feature_importance": write_feature_importance(
            shap_manifest, output_dir, blocks=blocks,
        ),
        "folds": write_folds(fold_table, output_dir),
        "model_comparison": write_model_comparison(comparison, output_dir),
        "run_metadata": write_run_metadata(
            output_dir,
            n_rows=n_rows,
            n_features=n_features,
            positive_rate=positive_rate,
            seeds=seeds,
            n_folds=n_folds,
            dataset_start=dataset_start,
            dataset_end=dataset_end,
            loader_config=loader_config,
        ),
    }
    logger.info("Exported %d artefacts to %s/", len(paths), output_dir)
    return paths
