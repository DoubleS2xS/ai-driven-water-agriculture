"""Explainable AI: SHAP-based interpretation of the irrigation model.

Publication-ready SHAP explanations for the tree-based classifiers
trained in :mod:`src.models.irrigation_ml`.

Visualisations
--------------
1. **Beeswarm summary** — global feature importance and the direction of
   each feature's effect, over a fold's held-out block.
2. **Waterfall** — a per-instance breakdown of one decision.
3. **Dependence** — how a single feature's contribution varies with its
   value, for the three most important features.

Choosing which instances to explain
-----------------------------------
The earlier revision hard-coded ``index=0`` — whichever row happened to
sort first in the test block.  That row carries no particular meaning,
and a waterfall plot of it tells the reader nothing about how the model
behaves; worse, it invites a narrative built on an arbitrary example.

:func:`select_explanation_instances` replaces that with three instances
chosen for what each one demonstrates:

``confident_true_positive``
    The highest-probability correct alarm.  Shows which features the
    model relies on when it is right and sure.
``confident_false_positive``
    The highest-probability *wrong* alarm.  The most informative single
    plot in the set: a confident error reveals which feature combination
    misleads the model, which a correct prediction cannot.
``borderline``
    The instance whose probability sits closest to the 0.5 decision
    boundary, where the model is effectively undecided and small feature
    changes flip the outcome.

Their indices, timestamps and probabilities are written to JSON so the
manuscript can describe exactly which hours the figures depict, and so a
reviewer can reproduce them.

Design notes
------------
* ``matplotlib.use("Agg")`` is set at import time so that the module
  works in headless CI environments without a display server.
* SHAP values are extracted via ``shap.TreeExplainer``, which is exact
  for gradient-boosted tree ensembles (XGBoost, LightGBM).

References
----------
* Lundberg & Lee (2017): "A Unified Approach to Interpreting Model
  Predictions" — NeurIPS 2017.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # headless backend — must be before pyplot import
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD: float = 0.5

#: Number of features given a dependence plot.
N_DEPENDENCE_FEATURES: int = 3


@dataclass
class InstanceSelection:
    """One instance chosen for a waterfall plot.

    Attributes:
        case: Which of the three roles this instance fills.
        position: Row offset within the explained block, i.e. the index
            passed to :meth:`SHAPExplainer.plot_local_decision`.
        row_index: Index label in the original design matrix, so the hour
            can be located in the source data.
        y_true: Ground-truth label.
        y_pred: Hard prediction at the operating threshold.
        y_proba: Predicted probability of irrigation.
        timestamp: UTC timestamp of the hour, when available.
        rationale: Why this instance was chosen, for the figure caption.
    """

    case: str
    position: int
    row_index: int
    y_true: int
    y_pred: int
    y_proba: float
    timestamp: Optional[str]
    rationale: str


def select_explanation_instances(
    y_true: Sequence[int],
    y_proba: Sequence[float],
    *,
    timestamps: Optional[Sequence] = None,
    row_indices: Optional[Sequence[int]] = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, Optional[InstanceSelection]]:
    """Pick the three instances worth plotting, by what each demonstrates.

    Args:
        y_true: Ground-truth labels for the explained block.
        y_proba: Predicted probability of the positive class.
        timestamps: Optional UTC timestamps aligned to the block.
        row_indices: Optional original design-matrix indices.
        threshold: Operating threshold separating the classes.

    Returns:
        Mapping from case name to the chosen instance, or ``None`` for a
        case with no candidate — a fold with no false positives has no
        confident false positive to show, and inventing a substitute
        would misrepresent the figure.

    Raises:
        ValueError: If the inputs disagree in length or are empty.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)

    if len(y_true) != len(y_proba):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, y_proba={len(y_proba)}."
        )
    if len(y_true) == 0:
        raise ValueError("Cannot select instances from an empty block.")

    y_pred = (y_proba >= threshold).astype(int)

    def _build(position: int, case: str, rationale: str) -> InstanceSelection:
        return InstanceSelection(
            case=case,
            position=int(position),
            row_index=(
                int(row_indices[position]) if row_indices is not None
                else int(position)
            ),
            y_true=int(y_true[position]),
            y_pred=int(y_pred[position]),
            y_proba=float(y_proba[position]),
            timestamp=(
                str(pd.Timestamp(timestamps[position]))
                if timestamps is not None else None
            ),
            rationale=rationale,
        )

    selections: Dict[str, Optional[InstanceSelection]] = {}

    # 1 — Most confident correct alarm.
    true_positive = np.flatnonzero((y_true == 1) & (y_pred == 1))
    selections["confident_true_positive"] = (
        _build(
            true_positive[np.argmax(y_proba[true_positive])],
            "confident_true_positive",
            "Highest-probability correct irrigation alarm: shows which "
            "features drive the model when it is both right and certain.",
        )
        if true_positive.size else None
    )

    # 2 — Most confident wrong alarm.
    false_positive = np.flatnonzero((y_true == 0) & (y_pred == 1))
    selections["confident_false_positive"] = (
        _build(
            false_positive[np.argmax(y_proba[false_positive])],
            "confident_false_positive",
            "Highest-probability false alarm: the most informative single "
            "explanation, since a confident error exposes the feature "
            "combination that misleads the model.",
        )
        if false_positive.size else None
    )

    # 3 — Closest to the decision boundary.
    selections["borderline"] = _build(
        int(np.argmin(np.abs(y_proba - threshold))),
        "borderline",
        f"Probability closest to the {threshold} decision boundary: the "
        f"model is effectively undecided and small feature changes flip "
        f"the outcome.",
    )

    for case, selection in selections.items():
        if selection is None:
            logger.warning(
                "No candidate for '%s' in this block — the fold contains "
                "none. The corresponding figure will be skipped.", case,
            )
        else:
            logger.info(
                "Selected %s: position=%d, p=%.4f, y_true=%d",
                case, selection.position, selection.y_proba, selection.y_true,
            )

    return selections


class SHAPExplainer:
    """SHAP-based explainability for tree-based irrigation classifiers.

    Wraps ``shap.TreeExplainer`` and provides convenience methods for
    global (summary) and local (waterfall) explanations.

    Attributes:
        model: The trained tree-based estimator.
        explainer: The ``shap.TreeExplainer`` instance (created lazily
            on first call to :meth:`get_shap_values`).
    """

    def __init__(self, model: Any) -> None:
        """Initialise with a trained tree-based model.

        Args:
            model: A fitted estimator compatible with
                ``shap.TreeExplainer`` (e.g. ``XGBClassifier``,
                ``LGBMClassifier``).
        """
        self.model: Any = model
        self._explainer: Optional[shap.TreeExplainer] = None
        logger.info(
            "SHAPExplainer initialised for model type: %s",
            type(model).__name__,
        )

    # ── Private helpers ───────────────────────────────────────────────

    def _ensure_explainer(self) -> shap.TreeExplainer:
        """Lazily create the TreeExplainer on first use.

        Returns:
            The ``shap.TreeExplainer`` instance.
        """
        if self._explainer is None:
            logger.info("Building shap.TreeExplainer …")
            self._explainer = shap.TreeExplainer(self.model)
            logger.info("TreeExplainer ready.")
        return self._explainer

    # ── Public API ────────────────────────────────────────────────────

    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Extract SHAP values for the given feature matrix.

        For binary classifiers the returned array corresponds to the
        positive class (class 1 = irrigation ON).

        Args:
            X: Feature matrix (n_samples × n_features).

        Returns:
            2-D numpy array of shape ``(n_samples, n_features)``
            containing per-feature SHAP contributions.
        """
        explainer = self._ensure_explainer()
        logger.info("Computing SHAP values for %d samples …", len(X))

        shap_values_raw = explainer.shap_values(X)

        # shap_values_raw may be a list [class_0, class_1] for binary
        # classifiers, or a single array.  Normalise to class-1 values.
        if isinstance(shap_values_raw, list):
            values = np.asarray(shap_values_raw[1])
        else:
            values = np.asarray(shap_values_raw)

        # Handle 3-D output from some SHAP versions: (n, features, 2)
        if values.ndim == 3:
            values = values[:, :, 1]

        logger.info(
            "SHAP values computed: shape=%s", values.shape,
        )
        return values

    def plot_summary(
        self,
        X: pd.DataFrame,
        save_path: str,
        *,
        max_display: int = 10,
        dpi: int = 300,
    ) -> None:
        """Generate and save a SHAP beeswarm summary plot.

        Args:
            X: Feature matrix used to compute SHAP values.
            save_path: File path for the output PNG.
            max_display: Maximum number of features to show.
            dpi: Resolution for publication-quality output.
        """
        shap_values = self.get_shap_values(X)

        logger.info("Generating SHAP summary plot → %s", save_path)

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X,
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()

        logger.info("Summary plot saved: %s", save_path)

    def plot_local_decision(
        self,
        X: pd.DataFrame,
        index: int,
        save_path: str,
        *,
        dpi: int = 300,
    ) -> None:
        """Generate and save a SHAP waterfall plot for a single instance.

        This is useful for explaining edge-case decisions in the
        manuscript (e.g. "Why did the model predict irrigation ON when
        soil moisture was 52 %?").

        Args:
            X: Feature matrix (same data used for training / testing).
            index: Row index of the instance to explain.
            save_path: File path for the output PNG.
            dpi: Resolution for publication-quality output.

        Raises:
            IndexError: If *index* is out of bounds.
        """
        if index < 0 or index >= len(X):
            raise IndexError(
                f"index {index} out of range [0, {len(X) - 1}]."
            )

        explainer = self._ensure_explainer()

        logger.info(
            "Generating SHAP waterfall plot for index=%d → %s",
            index,
            save_path,
        )

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        # Use the Explanation object API for waterfall plots
        explanation = explainer(X)

        # For binary classifiers, explanation may have shape
        # (n_samples, n_features, 2).  Select class 1.
        if explanation.values.ndim == 3:
            single_explanation = shap.Explanation(
                values=explanation.values[index, :, 1],
                base_values=explanation.base_values[index, 1],
                data=explanation.data[index],
                feature_names=explanation.feature_names,
            )
        else:
            single_explanation = shap.Explanation(
                values=explanation.values[index],
                base_values=(
                    explanation.base_values[index]
                    if hasattr(explanation.base_values, "__getitem__")
                    else explanation.base_values
                ),
                data=explanation.data[index],
                feature_names=explanation.feature_names,
            )

        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(single_explanation, show=False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()

        logger.info("Waterfall plot saved: %s", save_path)

    # ── Global importance ─────────────────────────────────────────────

    def mean_abs_shap(self, X: pd.DataFrame) -> pd.Series:
        """Return mean |SHAP| per feature, descending.

        The standard global-importance summary: the average magnitude of
        each feature's contribution, in log-odds units.  Unlike a tree's
        built-in ``feature_importances_`` (split counts or gain), it is
        computed on the data actually being explained and is directly
        comparable across models.

        Args:
            X: Feature matrix to explain.

        Returns:
            Series indexed by feature name, sorted descending.
        """
        shap_values = self.get_shap_values(X)
        importance = pd.Series(
            np.abs(shap_values).mean(axis=0), index=X.columns,
        )
        return importance.sort_values(ascending=False)

    def top_features(
        self, X: pd.DataFrame, k: int = N_DEPENDENCE_FEATURES,
    ) -> List[str]:
        """Return the *k* features with the largest mean |SHAP|."""
        return self.mean_abs_shap(X).head(k).index.tolist()

    def plot_dependence(
        self,
        X: pd.DataFrame,
        feature: str,
        save_path: str,
        *,
        interaction_index: Optional[str] = "auto",
        dpi: int = 300,
    ) -> None:
        """Plot how one feature's SHAP contribution varies with its value.

        Answers the question a beeswarm only hints at: not merely *how
        much* a feature matters, but in which direction and over what
        range. For soil moisture this is where the model's effective
        threshold becomes visible.

        Args:
            X: Feature matrix.
            feature: Column to plot.
            save_path: Output PNG path.
            interaction_index: Feature to colour by; ``"auto"`` lets SHAP
                pick the strongest interaction, ``None`` disables it.
            dpi: Output resolution.

        Raises:
            KeyError: If *feature* is not a column of *X*.
        """
        if feature not in X.columns:
            raise KeyError(
                f"Feature '{feature}' not in the matrix; available: "
                f"{list(X.columns)[:10]}…"
            )

        shap_values = self.get_shap_values(X)

        logger.info("Generating SHAP dependence plot for '%s'", feature)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(8, 6))
        shap.dependence_plot(
            feature,
            shap_values,
            X,
            interaction_index=interaction_index,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()

        logger.info("Dependence plot saved: %s", save_path)


# ======================================================================
# Orchestration
# ======================================================================

def explain_model(
    model: Any,
    X: pd.DataFrame,
    y_true: Sequence[int],
    y_proba: Sequence[float],
    output_dir: str | Path,
    *,
    timestamps: Optional[Sequence] = None,
    row_indices: Optional[Sequence[int]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    threshold: float = DEFAULT_THRESHOLD,
    dpi: int = 300,
) -> Dict[str, Any]:
    """Produce the full explanation set and record what it depicts.

    Writes the beeswarm, one waterfall per selected instance, dependence
    plots for the top three features, and a JSON manifest naming every
    instance and its probability.  Nothing about the figures is left
    implicit: a reader of the manuscript can tell exactly which hour each
    waterfall shows.

    Args:
        model: Fitted tree estimator (the bare model, not a pipeline).
        X: Feature matrix of the block being explained.
        y_true: Ground-truth labels for that block.
        y_proba: Predicted probabilities for that block.
        output_dir: Directory for the PNG and JSON outputs.
        timestamps: Optional UTC timestamps aligned to *X*.
        row_indices: Optional original design-matrix indices.
        metadata: Extra fields recorded in the manifest (model name,
            fold number, fold period).
        threshold: Operating threshold.
        dpi: Output resolution.

    Returns:
        The manifest that was written to ``shap_instances.json``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    explainer = SHAPExplainer(model)

    # ── Global view ───────────────────────────────────────────────────
    explainer.plot_summary(
        X, save_path=str(output_dir / "shap_summary_beeswarm.png"), dpi=dpi,
    )

    importance = explainer.mean_abs_shap(X)
    top = importance.head(N_DEPENDENCE_FEATURES).index.tolist()

    # ── Dependence plots ──────────────────────────────────────────────
    dependence_files = {}
    for rank, feature in enumerate(top, start=1):
        path = output_dir / f"shap_dependence_{rank}_{feature}.png"
        explainer.plot_dependence(X, feature, str(path), dpi=dpi)
        dependence_files[feature] = path.name

    # ── Instance-level views ──────────────────────────────────────────
    selections = select_explanation_instances(
        y_true, y_proba,
        timestamps=timestamps,
        row_indices=row_indices,
        threshold=threshold,
    )

    waterfall_files: Dict[str, Optional[str]] = {}
    for case, selection in selections.items():
        if selection is None:
            waterfall_files[case] = None
            continue
        path = output_dir / f"shap_waterfall_{case}.png"
        explainer.plot_local_decision(
            X, index=selection.position, save_path=str(path), dpi=dpi,
        )
        waterfall_files[case] = path.name

    # ── Manifest ──────────────────────────────────────────────────────
    manifest: Dict[str, Any] = {
        **(metadata or {}),
        "n_explained_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "threshold": threshold,
        "instances": {
            case: (asdict(sel) if sel is not None else None)
            for case, sel in selections.items()
        },
        "top_features": top,
        "mean_abs_shap": {
            name: float(value) for name, value in importance.items()
        },
        "figures": {
            "beeswarm": "shap_summary_beeswarm.png",
            "dependence": dependence_files,
            "waterfall": waterfall_files,
        },
    }

    manifest_path = output_dir / "shap_instances.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False, default=str)
    logger.info("SHAP manifest written → %s", manifest_path)

    return manifest
