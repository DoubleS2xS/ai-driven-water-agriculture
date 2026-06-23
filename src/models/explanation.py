"""Phase 3 — Explainable AI: SHAP-based model interpretation.

This module provides publication-ready SHAP explanations for the
tree-based irrigation classifiers trained in
:mod:`src.models.irrigation_ml`.

Two visualisation types are supported:

1. **Summary plot** — a beeswarm showing global feature importance and
   directional effects across the test set.
2. **Waterfall plot** — a per-instance breakdown explaining a single
   edge-case decision (e.g. why the model predicted irrigation when
   soil moisture was above the threshold).

Both methods save high-DPI PNG files suitable for direct inclusion in a
Q1 journal manuscript.

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

import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")  # headless backend — must be before pyplot import
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


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
