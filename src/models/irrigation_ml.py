"""Phase 3 — Predictive Modeling: binary irrigation-event classification.

This module wraps **XGBoost** and **LightGBM** gradient-boosted tree
classifiers behind a unified ``IrrigationPredictor`` interface so that
the pipeline can swap backends via a single constructor argument.

The target variable is ``irrigation_event ∈ {0, 1}`` (electrovalve
relay status).  Feature columns are the healed environmental telemetry
produced by Phase 2 (soil moisture, air temperature, humidity, wind
speed, solar radiation, flow rate).

Design notes
------------
* Default hyper-parameters are chosen for reproducibility
  (``random_state=42``) and modest complexity (``max_depth=6``,
  ``n_estimators=200``).  Full HPO is left to the experiment script.
* ``evaluate()`` reports Precision, Recall, F1, and ROC-AUC — the
  metrics most relevant for imbalanced irrigation datasets (Review
  §2.3.1).
* The trained estimator is exposed via ``.model`` for downstream SHAP
  explainability (see :mod:`src.models.explanation`).

References
----------
* Review §2.3.1, source [2]: XGBoost / LightGBM benchmarks on
  IoT-scale agricultural telemetry.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

# Type alias for the underlying estimator
_Estimator = Any


class IrrigationPredictor:
    """Binary classifier for irrigation-event prediction.

    Wraps XGBoost or LightGBM behind a common API.

    Attributes:
        model_type: ``"xgboost"`` or ``"lightgbm"``.
        model: The fitted scikit-learn-compatible estimator (available
            after calling :meth:`train`).
    """

    _VALID_TYPES = ("xgboost", "lightgbm")

    def __init__(
        self,
        model_type: str = "xgboost",
        random_state: int = 42,
        **model_kwargs: Any,
    ) -> None:
        """Initialise the predictor.

        Args:
            model_type: ``"xgboost"`` or ``"lightgbm"``.
            random_state: Seed for deterministic training.
            **model_kwargs: Extra keyword arguments forwarded to the
                underlying estimator constructor (e.g. ``max_depth``,
                ``n_estimators``).

        Raises:
            ValueError: If *model_type* is not one of the supported
                backends.
        """
        if model_type not in self._VALID_TYPES:
            raise ValueError(
                f"model_type must be one of {self._VALID_TYPES}, "
                f"got '{model_type}'."
            )

        self.model_type: str = model_type
        self._random_state: int = random_state
        self._model_kwargs: Dict[str, Any] = model_kwargs
        self.model: Optional[_Estimator] = None

        logger.info(
            "IrrigationPredictor initialised: type='%s', seed=%d, "
            "extra_kwargs=%s",
            model_type,
            random_state,
            model_kwargs or "{}",
        )

    # ── Private helpers ───────────────────────────────────────────────

    def _build_estimator(self) -> _Estimator:
        """Construct the underlying estimator.

        Returns:
            A freshly instantiated XGBClassifier or LGBMClassifier.
        """
        defaults: Dict[str, Any] = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": self._random_state,
        }

        if self.model_type == "xgboost":
            from xgboost import XGBClassifier

            defaults.update({
                "use_label_encoder": False,
                "eval_metric": "logloss",
                "verbosity": 0,
            })
            defaults.update(self._model_kwargs)
            return XGBClassifier(**defaults)

        else:  # lightgbm
            from lightgbm import LGBMClassifier

            defaults.update({
                "verbose": -1,
            })
            defaults.update(self._model_kwargs)
            return LGBMClassifier(**defaults)

    # ── Public API ────────────────────────────────────────────────────

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **kwargs: Any,
    ) -> None:
        """Fit the model on training data.

        Args:
            X_train: Feature matrix (n_samples × n_features).
            y_train: Binary target vector ``{0, 1}``.
            **kwargs: Extra arguments forwarded to the estimator's
                ``.fit()`` method (e.g. ``sample_weight``).
        """
        self.model = self._build_estimator()

        logger.info(
            "Training %s on %d samples × %d features …",
            self.model_type,
            len(X_train),
            X_train.shape[1],
        )
        self.model.fit(X_train, y_train, **kwargs)
        logger.info("Training complete.")

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Return binary predictions.

        Args:
            X_test: Feature matrix for inference.

        Returns:
            1-D array of ``{0, 1}`` predictions.

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call train() first.")
        return np.asarray(self.model.predict(X_test))

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """Return class probabilities.

        Args:
            X_test: Feature matrix for inference.

        Returns:
            2-D array of shape ``(n_samples, 2)`` with columns
            ``[P(class=0), P(class=1)]``.

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call train() first.")
        return np.asarray(self.model.predict_proba(X_test))

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, float]:
        """Evaluate the model and return classification metrics.

        Metrics are computed with ``zero_division=0`` so that
        degenerate class distributions do not raise.

        Args:
            X_test: Feature matrix.
            y_test: Ground-truth binary labels.

        Returns:
            Dictionary with keys ``"precision"``, ``"recall"``,
            ``"f1"``, and ``"roc_auc"``.
        """
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]

        precision = float(precision_score(
            y_test, y_pred, zero_division=0,
        ))
        recall = float(recall_score(
            y_test, y_pred, zero_division=0,
        ))
        f1 = float(f1_score(
            y_test, y_pred, zero_division=0,
        ))

        # ROC-AUC requires both classes present
        try:
            roc_auc = float(roc_auc_score(y_test, y_proba))
        except ValueError:
            roc_auc = float("nan")
            logger.warning(
                "ROC-AUC undefined (only one class present in y_test)."
            )

        metrics: Dict[str, float] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
        }

        logger.info(
            "Evaluation (%s): Precision=%.4f, Recall=%.4f, "
            "F1=%.4f, ROC-AUC=%.4f",
            self.model_type,
            precision,
            recall,
            f1,
            roc_auc,
        )
        return metrics
