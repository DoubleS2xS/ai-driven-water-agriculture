"""Phase 3 — Predictive Modeling: binary irrigation-event classification.

This module wraps **XGBoost** and **LightGBM** gradient-boosted tree
classifiers behind a unified ``IrrigationPredictor`` interface so that
the pipeline can swap backends via a single constructor argument.

The target variable is ``irrigation_event(t) ∈ {0, 1}`` (electrovalve
relay status).  Feature columns are the **causal lag features** built by
:mod:`src.features` — soil-moisture lags and rolling statistics,
meteorological lags and daily aggregates, calendar encodings, and
autoregressive lags of the target.

Two columns are deliberately absent from that set: ``soil_moisture(t)``,
because moisture rises as a consequence of the valve opening, and the
flow-meter channels, because they measure the water the valve delivered.
Both are enforced upstream by :func:`src.features.assert_no_forbidden_features`.

Design notes
------------
* Default hyper-parameters are chosen for reproducibility
  (``random_state=42``) and modest complexity (``max_depth=6``,
  ``n_estimators=200``).  Full HPO is left to the experiment script.
* ``evaluate()`` reports the full imbalanced-classification metric set
  via :mod:`src.metrics`, including PR-AUC, which is the metric of
  record at a ~23 % positive rate.
* Fitting goes through a :class:`sklearn.pipeline.Pipeline` built by
  :func:`src.preprocessing.build_model_pipeline`, so any preprocessing
  is fitted on training rows only.  There is no code path that fits a
  transformer on the test block.
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
from sklearn.pipeline import Pipeline

from src.metrics import compute_classification_metrics
from src.preprocessing import build_model_pipeline

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
        self.pipeline: Optional[Pipeline] = None

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

        Fitting goes through a pipeline, so any preprocessing is fitted
        on *X_train* alone.  Calling ``train`` again rebuilds the
        pipeline from scratch rather than warm-starting, which keeps
        every fold of a cross-validation independent of the last.

        Args:
            X_train: Feature matrix (n_samples × n_features).
            y_train: Binary target vector ``{0, 1}``.
            **kwargs: Extra arguments forwarded to the estimator's
                ``.fit()`` method (e.g. ``sample_weight``).
        """
        estimator = self._build_estimator()
        self.pipeline = build_model_pipeline(self.model_type, estimator)

        logger.info(
            "Training %s on %d samples × %d features …",
            self.model_type,
            len(X_train),
            X_train.shape[1],
        )
        fit_kwargs = {f"clf__{k}": v for k, v in kwargs.items()}
        self.pipeline.fit(X_train, y_train, **fit_kwargs)
        # Expose the bare estimator for SHAP, which needs the tree model
        # itself rather than the surrounding pipeline.
        self.model = self.pipeline.named_steps["clf"]
        logger.info("Training complete.")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **kwargs: Any,
    ) -> "IrrigationPredictor":
        """Scikit-learn-style alias for :meth:`train`, returning ``self``.

        Lets the main models and the baselines in :mod:`src.baselines`
        share one fold loop, one metric function and one statistical
        test, instead of maintaining parallel evaluation paths that could
        drift apart.
        """
        self.train(X_train, y_train, **kwargs)
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Return binary predictions.

        Args:
            X_test: Feature matrix for inference.

        Returns:
            1-D array of ``{0, 1}`` predictions.

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self.pipeline is None:
            raise RuntimeError("Model has not been trained. Call train() first.")
        return np.asarray(self.pipeline.predict(X_test))

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
        if self.pipeline is None:
            raise RuntimeError("Model has not been trained. Call train() first.")
        return np.asarray(self.pipeline.predict_proba(X_test))

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, float]:
        """Evaluate the model and return the full classification metrics.

        Delegates to :func:`src.metrics.compute_classification_metrics`,
        the single definition used for the main models, the baselines and
        the ablation study alike.

        Args:
            X_test: Feature matrix.
            y_test: Ground-truth binary labels.

        Returns:
            Metric mapping including ROC-AUC, PR-AUC, F1, precision,
            recall, Brier score, the confusion-matrix cells and the
            positive rate needed to interpret PR-AUC.
        """
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]

        metrics = compute_classification_metrics(y_test, y_pred, y_proba)

        logger.info(
            "Evaluation (%s): ROC-AUC=%.4f, PR-AUC=%.4f (prevalence %.4f), "
            "F1=%.4f, Precision=%.4f, Recall=%.4f, Brier=%.4f",
            self.model_type,
            metrics["roc_auc"],
            metrics["pr_auc"],
            metrics["positive_rate"],
            metrics["f1"],
            metrics["precision"],
            metrics["recall"],
            metrics["brier"],
        )
        return metrics
