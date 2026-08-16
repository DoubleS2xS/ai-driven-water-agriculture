"""Tests for Phase 3 — Predictive Modeling & Explainable AI.

The fixture generates a 200-row synthetic classification dataset where
low soil moisture strongly predicts ``irrigation_event = 1``, making the
task easily learnable for tree-based models and allowing deterministic
assertions on F1 > 0.8.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from src.models.irrigation_ml import IrrigationPredictor
from src.models.explanation import SHAPExplainer


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def classification_data() -> Tuple[
    pd.DataFrame, pd.Series, pd.DataFrame, pd.Series
]:
    """Generate synthetic train/test data for irrigation classification.

    Target rule: ``irrigation_event = 1`` when soil_moisture < 40 (with
    some noise), simulating a threshold-based irrigation controller.

    The feature distributions are tuned so that ~30-40 % of samples have
    soil_moisture < 40, producing enough positive examples for reliable
    gradient-boosted tree training on a 300-row dataset.

    Returns:
        Tuple of (X_train, y_train, X_test, y_test).
    """
    rng = np.random.default_rng(42)
    n = 300

    # Correlated environmental features with wider spread
    air_temp = (
        25.0
        + 12.0 * np.sin(2 * np.pi * np.arange(n) / 24)
        + rng.normal(0, 2.0, n)
    )
    solar_radiation = np.clip(
        400.0
        + 250.0 * np.sin(2 * np.pi * np.arange(n) / 24)
        + rng.normal(0, 60, n),
        0.0,
        1000.0,
    )
    humidity = 55.0 - 0.4 * air_temp + rng.normal(0, 4.0, n)
    soil_moisture = (
        45.0
        - 0.4 * air_temp
        + 0.15 * humidity
        - 0.003 * solar_radiation
        + rng.normal(0, 8.0, n)
    )
    soil_moisture = np.clip(soil_moisture, 5.0, 95.0)

    # Target: irrigate when soil moisture is low
    threshold = 40.0
    noise = rng.normal(0, 1.0, n)
    irrigation_event = ((soil_moisture + noise) < threshold).astype(int)

    features = pd.DataFrame({
        "soil_moisture": soil_moisture,
        "air_temp": air_temp,
        "humidity": humidity,
        "solar_radiation": solar_radiation,
    })
    target = pd.Series(irrigation_event, name="irrigation_event")

    # Stratified shuffle split to ensure balanced class distribution
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=42, stratify=target,
    )
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, y_train, X_test, y_test


# =====================================================================
# IrrigationPredictor tests
# =====================================================================


class TestIrrigationPredictor:
    """Validate training, prediction, and evaluation on synthetic data."""

    def test_predictor_training_and_evaluation_xgboost(
        self,
        classification_data: Tuple[
            pd.DataFrame, pd.Series, pd.DataFrame, pd.Series
        ],
    ) -> None:
        """XGBoost should achieve F1 > 0.8 on the synthetic dataset."""
        X_train, y_train, X_test, y_test = classification_data

        predictor = IrrigationPredictor(model_type="xgboost")
        predictor.train(X_train, y_train)
        metrics = predictor.evaluate(X_test, y_test)

        assert metrics["f1"] > 0.8, (
            f"XGBoost F1={metrics['f1']:.4f}, expected > 0.8"
        )
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_predictor_training_and_evaluation_lightgbm(
        self,
        classification_data: Tuple[
            pd.DataFrame, pd.Series, pd.DataFrame, pd.Series
        ],
    ) -> None:
        """LightGBM should achieve F1 > 0.8 on the synthetic dataset."""
        X_train, y_train, X_test, y_test = classification_data

        predictor = IrrigationPredictor(model_type="lightgbm")
        predictor.train(X_train, y_train)
        metrics = predictor.evaluate(X_test, y_test)

        assert metrics["f1"] > 0.8, (
            f"LightGBM F1={metrics['f1']:.4f}, expected > 0.8"
        )
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_predict_returns_binary(
        self,
        classification_data: Tuple[
            pd.DataFrame, pd.Series, pd.DataFrame, pd.Series
        ],
    ) -> None:
        """Predictions should only contain {0, 1}."""
        X_train, y_train, X_test, _ = classification_data

        predictor = IrrigationPredictor(model_type="xgboost")
        predictor.train(X_train, y_train)
        preds = predictor.predict(X_test)

        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_shape(
        self,
        classification_data: Tuple[
            pd.DataFrame, pd.Series, pd.DataFrame, pd.Series
        ],
    ) -> None:
        """predict_proba should return (n_samples, 2)."""
        X_train, y_train, X_test, _ = classification_data

        predictor = IrrigationPredictor(model_type="xgboost")
        predictor.train(X_train, y_train)
        proba = predictor.predict_proba(X_test)

        assert proba.shape == (len(X_test), 2)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)

    def test_untrained_model_raises(self) -> None:
        """Calling predict before train should raise RuntimeError."""
        predictor = IrrigationPredictor(model_type="xgboost")
        dummy = pd.DataFrame({"a": [1, 2]})

        with pytest.raises(RuntimeError, match="not been trained"):
            predictor.predict(dummy)

    def test_invalid_model_type_raises(self) -> None:
        """Invalid model_type should raise ValueError."""
        with pytest.raises(ValueError, match="model_type"):
            IrrigationPredictor(model_type="random_forest")

    def test_evaluate_returns_all_keys(
        self,
        classification_data: Tuple[
            pd.DataFrame, pd.Series, pd.DataFrame, pd.Series
        ],
    ) -> None:
        """evaluate() must return the full imbalanced-metric set.

        Widened from the original four keys when PR-AUC, the Brier score
        and the confusion matrix became mandatory for the paper: at a
        ~23 % positive rate, precision/recall/F1/ROC-AUC alone do not
        characterise the classifier.
        """
        from src.metrics import PRIMARY_METRICS

        X_train, y_train, X_test, y_test = classification_data

        predictor = IrrigationPredictor(model_type="xgboost")
        predictor.train(X_train, y_train)
        metrics = predictor.evaluate(X_test, y_test)

        assert set(PRIMARY_METRICS) <= set(metrics)
        assert {"tn", "fp", "fn", "tp"} <= set(metrics)
        assert {"n", "n_positive", "positive_rate"} <= set(metrics)


# =====================================================================
# SHAPExplainer tests
# =====================================================================


class TestSHAPExplainer:
    """Validate SHAP value extraction and plot generation."""

    def test_shap_explainer_generation(
        self,
        classification_data: Tuple[
            pd.DataFrame, pd.Series, pd.DataFrame, pd.Series
        ],
    ) -> None:
        """SHAP values should have shape (n_samples, n_features)."""
        X_train, y_train, X_test, _ = classification_data

        predictor = IrrigationPredictor(model_type="xgboost")
        predictor.train(X_train, y_train)

        explainer = SHAPExplainer(predictor.model)
        shap_values = explainer.get_shap_values(X_test)

        assert shap_values.shape == X_test.shape, (
            f"Expected shape {X_test.shape}, got {shap_values.shape}"
        )

    def test_shap_explainer_lightgbm(
        self,
        classification_data: Tuple[
            pd.DataFrame, pd.Series, pd.DataFrame, pd.Series
        ],
    ) -> None:
        """SHAP values should also work with LightGBM."""
        X_train, y_train, X_test, _ = classification_data

        predictor = IrrigationPredictor(model_type="lightgbm")
        predictor.train(X_train, y_train)

        explainer = SHAPExplainer(predictor.model)
        shap_values = explainer.get_shap_values(X_test)

        assert shap_values.shape == X_test.shape

    def test_summary_plot_saves_file(
        self,
        classification_data: Tuple[
            pd.DataFrame, pd.Series, pd.DataFrame, pd.Series
        ],
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """Summary plot should produce a non-empty PNG file."""
        X_train, y_train, X_test, _ = classification_data

        predictor = IrrigationPredictor(model_type="xgboost")
        predictor.train(X_train, y_train)

        explainer = SHAPExplainer(predictor.model)
        out_path = str(tmp_path / "shap_summary.png")
        explainer.plot_summary(X_test, save_path=out_path)

        from pathlib import Path
        assert Path(out_path).exists()
        assert Path(out_path).stat().st_size > 0

    def test_waterfall_plot_saves_file(
        self,
        classification_data: Tuple[
            pd.DataFrame, pd.Series, pd.DataFrame, pd.Series
        ],
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """Waterfall plot should produce a non-empty PNG file."""
        X_train, y_train, X_test, _ = classification_data

        predictor = IrrigationPredictor(model_type="xgboost")
        predictor.train(X_train, y_train)

        explainer = SHAPExplainer(predictor.model)
        out_path = str(tmp_path / "shap_waterfall.png")
        explainer.plot_local_decision(X_test, index=0, save_path=out_path)

        from pathlib import Path
        assert Path(out_path).exists()
        assert Path(out_path).stat().st_size > 0

    def test_waterfall_index_out_of_range(
        self,
        classification_data: Tuple[
            pd.DataFrame, pd.Series, pd.DataFrame, pd.Series
        ],
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """Out-of-range index should raise IndexError."""
        X_train, y_train, X_test, _ = classification_data

        predictor = IrrigationPredictor(model_type="xgboost")
        predictor.train(X_train, y_train)

        explainer = SHAPExplainer(predictor.model)
        with pytest.raises(IndexError, match="out of range"):
            explainer.plot_local_decision(
                X_test, index=9999,
                save_path=str(tmp_path / "fail.png"),
            )
