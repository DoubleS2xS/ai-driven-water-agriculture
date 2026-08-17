"""Tests for src.tuning — nested cross-validation.

Nested CV exists to make hyperparameter selection honest, so the decisive
test is the leakage test: corrupt the outer test block and confirm the
selected parameters do not move. If selection could see the outer test,
tuning would flatter the model and the whole exercise would be worse than
not tuning at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.tuning import (
    DEFAULT_INNER_FOLDS,
    PARAM_GRID,
    _grid_points,
    _translate_params,
    nested_cv_evaluate,
    select_hyperparameters,
    summarise_nested_cv,
)
from src.validation import rolling_origin_splits


# ── Fixtures ──────────────────────────────────────────────────────────


def _data(n: int = 600, seed: int = 0):
    """Autocorrelated, learnable, imbalanced — like the real target."""
    rng = np.random.default_rng(seed)
    hours = np.arange(n)
    moisture = 55.0 + 20.0 * np.sin(hours / 23.0) + rng.normal(0, 1.0, n)
    X = pd.DataFrame({
        "soil_moisture_lag1h": moisture,
        "soil_moisture_lag2h": np.roll(moisture, 1),
        "soil_moisture_roll6h_mean": pd.Series(moisture).rolling(
            6, min_periods=1
        ).mean().to_numpy(),
        "air_temp_lag1h": rng.normal(22, 5, n),
    })
    y = pd.Series((moisture < 42).astype(int))
    return X, y


@pytest.fixture(scope="module")
def data():
    return _data()


# ── Grid handling ────────────────────────────────────────────────────


class TestGrid:
    def test_expands_to_the_cartesian_product(self) -> None:
        points = _grid_points({"a": (1, 2), "b": (3, 4, 5)})
        assert len(points) == 6
        assert {"a": 1, "b": 3} in points

    def test_default_grid_is_small_on_purpose(self) -> None:
        """A large grid searched against single-digit positives selects noise."""
        assert len(_grid_points(PARAM_GRID)) <= 16

    def test_grid_covers_capacity_and_regularisation(self) -> None:
        assert "max_depth" in PARAM_GRID
        assert "n_estimators" in PARAM_GRID
        assert "min_child_weight" in PARAM_GRID


class TestParamTranslation:
    def test_xgboost_names_pass_through(self) -> None:
        params = {"max_depth": 4, "min_child_weight": 5}
        assert _translate_params("xgboost", params) == params

    def test_lightgbm_gets_its_own_name(self) -> None:
        """Passing XGBoost's name to LightGBM would silently do nothing.

        That would leave one axis of the grid untuned for one model only,
        making the two backends incomparable.
        """
        translated = _translate_params("lightgbm", {"min_child_weight": 5})
        assert translated == {"min_child_samples": 5}
        assert "min_child_weight" not in translated

    def test_translation_does_not_mutate_the_input(self) -> None:
        params = {"min_child_weight": 5}
        _translate_params("lightgbm", params)
        assert params == {"min_child_weight": 5}


# ── Selection ────────────────────────────────────────────────────────


class TestSelectHyperparameters:
    def test_returns_a_grid_point(self, data) -> None:
        X, y = data
        params, score, n_scored = select_hyperparameters(
            X.iloc[:400], y.iloc[:400], "xgboost",
        )
        assert set(params) == set(PARAM_GRID)
        assert n_scored > 0
        assert np.isfinite(score)

    def test_selected_values_come_from_the_grid(self, data) -> None:
        X, y = data
        params, _score, _n = select_hyperparameters(
            X.iloc[:400], y.iloc[:400], "xgboost",
        )
        for key, value in params.items():
            assert value in PARAM_GRID[key]

    def test_too_few_rows_falls_back_to_defaults(self, data) -> None:
        """Better library defaults than a choice made on nothing."""
        X, y = data
        params, score, n_scored = select_hyperparameters(
            X.iloc[:3], y.iloc[:3], "xgboost",
        )
        assert params == {}
        assert n_scored == 0
        assert np.isnan(score)

    def test_no_positives_falls_back_to_defaults(self) -> None:
        X, _y = _data(n=300)
        y = pd.Series(np.zeros(300, dtype=int))
        params, _score, n_scored = select_hyperparameters(
            X, y, "xgboost",
        )
        assert params == {}
        assert n_scored == 0

    def test_is_reproducible(self, data) -> None:
        X, y = data
        a, _s1, _n1 = select_hyperparameters(
            X.iloc[:400], y.iloc[:400], "xgboost", seed=7,
        )
        b, _s2, _n2 = select_hyperparameters(
            X.iloc[:400], y.iloc[:400], "xgboost", seed=7,
        )
        assert a == b


class TestNoOuterTestLeakage:
    """Selection must depend on the training fold alone."""

    def test_corrupting_the_outer_test_block_changes_nothing(
        self, data,
    ) -> None:
        """The decisive test for nested CV.

        Multiply the outer test rows by 1000 and flip their labels. If
        selection could see them — or if the inner splits were built over
        the whole matrix instead of the training block — the chosen
        parameters would move.
        """
        X, y = data
        outer_splits = rolling_origin_splits(len(X))
        train_idx, test_idx = outer_splits[-1]

        clean = nested_cv_evaluate(
            X, y, [outer_splits[-1]], model_names=("xgboost",),
        )

        X_corrupt = X.copy()
        y_corrupt = y.copy()
        X_corrupt.iloc[test_idx] *= 1000.0
        y_corrupt.iloc[test_idx] = 1 - y_corrupt.iloc[test_idx]

        corrupted = nested_cv_evaluate(
            X_corrupt, y_corrupt, [outer_splits[-1]],
            model_names=("xgboost",),
        )

        assert (
            clean.iloc[0]["selected_params"]
            == corrupted.iloc[0]["selected_params"]
        )
        assert clean.iloc[0]["inner_score"] == pytest.approx(
            corrupted.iloc[0]["inner_score"]
        )

    def test_inner_folds_stay_inside_the_training_block(self, data) -> None:
        """An inner split may not reach past the training rows."""
        X, y = data
        outer_splits = rolling_origin_splits(len(X))
        train_idx, _test_idx = outer_splits[2]

        inner = rolling_origin_splits(
            len(train_idx),
        )
        for inner_train, inner_test in inner:
            assert inner_train.max() < len(train_idx)
            assert inner_test.max() < len(train_idx)


# ── Nested evaluation ────────────────────────────────────────────────


class TestNestedCVEvaluate:
    @pytest.fixture(scope="class")
    def results(self, data):
        X, y = data
        return nested_cv_evaluate(
            X, y, rolling_origin_splits(len(X)),
            model_names=("xgboost",), inner_folds=DEFAULT_INNER_FOLDS,
        )

    def test_row_per_outer_fold(self, results) -> None:
        assert len(results) == 5
        assert list(results["fold"]) == [1, 2, 3, 4, 5]

    def test_records_selected_params_per_fold(self, results) -> None:
        assert results["selected_params"].notna().all()
        assert (results["selected_params"].str.len() > 0).all()

    def test_records_training_positives(self, results) -> None:
        """Needed to read why an early fold selected a tiny model."""
        assert "n_train_positives" in results.columns
        assert results["n_train_positives"].is_monotonic_increasing

    def test_reports_outer_metrics(self, results) -> None:
        for metric in ("roc_auc", "pr_auc", "f1", "brier"):
            assert metric in results.columns

    def test_training_set_grows_across_folds(self, results) -> None:
        assert results["n_train"].is_monotonic_increasing

    def test_summary_carries_intervals_and_params(self, results) -> None:
        summary = summarise_nested_cv(results)
        assert len(summary) == 1
        row = summary.iloc[0]
        assert np.isfinite(row["pr_auc_mean"])
        assert "f1:" in row["params_per_fold"]

    def test_two_models_produce_two_summary_rows(self, data) -> None:
        X, y = data
        results = nested_cv_evaluate(
            X, y, rolling_origin_splits(len(X)),
            model_names=("xgboost", "lightgbm"),
        )
        assert len(summarise_nested_cv(results)) == 2
