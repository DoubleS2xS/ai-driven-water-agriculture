"""Tests for src.models.explanation — instance selection and outputs.

The instance selection is the part that replaced a hard-coded
``index=0``, so it is tested against hand-built cases where the correct
choice is known by construction. A selector that quietly picked the wrong
row would produce figures that illustrate a claim the data does not
support.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.explanation import (
    N_DEPENDENCE_FEATURES,
    InstanceSelection,
    SHAPExplainer,
    explain_model,
    select_explanation_instances,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def fitted_model_and_data():
    """A small fitted XGBoost model with a separable signal."""
    from src.models.irrigation_ml import IrrigationPredictor

    rng = np.random.default_rng(0)
    n = 300
    moisture = rng.uniform(20, 80, n)
    X = pd.DataFrame({
        "soil_moisture_lag1h": moisture,
        "soil_moisture_roll6h_mean": moisture + rng.normal(0, 2, n),
        "air_temp_lag1h": rng.normal(20, 5, n),
        "hour_sin": np.sin(np.arange(n) / 24 * 2 * np.pi),
    })
    y = pd.Series((moisture < 45).astype(int))

    predictor = IrrigationPredictor(model_type="xgboost")
    predictor.train(X, y)
    proba = predictor.predict_proba(X)[:, 1]
    return predictor.model, X, y, proba


# ── Instance selection ───────────────────────────────────────────────


class TestSelectExplanationInstances:
    def test_picks_highest_probability_true_positive(self) -> None:
        y_true = np.array([1, 1, 1, 0])
        y_proba = np.array([0.60, 0.95, 0.75, 0.10])
        result = select_explanation_instances(y_true, y_proba)
        assert result["confident_true_positive"].position == 1
        assert result["confident_true_positive"].y_proba == pytest.approx(0.95)

    def test_picks_highest_probability_false_positive(self) -> None:
        """The most confident *wrong* alarm, not merely any error."""
        y_true = np.array([0, 0, 1, 0])
        y_proba = np.array([0.55, 0.99, 0.90, 0.20])
        result = select_explanation_instances(y_true, y_proba)
        selected = result["confident_false_positive"]
        assert selected.position == 1
        assert selected.y_true == 0
        assert selected.y_pred == 1

    def test_picks_probability_nearest_the_boundary(self) -> None:
        y_true = np.array([1, 0, 1, 0])
        y_proba = np.array([0.95, 0.51, 0.05, 0.80])
        result = select_explanation_instances(y_true, y_proba)
        assert result["borderline"].position == 1

    def test_borderline_may_sit_just_below_the_threshold(self) -> None:
        y_true = np.array([1, 0, 1])
        y_proba = np.array([0.95, 0.49, 0.05])
        result = select_explanation_instances(y_true, y_proba)
        assert result["borderline"].position == 1
        assert result["borderline"].y_pred == 0

    def test_never_returns_index_zero_by_default(self) -> None:
        """The defect this function exists to fix."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_proba = np.array([0.01, 0.99, 0.02, 0.88, 0.52])
        result = select_explanation_instances(y_true, y_proba)
        chosen = {s.position for s in result.values() if s is not None}
        assert 0 not in chosen

    def test_missing_case_returns_none_not_a_substitute(self) -> None:
        """A fold with no false positives must not fake one."""
        y_true = np.array([1, 1, 0])
        y_proba = np.array([0.90, 0.80, 0.10])
        result = select_explanation_instances(y_true, y_proba)
        assert result["confident_false_positive"] is None
        assert result["confident_true_positive"] is not None

    def test_all_three_cases_are_reported(self) -> None:
        y_true = np.array([1, 0, 1, 0])
        y_proba = np.array([0.99, 0.97, 0.10, 0.50])
        result = select_explanation_instances(y_true, y_proba)
        assert set(result) == {
            "confident_true_positive",
            "confident_false_positive",
            "borderline",
        }

    def test_row_indices_are_carried_through(self) -> None:
        y_true = np.array([1, 1])
        y_proba = np.array([0.7, 0.95])
        result = select_explanation_instances(
            y_true, y_proba, row_indices=[900, 901],
        )
        selected = result["confident_true_positive"]
        assert selected.position == 1
        assert selected.row_index == 901

    def test_timestamps_are_recorded(self) -> None:
        y_true = np.array([1, 1])
        y_proba = np.array([0.7, 0.95])
        ts = pd.date_range("2022-08-27", periods=2, freq="1h")
        result = select_explanation_instances(y_true, y_proba, timestamps=ts)
        assert "2022-08-27 01:00:00" in (
            result["confident_true_positive"].timestamp
        )

    def test_custom_threshold_changes_classification(self) -> None:
        y_true = np.array([0, 0])
        y_proba = np.array([0.60, 0.55])
        strict = select_explanation_instances(y_true, y_proba, threshold=0.9)
        loose = select_explanation_instances(y_true, y_proba, threshold=0.5)
        assert strict["confident_false_positive"] is None
        assert loose["confident_false_positive"] is not None

    def test_every_selection_carries_a_rationale(self) -> None:
        y_true = np.array([1, 0, 1, 0])
        y_proba = np.array([0.99, 0.97, 0.10, 0.50])
        result = select_explanation_instances(y_true, y_proba)
        for selection in result.values():
            if selection is not None:
                assert len(selection.rationale) > 20

    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            select_explanation_instances([0, 1], [0.5])

    def test_empty_input_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty block"):
            select_explanation_instances([], [])


# ── Global importance ────────────────────────────────────────────────


class TestMeanAbsShap:
    def test_sorted_descending(self, fitted_model_and_data) -> None:
        model, X, _, _ = fitted_model_and_data
        importance = SHAPExplainer(model).mean_abs_shap(X)
        assert list(importance) == sorted(importance, reverse=True)

    def test_covers_every_feature(self, fitted_model_and_data) -> None:
        model, X, _, _ = fitted_model_and_data
        importance = SHAPExplainer(model).mean_abs_shap(X)
        assert set(importance.index) == set(X.columns)

    def test_values_are_non_negative(self, fitted_model_and_data) -> None:
        model, X, _, _ = fitted_model_and_data
        importance = SHAPExplainer(model).mean_abs_shap(X)
        assert (importance >= 0).all()

    def test_identifies_the_driving_feature(
        self, fitted_model_and_data,
    ) -> None:
        """The target is generated from moisture, so moisture must lead."""
        model, X, _, _ = fitted_model_and_data
        top = SHAPExplainer(model).top_features(X, k=2)
        assert any("soil_moisture" in name for name in top)

    def test_top_features_respects_k(self, fitted_model_and_data) -> None:
        model, X, _, _ = fitted_model_and_data
        assert len(SHAPExplainer(model).top_features(X, k=2)) == 2


class TestDependencePlot:
    def test_saves_a_file(self, fitted_model_and_data, tmp_path) -> None:
        model, X, _, _ = fitted_model_and_data
        out = tmp_path / "dep.png"
        SHAPExplainer(model).plot_dependence(
            X, "soil_moisture_lag1h", str(out), interaction_index=None,
        )
        assert out.exists() and out.stat().st_size > 0

    def test_unknown_feature_rejected(
        self, fitted_model_and_data, tmp_path,
    ) -> None:
        model, X, _, _ = fitted_model_and_data
        with pytest.raises(KeyError, match="not in the matrix"):
            SHAPExplainer(model).plot_dependence(
                X, "no_such_feature", str(tmp_path / "x.png"),
            )


# ── Orchestration ────────────────────────────────────────────────────


class TestExplainModel:
    @pytest.fixture(scope="class")
    def manifest_and_dir(self, fitted_model_and_data, tmp_path_factory):
        model, X, y, proba = fitted_model_and_data
        out = tmp_path_factory.mktemp("shap")
        ts = pd.date_range("2022-08-27", periods=len(X), freq="1h")
        manifest = explain_model(
            model, X, y.to_numpy(), proba, out,
            timestamps=ts,
            row_indices=np.arange(len(X)) + 500,
            metadata={"model": "xgboost", "fold": 3},
        )
        return manifest, out

    def test_writes_the_json_manifest(self, manifest_and_dir) -> None:
        _, out = manifest_and_dir
        path = out / "shap_instances.json"
        assert path.exists()
        json.loads(path.read_text(encoding="utf-8"))

    def test_manifest_records_probabilities_and_indices(
        self, manifest_and_dir,
    ) -> None:
        """The paper must be able to say which hour each figure shows."""
        manifest, _ = manifest_and_dir
        for case, inst in manifest["instances"].items():
            if inst is None:
                continue
            assert 0.0 <= inst["y_proba"] <= 1.0
            assert inst["row_index"] >= 500
            assert inst["timestamp"] is not None

    def test_saves_the_beeswarm(self, manifest_and_dir) -> None:
        _, out = manifest_and_dir
        assert (out / "shap_summary_beeswarm.png").stat().st_size > 0

    def test_saves_one_dependence_plot_per_top_feature(
        self, manifest_and_dir,
    ) -> None:
        manifest, out = manifest_and_dir
        files = list(out.glob("shap_dependence_*.png"))
        assert len(files) == N_DEPENDENCE_FEATURES
        assert len(manifest["figures"]["dependence"]) == N_DEPENDENCE_FEATURES

    def test_saves_a_waterfall_per_available_case(
        self, manifest_and_dir,
    ) -> None:
        manifest, out = manifest_and_dir
        for case, filename in manifest["figures"]["waterfall"].items():
            if filename is None:
                assert manifest["instances"][case] is None
            else:
                assert (out / filename).stat().st_size > 0

    def test_manifest_carries_supplied_metadata(
        self, manifest_and_dir,
    ) -> None:
        manifest, _ = manifest_and_dir
        assert manifest["model"] == "xgboost"
        assert manifest["fold"] == 3

    def test_mean_abs_shap_is_serialised(self, manifest_and_dir) -> None:
        manifest, _ = manifest_and_dir
        assert len(manifest["mean_abs_shap"]) == manifest["n_features"]
        assert all(
            isinstance(v, float) for v in manifest["mean_abs_shap"].values()
        )


# ── Fold selection ───────────────────────────────────────────────────


class TestSelectFoldForExplanation:
    @staticmethod
    def _results() -> pd.DataFrame:
        return pd.DataFrame({
            "model": ["xgboost"] * 5,
            "fold": [1, 2, 3, 4, 5],
            "pr_auc": [0.07, 0.19, 0.81, 0.99, 0.91],
        })

    @staticmethod
    def _fold_table() -> pd.DataFrame:
        return pd.DataFrame({
            "fold": [1, 2, 3, 4, 5],
            "test_positive_rate": [0.03, 0.11, 0.18, 0.80, 0.27],
        })

    def test_best_strategy_picks_top_scoring_fold(self) -> None:
        from src.evaluate_pipeline import select_fold_for_explanation

        fold = select_fold_for_explanation(
            self._results(), self._fold_table(),
            model_name="xgboost", strategy="best",
        )
        assert fold == 4

    def test_last_strategy_picks_final_fold(self) -> None:
        from src.evaluate_pipeline import select_fold_for_explanation

        fold = select_fold_for_explanation(
            self._results(), self._fold_table(),
            model_name="xgboost", strategy="last",
        )
        assert fold == 5

    def test_extreme_balance_is_warned_about(self, caplog) -> None:
        """Explaining an 80 %-positive fold deserves a caveat."""
        from src.evaluate_pipeline import select_fold_for_explanation

        with caplog.at_level("WARNING"):
            select_fold_for_explanation(
                self._results(), self._fold_table(),
                model_name="xgboost", strategy="best",
            )
        assert "extreme regime" in caplog.text

    def test_unknown_strategy_rejected(self) -> None:
        from src.evaluate_pipeline import select_fold_for_explanation

        with pytest.raises(ValueError, match="strategy must be"):
            select_fold_for_explanation(
                self._results(), self._fold_table(),
                model_name="xgboost", strategy="median",
            )

    def test_unknown_model_rejected(self) -> None:
        from src.evaluate_pipeline import select_fold_for_explanation

        with pytest.raises(ValueError, match="No results for model"):
            select_fold_for_explanation(
                self._results(), self._fold_table(), model_name="catboost",
            )


class TestGenerateShapExplanationsGuards:
    def test_non_tree_model_rejected(self) -> None:
        """TreeExplainer is exact for trees and wrong for anything else."""
        from src.evaluate_pipeline import generate_shap_explanations

        with pytest.raises(ValueError, match="gradient-boosted model"):
            generate_shap_explanations(
                pd.DataFrame({"a": [1.0]}), pd.Series([0]),
                pd.Series(pd.to_datetime(["2022-08-27"])),
                [(np.array([0]), np.array([0]))], 1,
                model_name="logistic",
            )

    def test_fold_out_of_range_rejected(self) -> None:
        from src.evaluate_pipeline import generate_shap_explanations

        with pytest.raises(ValueError, match="out of range"):
            generate_shap_explanations(
                pd.DataFrame({"a": [1.0]}), pd.Series([0]),
                pd.Series(pd.to_datetime(["2022-08-27"])),
                [(np.array([0]), np.array([0]))], 9,
                model_name="xgboost",
            )


# ── Manuscript styling of the waterfall ──────────────────────────────


class TestWaterfallStyle:
    """Figure 9 must match the other figures, not SHAP's own defaults."""

    @pytest.fixture(scope="class")
    def rendered(self, fitted_model_and_data, tmp_path_factory):
        model, X, y, proba = fitted_model_and_data
        out = tmp_path_factory.mktemp("waterfall")
        path = out / "waterfall.png"
        SHAPExplainer(model).plot_local_decision(
            X, index=3, save_path=str(path),
            training_medians=X.median(),
        )
        return path

    def test_writes_a_vector_sibling(self, rendered) -> None:
        """The manuscript needs PDF; the pipeline references PNG."""
        assert rendered.exists()
        assert rendered.with_suffix(".pdf").exists()
        assert rendered.with_suffix(".pdf").stat().st_size > 0

    def test_training_medians_reach_the_labels(
        self, fitted_model_and_data,
    ) -> None:
        from src.models.explanation import _format_waterfall_label

        label = _format_waterfall_label("soil_moisture_lag1h", 70.3, 72.38)
        assert "70.3" in label
        assert "soil_moisture_lag1h" in label
        assert "72.38" in label

    def test_label_omits_the_median_when_absent(self) -> None:
        from src.models.explanation import _format_waterfall_label

        label = _format_waterfall_label("soil_moisture_lag1h", 70.3, None)
        assert "median" not in label

    def test_medians_may_be_passed_as_a_series(
        self, fitted_model_and_data, tmp_path,
    ) -> None:
        model, X, _y, _p = fitted_model_and_data
        SHAPExplainer(model).plot_local_decision(
            X, index=0, save_path=str(tmp_path / "w.png"),
            training_medians=X.median(),
        )
        assert (tmp_path / "w.png").exists()

    def test_mismatched_median_length_rejected(
        self, fitted_model_and_data, tmp_path,
    ) -> None:
        """A silent misalignment would mislabel every row."""
        model, X, _y, _p = fitted_model_and_data
        with pytest.raises(ValueError, match="same length"):
            SHAPExplainer(model).plot_local_decision(
                X, index=0, save_path=str(tmp_path / "w.png"),
                training_medians=[1.0, 2.0],
            )

    @pytest.mark.parametrize("max_display", [1, 2, 3, 10])
    def test_pooled_row_keeps_the_bars_summing_to_fx(
        self, max_display: int,
    ) -> None:
        """Truncating to max_display must not lose contribution mass.

        The figure's promise is that the bars carry E[f(x)] to f(x). If
        the pooled row were dropped or miscomputed, the bars would stop
        short of f(x) and the reader could not check the arithmetic.
        """
        from src.models.explanation import _waterfall_rows

        values = np.array([2.5, -1.2, 0.8, -0.4, 0.15, -0.05])
        names = [f"f{i}" for i in range(len(values))]
        row_values, row_labels = _waterfall_rows(
            values, names, values, [None] * len(values), max_display,
        )
        assert sum(row_values) == pytest.approx(values.sum())
        assert len(row_values) == len(row_labels)

    def test_largest_contributors_come_first(self) -> None:
        from src.models.explanation import _waterfall_rows

        values = np.array([0.1, -3.0, 0.5])
        row_values, row_labels = _waterfall_rows(
            values, ["a", "b", "c"], values, [None] * 3, 3,
        )
        assert row_values[0] == pytest.approx(-3.0)
        assert row_labels[0].endswith("b")

    def test_pooled_row_is_labelled_with_its_count(self) -> None:
        from src.models.explanation import _waterfall_rows

        values = np.array([2.0, 1.0, 0.5, 0.25, 0.1])
        _row_values, row_labels = _waterfall_rows(
            values, [f"f{i}" for i in range(5)], values, [None] * 5, 2,
        )
        assert row_labels[-1] == "3 other features"

    def test_no_pooled_row_when_everything_fits(self) -> None:
        from src.models.explanation import _waterfall_rows

        values = np.array([2.0, 1.0])
        _row_values, row_labels = _waterfall_rows(
            values, ["a", "b"], values, [None] * 2, 10,
        )
        assert not any("other features" in label for label in row_labels)

    def test_respects_max_display(
        self, fitted_model_and_data, tmp_path,
    ) -> None:
        model, X, _y, _p = fitted_model_and_data
        SHAPExplainer(model).plot_local_decision(
            X, index=0, save_path=str(tmp_path / "w.png"), max_display=2,
        )
        assert (tmp_path / "w.png").exists()
