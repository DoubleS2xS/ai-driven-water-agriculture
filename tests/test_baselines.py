"""Tests for src.baselines and src.metrics.

A baseline is only useful if it does exactly what its name claims, so
each one is checked against a hand-computed expectation rather than
against itself.  A persistence baseline that quietly did something
cleverer would make the main models look worse than they are; one that
did something dumber would make them look better.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.baselines import (
    BASELINE_REGISTRY,
    LogisticRegressionBaseline,
    MajorityClassBaseline,
    MoistureThresholdBaseline,
    PersistenceBaseline,
    available_baselines,
    make_baseline,
)
from src.metrics import (
    HIGHER_IS_BETTER,
    PRIMARY_METRICS,
    compute_classification_metrics,
    format_confusion_matrix,
)


# ── Fixtures ──────────────────────────────────────────────────────────


def _design(n: int = 300, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Design matrix whose target is driven by moisture, with persistence.

    Soil moisture is generated as a smooth wetting/drying cycle rather
    than i.i.d. noise, so ``moisture < 40`` produces *blocky* irrigation
    episodes the way the real relay data does.  An i.i.d. fixture would
    have no autocorrelation at all, which would make the persistence
    baseline untestable — its whole premise is that the target repeats.
    """
    rng = np.random.default_rng(seed)
    hours = np.arange(n)
    moisture = 50.0 + 25.0 * np.sin(hours / 18.0) + rng.normal(0, 2.0, n)
    y = (moisture < 40).astype(int)
    X = pd.DataFrame({
        "soil_moisture_lag1h": moisture,
        "irrigation_event_lag1h": np.concatenate([[0.0], y[:-1]]).astype(float),
        "air_temp_lag1h": rng.normal(20, 5, n),
    })
    return X, pd.Series(y)


@pytest.fixture
def design() -> tuple[pd.DataFrame, pd.Series]:
    return _design()


# ── Majority ─────────────────────────────────────────────────────────


class TestMajorityClassBaseline:
    def test_predicts_zero_under_imbalance(self, design) -> None:
        X, y = design
        model = MajorityClassBaseline().fit(X, y)
        assert model.majority_class_ == 0
        assert set(np.unique(model.predict(X))) == {0}

    def test_f1_is_zero(self, design) -> None:
        """Its purpose: show that high accuracy can coexist with F1 = 0."""
        X, y = design
        model = MajorityClassBaseline().fit(X, y)
        m = compute_classification_metrics(
            y, model.predict(X), model.predict_proba(X)[:, 1],
        )
        assert m["f1"] == 0.0
        assert m["tp"] == 0

    def test_roc_auc_is_exactly_half(self, design) -> None:
        """A constant score cannot rank, so AUC is 0.5 by construction."""
        X, y = design
        model = MajorityClassBaseline().fit(X, y)
        m = compute_classification_metrics(
            y, model.predict(X), model.predict_proba(X)[:, 1],
        )
        assert m["roc_auc"] == pytest.approx(0.5)

    def test_pr_auc_approximates_prevalence(self, design) -> None:
        X, y = design
        model = MajorityClassBaseline().fit(X, y)
        m = compute_classification_metrics(
            y, model.predict(X), model.predict_proba(X)[:, 1],
        )
        assert m["pr_auc"] == pytest.approx(y.mean(), abs=0.02)

    def test_follows_majority_when_positives_dominate(self) -> None:
        X, y = _design()
        y_flipped = pd.Series(np.ones(len(y), dtype=int))
        model = MajorityClassBaseline().fit(X, y_flipped)
        assert model.majority_class_ == 1


# ── Persistence ──────────────────────────────────────────────────────


class TestPersistenceBaseline:
    def test_copies_previous_valve_state_exactly(self, design) -> None:
        X, y = design
        model = PersistenceBaseline().fit(X, y)
        expected = (X["irrigation_event_lag1h"] > 0.5).astype(int).to_numpy()
        np.testing.assert_array_equal(model.predict(X), expected)

    def test_conditional_rates_are_learned_from_train(self) -> None:
        X = pd.DataFrame({"irrigation_event_lag1h": [0.0, 0.0, 1.0, 1.0]})
        y = pd.Series([0, 1, 1, 1])
        model = PersistenceBaseline().fit(X, y)
        assert model.rate_given_off_ == pytest.approx(0.5)
        assert model.rate_given_on_ == pytest.approx(1.0, abs=1e-5)

    def test_probabilities_track_the_lag(self, design) -> None:
        X, y = design
        model = PersistenceBaseline().fit(X, y)
        proba = model.predict_proba(X)[:, 1]
        on = X["irrigation_event_lag1h"] > 0.5
        assert proba[on].min() > proba[~on].max()

    def test_unobserved_state_falls_back_to_prior(self) -> None:
        """No invented rate for a state absent from the training fold."""
        X = pd.DataFrame({"irrigation_event_lag1h": [0.0, 0.0, 0.0]})
        y = pd.Series([0, 1, 0])
        model = PersistenceBaseline().fit(X, y)
        assert model.rate_given_on_ == pytest.approx(y.mean())

    def test_missing_feature_raises_directive_error(self) -> None:
        X = pd.DataFrame({"air_temp_lag1h": [1.0, 2.0]})
        y = pd.Series([0, 1])
        with pytest.raises(ValueError, match="irrigation_event_lag1h"):
            PersistenceBaseline().fit(X, y)


# ── Moisture threshold ───────────────────────────────────────────────


class TestMoistureThresholdBaseline:
    def test_recovers_the_generating_threshold(self, design) -> None:
        """Data are generated by moisture < 40; the fit must find it."""
        X, y = design
        model = MoistureThresholdBaseline().fit(X, y)
        assert model.threshold_ == pytest.approx(40.0, abs=1.0)
        assert model.train_f1_ > 0.95

    def test_predicts_below_threshold(self, design) -> None:
        X, y = design
        model = MoistureThresholdBaseline().fit(X, y)
        expected = (
            X["soil_moisture_lag1h"] < model.threshold_
        ).astype(int).to_numpy()
        np.testing.assert_array_equal(model.predict(X), expected)

    def test_threshold_is_selected_on_training_data_only(self) -> None:
        """Refitting on a different fold must move the threshold."""
        X_a, y_a = _design(seed=1)
        X_b = X_a.copy()
        y_b = (X_a["soil_moisture_lag1h"] < 60).astype(int)

        model_a = MoistureThresholdBaseline().fit(X_a, y_a)
        model_b = MoistureThresholdBaseline().fit(X_b, y_b)
        assert model_a.threshold_ != pytest.approx(model_b.threshold_, abs=5.0)

    def test_probabilities_decrease_with_moisture(self, design) -> None:
        X, y = design
        model = MoistureThresholdBaseline().fit(X, y)
        order = np.argsort(X["soil_moisture_lag1h"].to_numpy())
        proba = model.predict_proba(X)[:, 1][order]
        assert proba[0] > proba[-1]

    def test_single_class_training_fold_does_not_crash(self) -> None:
        X, _ = _design(n=50)
        y = pd.Series(np.zeros(50, dtype=int))
        model = MoistureThresholdBaseline().fit(X, y)
        proba = model.predict_proba(X)[:, 1]
        assert np.all(np.isfinite(proba))

    def test_missing_feature_raises_directive_error(self) -> None:
        X = pd.DataFrame({"air_temp_lag1h": [1.0, 2.0]})
        y = pd.Series([0, 1])
        with pytest.raises(ValueError, match="soil_moisture_lag1h"):
            MoistureThresholdBaseline().fit(X, y)


# ── Logistic regression ──────────────────────────────────────────────


class TestLogisticRegressionBaseline:
    def test_learns_the_separable_rule(self, design) -> None:
        X, y = design
        model = LogisticRegressionBaseline().fit(X, y)
        m = compute_classification_metrics(
            y, model.predict(X), model.predict_proba(X)[:, 1],
        )
        assert m["roc_auc"] > 0.95

    def test_handles_nan_via_fold_local_imputer(self, design) -> None:
        X, y = design
        X = X.copy()
        X.iloc[:10, 0] = np.nan
        model = LogisticRegressionBaseline().fit(X, y)
        assert np.all(np.isfinite(model.predict_proba(X)))

    def test_is_deterministic_for_a_seed(self, design) -> None:
        X, y = design
        a = LogisticRegressionBaseline(random_state=7).fit(X, y)
        b = LogisticRegressionBaseline(random_state=7).fit(X, y)
        np.testing.assert_allclose(a.predict_proba(X), b.predict_proba(X))

    def test_single_class_fold_falls_back_to_the_prior(self, design) -> None:
        """Must not crash: at a 4 % onset rate this fold really occurs.

        The tree models degrade to a constant on their own; sklearn's
        LogisticRegression raises instead, which would abort the whole
        protocol on one unlucky fold.
        """
        X, _y = design
        y = pd.Series(np.zeros(len(X), dtype=int))
        model = LogisticRegressionBaseline().fit(X, y)

        assert model.degenerate_ is True
        proba = model.predict_proba(X)
        assert np.all(np.isfinite(proba))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
        assert set(np.unique(model.predict(X))) == {0}

    def test_two_class_fold_is_not_flagged_degenerate(self, design) -> None:
        X, y = design
        assert LogisticRegressionBaseline().fit(X, y).degenerate_ is False


# ── Registry ─────────────────────────────────────────────────────────


class TestRegistry:
    @pytest.mark.parametrize("name", list(BASELINE_REGISTRY))
    def test_every_baseline_fits_and_scores(self, name, design) -> None:
        X, y = design
        model = make_baseline(name)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
        assert set(np.unique(model.predict(X))) <= {0, 1}

    def test_unknown_baseline_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown baseline"):
            make_baseline("magic")

    def test_available_baselines_drops_unusable_ones(self) -> None:
        usable = available_baselines(["air_temp_lag1h", "hour_sin"])
        assert "persistence" not in usable
        assert "moisture_threshold" not in usable
        assert "majority" in usable
        assert "logistic" in usable

    def test_available_baselines_keeps_all_on_full_set(self, design) -> None:
        X, _ = design
        assert set(available_baselines(X.columns)) == set(BASELINE_REGISTRY)


# ── Metrics ──────────────────────────────────────────────────────────


class TestMetrics:
    def test_perfect_prediction(self) -> None:
        y = np.array([0, 0, 1, 1])
        m = compute_classification_metrics(y, y, y.astype(float))
        assert m["f1"] == 1.0
        assert m["roc_auc"] == 1.0
        assert m["pr_auc"] == 1.0
        assert m["brier"] == 0.0

    def test_confusion_cells_sum_to_n(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.choice([0, 1], 100)
        pred = rng.choice([0, 1], 100)
        m = compute_classification_metrics(y, pred, pred.astype(float))
        assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == m["n"] == 100

    def test_single_class_gives_nan_not_zero(self) -> None:
        """AUC is undefined with one class — it must not read as 0.0."""
        y = np.zeros(10, dtype=int)
        m = compute_classification_metrics(y, y, np.full(10, 0.1))
        assert np.isnan(m["roc_auc"])
        assert np.isnan(m["pr_auc"])
        assert np.isfinite(m["brier"])

    def test_positive_rate_reported_for_pr_auc_context(self) -> None:
        y = np.array([0, 0, 0, 1])
        m = compute_classification_metrics(y, y, y.astype(float))
        assert m["positive_rate"] == pytest.approx(0.25)
        assert m["n_positive"] == 1

    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_classification_metrics([0, 1], [0], [0.1, 0.9])

    def test_empty_input_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty test block"):
            compute_classification_metrics([], [], [])

    def test_all_primary_metrics_present(self) -> None:
        y = np.array([0, 1, 0, 1])
        m = compute_classification_metrics(y, y, y.astype(float))
        assert set(PRIMARY_METRICS) <= set(m)

    def test_brier_is_the_only_lower_is_better_metric(self) -> None:
        lower = [k for k, v in HIGHER_IS_BETTER.items() if not v]
        assert lower == ["brier"]

    def test_confusion_matrix_renders(self) -> None:
        y = np.array([0, 1, 0, 1])
        m = compute_classification_metrics(y, y, y.astype(float))
        text = format_confusion_matrix(m)
        assert "pred 0" in text and "actual 1" in text
