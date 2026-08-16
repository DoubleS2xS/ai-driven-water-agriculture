"""Tests for src.validation and src.preprocessing.

Two properties matter here and both are enforced empirically:

1. **Temporal order** — no fold may train on data that comes after its
   test block, and train/test may never overlap.
2. **No preprocessing leakage** — a transformer must be fitted on
   training rows only.  The decisive test perturbs the test block and
   confirms the fitted statistics do not move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.config import ValidationConfig
from src.preprocessing import (
    LINEAR_MODELS,
    TREE_MODELS,
    build_model_pipeline,
    make_preprocessor,
)
from src.validation import (
    assert_splits_are_ordered,
    chronological_holdout_split,
    describe_folds,
    rolling_origin_splits,
)


# ── Rolling-origin CV ────────────────────────────────────────────────


class TestRollingOriginSplits:
    def test_default_fold_count(self) -> None:
        splits = rolling_origin_splits(1200)
        assert len(splits) == ValidationConfig().n_folds == 5

    def test_folds_are_temporally_ordered(self) -> None:
        splits = rolling_origin_splits(1200)
        assert_splits_are_ordered(splits)  # must not raise

    def test_training_set_expands_monotonically(self) -> None:
        splits = rolling_origin_splits(1200, ValidationConfig(expanding=True))
        sizes = [len(train) for train, _ in splits]
        assert sizes == sorted(sizes)
        assert sizes[0] < sizes[-1]

    def test_sliding_window_keeps_train_size_bounded(self) -> None:
        splits = rolling_origin_splits(1200, ValidationConfig(expanding=False))
        sizes = [len(train) for train, _ in splits]
        assert max(sizes) <= 1200 // 6

    def test_test_blocks_are_contiguous_and_disjoint(self) -> None:
        splits = rolling_origin_splits(1200)
        test_blocks = [test for _, test in splits]
        for block in test_blocks:
            assert np.all(np.diff(block) == 1), "test block is not contiguous"
        pooled = np.concatenate(test_blocks)
        assert len(pooled) == len(set(pooled.tolist())), "test blocks overlap"

    def test_no_row_is_in_train_and_test_of_same_fold(self) -> None:
        for train, test in rolling_origin_splits(1200):
            assert not set(train.tolist()) & set(test.tolist())

    def test_gap_is_respected(self) -> None:
        gap = 24
        splits = rolling_origin_splits(1200, ValidationConfig(gap_hours=gap))
        for train, test in splits:
            assert test.min() - train.max() > gap

    def test_too_few_folds_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_folds must be >= 2"):
            rolling_origin_splits(1200, ValidationConfig(n_folds=1))

    def test_too_few_samples_rejected(self) -> None:
        with pytest.raises(ValueError, match="Need at least"):
            rolling_origin_splits(3, ValidationConfig(n_folds=5))


# ── Chronological holdout ────────────────────────────────────────────


class TestChronologicalHoldout:
    def test_split_point_matches_fraction(self) -> None:
        train, test = chronological_holdout_split(1000)
        assert len(train) == 800
        assert len(test) == 200

    def test_test_block_is_the_most_recent(self) -> None:
        train, test = chronological_holdout_split(1000)
        assert test.min() > train.max()

    def test_never_shuffles(self) -> None:
        train, test = chronological_holdout_split(1000)
        assert np.array_equal(train, np.arange(800))
        assert np.array_equal(test, np.arange(800, 1000))

    @pytest.mark.parametrize("fraction", [0.0, 1.0])
    def test_degenerate_fraction_rejected(self, fraction: float) -> None:
        with pytest.raises(ValueError, match="empty side"):
            chronological_holdout_split(
                1000, ValidationConfig(holdout_train_fraction=fraction),
            )


# ── Order guard ──────────────────────────────────────────────────────


class TestAssertSplitsAreOrdered:
    def test_shuffled_split_rejected(self) -> None:
        """A shuffled k-fold must be caught, not silently scored."""
        rng = np.random.default_rng(0)
        idx = rng.permutation(100)
        bad = [(idx[:80], idx[80:])]
        with pytest.raises(ValueError, match="not after the last training"):
            assert_splits_are_ordered(bad)

    def test_overlapping_split_rejected(self) -> None:
        bad = [(np.arange(0, 60), np.arange(50, 100))]
        with pytest.raises(ValueError, match="train and test overlap"):
            assert_splits_are_ordered(bad)

    def test_reversed_split_rejected(self) -> None:
        """Training on the future to predict the past must be caught."""
        bad = [(np.arange(50, 100), np.arange(0, 50))]
        with pytest.raises(ValueError, match="not after the last training"):
            assert_splits_are_ordered(bad)

    def test_empty_block_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty train or test"):
            assert_splits_are_ordered([(np.arange(10), np.array([], dtype=int))])


# ── Fold description ─────────────────────────────────────────────────


class TestDescribeFolds:
    @staticmethod
    def _target(n: int = 1200, seed: int = 0) -> pd.Series:
        rng = np.random.default_rng(seed)
        return pd.Series(rng.choice([0, 1], n, p=[0.77, 0.23]))

    def test_row_per_fold(self) -> None:
        y = self._target()
        table = describe_folds(rolling_origin_splits(len(y)), y)
        assert len(table) == 5

    def test_reports_both_train_and_test_positives(self) -> None:
        y = self._target()
        table = describe_folds(rolling_origin_splits(len(y)), y)
        assert {"train_positives", "test_positives"} <= set(table.columns)
        assert (table["test_positives"] > 0).all()

    def test_flags_fold_with_too_few_training_positives(self) -> None:
        """A fold can be unusable because of its *training* block."""
        y = pd.Series(np.zeros(1200, dtype=int))
        y.iloc[900:] = 1  # positives only in the final fold
        table = describe_folds(rolling_origin_splits(len(y)), y)
        assert not table["sufficient_positives"].iloc[0]

    def test_timestamps_reported_when_supplied(self) -> None:
        y = self._target()
        ts = pd.Series(pd.date_range("2022-07-12", periods=len(y), freq="1h"))
        table = describe_folds(rolling_origin_splits(len(y)), y, ts)
        assert {"test_start", "test_end"} <= set(table.columns)
        assert table["test_start"].is_monotonic_increasing


# ── Preprocessing policy ─────────────────────────────────────────────


class TestPreprocessor:
    @pytest.mark.parametrize("model_type", TREE_MODELS)
    def test_tree_models_get_passthrough(self, model_type: str) -> None:
        assert make_preprocessor(model_type) == "passthrough"

    @pytest.mark.parametrize("model_type", LINEAR_MODELS)
    def test_linear_models_get_impute_and_scale(self, model_type: str) -> None:
        pre = make_preprocessor(model_type)
        assert isinstance(pre, Pipeline)
        assert list(pre.named_steps) == ["impute", "scale"]

    def test_unknown_model_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown model_type"):
            make_preprocessor("random_forest")

    def test_returned_preprocessor_is_unfitted(self) -> None:
        """It must be fitted per fold, so it cannot arrive pre-fitted."""
        pre = make_preprocessor("logistic")
        assert not hasattr(pre.named_steps["scale"], "mean_")


class TestNoPreprocessingLeakage:
    """The fitted statistics must depend on training rows only."""

    @staticmethod
    def _data(seed: int = 0):
        rng = np.random.default_rng(seed)
        X = pd.DataFrame({
            "a": rng.normal(0, 1, 200),
            "b": rng.normal(5, 2, 200),
        })
        y = pd.Series(rng.choice([0, 1], 200))
        return X, y

    def test_scaler_ignores_test_block(self) -> None:
        """Corrupt the test block; the fitted scale must not move.

        If any transformer were fitted on the full frame, blowing up the
        held-out rows by three orders of magnitude would shift the mean
        and variance and this comparison would fail.
        """
        X, y = self._data()
        train_idx = np.arange(160)
        test_idx = np.arange(160, 200)

        from sklearn.linear_model import LogisticRegression

        pipe_clean = build_model_pipeline("logistic", LogisticRegression())
        pipe_clean.fit(X.iloc[train_idx], y.iloc[train_idx])

        X_corrupted = X.copy()
        X_corrupted.iloc[test_idx] *= 1000.0

        pipe_corrupted = build_model_pipeline("logistic", LogisticRegression())
        pipe_corrupted.fit(X_corrupted.iloc[train_idx], y.iloc[train_idx])

        np.testing.assert_allclose(
            pipe_clean.named_steps["preprocess"].named_steps["scale"].mean_,
            pipe_corrupted.named_steps["preprocess"].named_steps["scale"].mean_,
        )

    def test_imputer_statistics_come_from_train_only(self) -> None:
        X, y = self._data()
        X.iloc[170:, 0] = np.nan  # missing values only in the test block
        train_idx = np.arange(160)

        from sklearn.linear_model import LogisticRegression

        pipe = build_model_pipeline("logistic", LogisticRegression())
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])

        fitted_median = (
            pipe.named_steps["preprocess"].named_steps["impute"].statistics_[0]
        )
        assert fitted_median == pytest.approx(
            X.iloc[train_idx, 0].median()
        )

    def test_pipeline_wires_preprocess_then_classifier(self) -> None:
        from sklearn.linear_model import LogisticRegression

        pipe = build_model_pipeline("logistic", LogisticRegression())
        assert list(pipe.named_steps) == ["preprocess", "clf"]


# ── Integration with the real design matrix ──────────────────────────


class TestRealDesignMatrix:
    def test_pipeline_produces_ordered_usable_folds(self) -> None:
        from src.evaluate_pipeline import (
            build_design_matrix,
            load_modeling_frame,
        )

        df = load_modeling_frame()
        X, y, ts, blocks = build_design_matrix(df)

        assert ts.is_monotonic_increasing
        assert len(X) == len(y) == len(ts)
        assert "soil_moisture" not in X.columns
        assert not [c for c in X.columns if c.startswith("flow_l")]

        splits = rolling_origin_splits(len(X))
        assert_splits_are_ordered(splits)
        assert len(splits) == 5
