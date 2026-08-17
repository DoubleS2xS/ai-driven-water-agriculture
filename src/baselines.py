"""Reference models the gradient-boosted classifiers must beat.

Without these, a reported F1 or AUC is uninterpretable.  Two facts about
this dataset make that concrete:

* ``irrigation_event(t-1)`` correlates with the target at **r = 0.81**.
  Most of the achievable score is available from the previous hour's
  valve state alone, so any model that does not clear
  :class:`PersistenceBaseline` has learned nothing about irrigation.
* Fold 4's test block is 79.8 % positive, where predicting "always
  irrigate" scores F1 ≈ 0.887.  A headline F1 of 0.98 on that fold is a
  small improvement on a constant, not a strong result.

Every baseline implements the scikit-learn classifier API
(``fit``/``predict``/``predict_proba``) so the identical fold loop,
metrics and statistical tests apply to baselines and to the main models.

Probabilities
-------------
Each baseline exposes calibrated probabilities as well as hard
predictions.  Without them, ROC-AUC and PR-AUC would be degenerate and
the Brier score meaningless, and the comparison against the main models
would have to fall back on threshold-dependent metrics only.  Where a
baseline's *decision rule* is a hard rule (persistence, moisture
threshold), the rule is preserved exactly in ``predict`` and the
probabilities are fitted separately on the training fold — so
``predict`` and ``predict_proba`` may disagree at the 0.5 mark, exactly
as they would for any classifier with a tuned operating point.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.preprocessing import build_model_pipeline

logger = logging.getLogger(__name__)

#: Feature carrying the previous hour's valve state.
PERSISTENCE_FEATURE: str = "irrigation_event_lag1h"

#: Feature carrying the previous hour's soil moisture.
MOISTURE_FEATURE: str = "soil_moisture_lag1h"

#: Number of candidate cut-points evaluated by the threshold baseline.
N_THRESHOLD_CANDIDATES: int = 200

#: Bounds applied to fitted conditional rates, so a fold in which a bin
#: is perfectly pure does not yield an infinite log-loss / zero Brier.
_PROBA_FLOOR: float = 1e-6
_PROBA_CEIL: float = 1.0 - 1e-6


def _require_column(X: pd.DataFrame, column: str, model: str) -> pd.Series:
    """Return *column* from *X* or raise a directive error."""
    if column not in X.columns:
        raise ValueError(
            f"{model} requires the feature '{column}', which is absent from "
            f"the design matrix. Available: {sorted(X.columns)[:8]}… "
            f"This baseline cannot be evaluated on a feature set that "
            f"excludes it (e.g. an ablation set without the relevant block)."
        )
    return X[column]


# ======================================================================
# 1 — Majority class
# ======================================================================

class MajorityClassBaseline(BaseEstimator, ClassifierMixin):
    """Always predicts the most frequent class of the training fold.

    The floor for any classifier.  Because irrigation hours are the
    minority throughout, this predicts ``0`` everywhere and scores
    F1 = 0 — its purpose is to expose the accuracy trap: at a 23 %
    positive rate it is already 77 % *accurate* while being useless,
    which is why accuracy is not among the reported metrics.

    Its constant probability makes ROC-AUC exactly 0.5 by construction,
    the reference point for the AUC comparisons.

    Attributes:
        majority_class_: The class predicted for every row.
        prior_: Training-fold positive rate, used as the constant score.
    """

    def fit(
        self, X: pd.DataFrame, y: Sequence[int],
    ) -> "MajorityClassBaseline":
        """Record the majority class and the positive prior."""
        y = np.asarray(y)
        self.prior_ = float(y.mean())
        self.majority_class_ = int(self.prior_ > 0.5)
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.majority_class_, dtype=int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.clip(self.prior_, _PROBA_FLOOR, _PROBA_CEIL)
        return np.column_stack([np.full(len(X), 1 - p), np.full(len(X), p)])


# ======================================================================
# 2 — Persistence
# ======================================================================

class PersistenceBaseline(BaseEstimator, ClassifierMixin):
    """Predicts ``irrigation_event(t) = irrigation_event(t-1)``.

    The baseline that matters most here.  Irrigation runs in blocks —
    52 episodes, median length 2 h, longest 117 h — so simply repeating
    the previous hour is a strong predictor, and the main models must be
    shown to beat it before any claim about learned hydrology is
    credible.

    ``predict`` copies the lagged valve state exactly.  ``predict_proba``
    returns the empirical ``P(y = 1 | previous state)`` measured on the
    training fold, a two-cell lookup, so that the AUC and Brier
    comparisons are defined.

    Attributes:
        rate_given_on_: Training-fold P(y=1 | previous hour irrigating).
        rate_given_off_: Training-fold P(y=1 | previous hour idle).
    """

    def __init__(self, lag_feature: str = PERSISTENCE_FEATURE) -> None:
        self.lag_feature = lag_feature

    def fit(self, X: pd.DataFrame, y: Sequence[int]) -> "PersistenceBaseline":
        """Estimate the conditional positive rate for each previous state."""
        lag = _require_column(X, self.lag_feature, "PersistenceBaseline")
        y = pd.Series(np.asarray(y), index=lag.index)

        on = lag > 0.5
        # Fall back to the overall prior when a state is unobserved in
        # this fold, rather than inventing a rate for it.
        prior = float(y.mean())
        self.rate_given_on_ = float(y[on].mean()) if on.any() else prior
        self.rate_given_off_ = float(y[~on].mean()) if (~on).any() else prior
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        lag = _require_column(X, self.lag_feature, "PersistenceBaseline")
        return (lag.to_numpy() > 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        lag = _require_column(X, self.lag_feature, "PersistenceBaseline")
        p = np.where(
            lag.to_numpy() > 0.5, self.rate_given_on_, self.rate_given_off_,
        )
        p = np.clip(p, _PROBA_FLOOR, _PROBA_CEIL)
        return np.column_stack([1 - p, p])


# ======================================================================
# 3 — Soil-moisture threshold
# ======================================================================

class MoistureThresholdBaseline(BaseEstimator, ClassifierMixin):
    """Irrigates when the previous hour's soil moisture is low enough.

    The agronomic rule of thumb, and the decision logic the physical
    controller in the source dataset appears to implement.  The cut-point
    is **selected on the training fold** by maximising F1 over
    :data:`N_THRESHOLD_CANDIDATES` quantiles of the training
    distribution; it is never tuned on test data.

    ``predict`` applies that hard cut-point.  ``predict_proba`` comes
    from a one-dimensional logistic regression fitted on the same
    training fold, which supplies the calibrated scores that ROC-AUC,
    PR-AUC and Brier require.  The two can disagree around 0.5 because
    the cut-point is chosen for F1, not for calibration.

    Attributes:
        threshold_: Fitted moisture cut-point; irrigate when below it.
        train_f1_: F1 achieved by that cut-point on the training fold.
    """

    def __init__(
        self,
        moisture_feature: str = MOISTURE_FEATURE,
        n_candidates: int = N_THRESHOLD_CANDIDATES,
    ) -> None:
        self.moisture_feature = moisture_feature
        self.n_candidates = n_candidates

    def fit(
        self, X: pd.DataFrame, y: Sequence[int],
    ) -> "MoistureThresholdBaseline":
        """Select the F1-optimal cut-point and calibrate probabilities."""
        moisture = _require_column(
            X, self.moisture_feature, "MoistureThresholdBaseline",
        )
        y_arr = np.asarray(y)
        values = moisture.to_numpy()

        candidates = np.unique(
            np.quantile(values, np.linspace(0.0, 1.0, self.n_candidates))
        )
        best_threshold = float(candidates[0])
        best_f1 = -1.0
        for candidate in candidates:
            score = f1_score(
                y_arr, (values < candidate).astype(int), zero_division=0,
            )
            if score > best_f1:
                best_f1 = float(score)
                best_threshold = float(candidate)

        self.threshold_ = best_threshold
        self.train_f1_ = best_f1

        self._calibrator = LogisticRegression(max_iter=1000)
        if len(np.unique(y_arr)) > 1:
            self._calibrator.fit(values.reshape(-1, 1), y_arr)
        else:
            self._calibrator = None
            self._constant_rate = float(y_arr.mean())

        self.classes_ = np.array([0, 1])
        logger.debug(
            "MoistureThresholdBaseline: threshold=%.4f (train F1=%.4f)",
            self.threshold_, self.train_f1_,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        moisture = _require_column(
            X, self.moisture_feature, "MoistureThresholdBaseline",
        )
        return (moisture.to_numpy() < self.threshold_).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        moisture = _require_column(
            X, self.moisture_feature, "MoistureThresholdBaseline",
        )
        values = moisture.to_numpy().reshape(-1, 1)
        if self._calibrator is None:
            p = np.full(len(values), self._constant_rate)
            p = np.clip(p, _PROBA_FLOOR, _PROBA_CEIL)
            return np.column_stack([1 - p, p])
        return np.asarray(self._calibrator.predict_proba(values))


# ======================================================================
# 4 — Logistic regression
# ======================================================================

class LogisticRegressionBaseline(BaseEstimator, ClassifierMixin):
    """Linear model on the same causal lag features as the main models.

    Separates "the features carry signal" from "the signal needs a
    non-linear model".  If the gradient-boosted trees do not clearly beat
    this, the extra capacity is not earning its complexity and the paper
    should say so.

    Imputation and standardisation are fitted inside the training fold by
    :func:`src.preprocessing.build_model_pipeline`; the linear model
    needs both, unlike the trees.

    Class weights are left at their default, matching the main models, so
    that any difference is attributable to the hypothesis class rather
    than to a different treatment of the imbalance.

    Single-class training folds
    --------------------------
    ``LogisticRegression`` raises when a training fold contains only one
    class, unlike the tree models, which fall back to a constant.  That
    is not a hypothetical: at the ~4 % onset rate of
    :mod:`src.onset`, an early fold's training block can legitimately
    contain no positive example at all.  Rather than crash the whole
    protocol, this baseline then predicts the training prior — the only
    thing the fold supports — and records the fallback in
    :attr:`degenerate_`.

    Attributes:
        degenerate_: ``True`` when the training fold held a single class
            and the model fell back to the prior.
    """

    def __init__(self, random_state: int = 42, max_iter: int = 1000) -> None:
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(
        self, X: pd.DataFrame, y: Sequence[int],
    ) -> "LogisticRegressionBaseline":
        y_arr = np.asarray(y)
        self.classes_ = np.array([0, 1])
        self.degenerate_ = len(np.unique(y_arr)) < 2

        if self.degenerate_:
            self._pipeline = None
            self._prior = float(y_arr.mean())
            logger.warning(
                "LogisticRegressionBaseline: training fold contains a single "
                "class (%d rows, %d positive). Falling back to the constant "
                "training prior %.4f; no coefficients can be estimated.",
                len(y_arr), int(y_arr.sum()), self._prior,
            )
            return self

        self._pipeline = build_model_pipeline(
            "logistic",
            LogisticRegression(
                max_iter=self.max_iter, random_state=self.random_state,
            ),
        )
        self._pipeline.fit(X, y_arr)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._pipeline is None:
            return np.full(len(X), int(self._prior > 0.5), dtype=int)
        return np.asarray(self._pipeline.predict(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._pipeline is None:
            p = np.clip(self._prior, _PROBA_FLOOR, _PROBA_CEIL)
            return np.column_stack(
                [np.full(len(X), 1 - p), np.full(len(X), p)]
            )
        return np.asarray(self._pipeline.predict_proba(X))


# ======================================================================
# Registry
# ======================================================================

#: Human-readable name → zero-argument constructor.
BASELINE_REGISTRY: Dict[str, type] = {
    "majority": MajorityClassBaseline,
    "persistence": PersistenceBaseline,
    "moisture_threshold": MoistureThresholdBaseline,
    "logistic": LogisticRegressionBaseline,
}


def make_baseline(name: str, seed: int = 42) -> BaseEstimator:
    """Instantiate a baseline by name.

    Args:
        name: Key of :data:`BASELINE_REGISTRY`.
        seed: Random seed, forwarded to baselines that take one.

    Returns:
        An unfitted estimator.

    Raises:
        ValueError: If *name* is unknown.
    """
    if name not in BASELINE_REGISTRY:
        raise ValueError(
            f"Unknown baseline '{name}'. Available: "
            f"{sorted(BASELINE_REGISTRY)}."
        )
    cls = BASELINE_REGISTRY[name]
    if cls is LogisticRegressionBaseline:
        return cls(random_state=seed)
    return cls()


def available_baselines(feature_names: Sequence[str]) -> List[str]:
    """Return the baselines evaluable on a given feature set.

    Persistence and the moisture threshold each need one specific
    column; on an ablation set that omits it, the baseline is skipped
    rather than silently evaluated on a substitute feature.

    Args:
        feature_names: Columns of the design matrix.

    Returns:
        Baseline names, in registry order.
    """
    names = set(feature_names)
    usable = []
    for name in BASELINE_REGISTRY:
        if name == "persistence" and PERSISTENCE_FEATURE not in names:
            continue
        if name == "moisture_threshold" and MOISTURE_FEATURE not in names:
            continue
        usable.append(name)
    return usable
