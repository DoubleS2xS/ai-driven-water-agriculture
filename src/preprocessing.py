"""Per-fold preprocessing policy.

Single place where it is decided what happens to a feature matrix before
an estimator sees it.

The leakage rule
----------------
Every transformer declared here is returned **unfitted**.  It is fitted
inside a training fold and only ``transform``-ed on that fold's test
block, never the other way round and never on the whole frame.  The
earlier revision of this pipeline ran MICE imputation across the entire
DataFrame *before* splitting, which let test-set values inform the
imputation model that was later applied to the test set — a textbook
train/test contamination that inflates every downstream metric.

Wrapping the transformer and the estimator in a single
:class:`sklearn.pipeline.Pipeline` makes the correct behaviour the only
reachable one: ``pipeline.fit(X_train, y_train)`` cannot see test rows,
and ``pipeline.predict(X_test)`` cannot refit.

Why trees get ``"passthrough"``
-------------------------------
Gradient-boosted trees are invariant to monotone rescaling, so
standardisation would change nothing, and both XGBoost and LightGBM route
NaN down a learned default branch, which is strictly more informative
than substituting a median. Imposing impute-then-scale on them would
discard information and add fitted state for no benefit. Linear models
have neither property and get the full treatment.
"""

from __future__ import annotations

import logging
from typing import Tuple, Union

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

#: Estimators that consume raw features directly.
TREE_MODELS: Tuple[str, ...] = ("xgboost", "lightgbm")

#: Estimators that need finite, comparably scaled inputs.
LINEAR_MODELS: Tuple[str, ...] = ("logistic",)

Preprocessor = Union[str, Pipeline]


def make_preprocessor(model_type: str) -> Preprocessor:
    """Return the unfitted preprocessing step for *model_type*.

    Parameters
    ----------
    model_type : str
        One of :data:`TREE_MODELS` or :data:`LINEAR_MODELS`.

    Returns
    -------
    str or Pipeline
        ``"passthrough"`` for tree ensembles; a median-imputer plus
        standard-scaler ``Pipeline`` for linear models.  Always
        unfitted — the caller must fit it on training rows only.

    Raises
    ------
    ValueError
        If *model_type* is unknown.
    """
    if model_type in TREE_MODELS:
        return "passthrough"

    if model_type in LINEAR_MODELS:
        return Pipeline([
            # Median rather than mean: the lag features inherit the
            # skewed, spiky distribution of the raw sensor channels.
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])

    raise ValueError(
        f"Unknown model_type '{model_type}'. Expected one of "
        f"{TREE_MODELS + LINEAR_MODELS}."
    )


def build_model_pipeline(model_type: str, estimator: object) -> Pipeline:
    """Wrap *estimator* behind its preprocessing step.

    Parameters
    ----------
    model_type : str
        Selects the preprocessing policy.
    estimator : object
        An unfitted scikit-learn-compatible classifier.

    Returns
    -------
    Pipeline
        Two-step pipeline ``preprocess → clf``.  Fitting it fits both
        steps on the same rows; there is no code path that fits the
        transformer on data the estimator does not also see.
    """
    return Pipeline([
        ("preprocess", make_preprocessor(model_type)),
        ("clf", estimator),
    ])
