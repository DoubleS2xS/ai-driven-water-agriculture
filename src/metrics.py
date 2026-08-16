"""Classification metrics for an imbalanced, thresholded decision.

Roughly 23 % of hours are irrigation hours, so the metric set is chosen
for that regime rather than for a balanced one.

Why PR-AUC is the metric of record
----------------------------------
ROC-AUC is computed from true- and false-positive *rates*, and the
false-positive rate has the large negative class in its denominator.  A
classifier can therefore raise many false alarms relative to the number
of true alarms while barely moving its ROC curve.  Average precision
(PR-AUC) puts precision on the y-axis, where false positives compete
directly with true positives, and so it responds to exactly the failure
mode that matters for an irrigation controller: watering when the field
did not need it.

PR-AUC must be read against the positive rate, not against 0.5.  A
useless classifier scores ≈ the prevalence (≈ 0.23 here), not 0.5, so
``positive_rate`` is returned alongside it and no PR-AUC should be quoted
without it.

Threshold-free versus thresholded
---------------------------------
ROC-AUC, PR-AUC and the Brier score are computed from probabilities and
are independent of any operating point.  Precision, recall and F1 require
a threshold and are reported at the conventional 0.5.  On folds where the
positive rate is 3 % that threshold is far from optimal, which is why the
threshold-free metrics are the ones to compare models on.
"""

from __future__ import annotations

import logging
from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

#: Threshold-free metrics, comparable across folds with different balance.
THRESHOLD_FREE_METRICS: tuple[str, ...] = ("roc_auc", "pr_auc", "brier")

#: Metrics that depend on the 0.5 operating point.
THRESHOLDED_METRICS: tuple[str, ...] = ("precision", "recall", "f1")

#: Full ordered metric set reported in the paper.
PRIMARY_METRICS: tuple[str, ...] = (
    "roc_auc", "pr_auc", "f1", "precision", "recall", "brier",
)

#: Metrics where a larger value is better.  ``brier`` is the exception.
HIGHER_IS_BETTER: Dict[str, bool] = {
    "roc_auc": True,
    "pr_auc": True,
    "f1": True,
    "precision": True,
    "recall": True,
    "brier": False,
}

DEFAULT_THRESHOLD: float = 0.5


def compute_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    y_proba: Sequence[float] | np.ndarray,
) -> Dict[str, float]:
    """Compute the full metric set for one train/test evaluation.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Hard predictions at the operating threshold.
        y_proba: Predicted probability of the positive class.

    Returns:
        Mapping with the metrics in :data:`PRIMARY_METRICS`, the
        confusion-matrix cells (``tn``, ``fp``, ``fn``, ``tp``), and the
        context needed to read them (``n``, ``n_positive``,
        ``positive_rate``).  Threshold-free metrics are ``nan`` when the
        test block contains a single class, since they are undefined
        there rather than zero.

    Raises:
        ValueError: If the input lengths disagree.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba, dtype=float)

    if not (len(y_true) == len(y_pred) == len(y_proba)):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}, "
            f"y_proba={len(y_proba)}."
        )
    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics on an empty test block.")

    n_positive = int(y_true.sum())
    both_classes = 0 < n_positive < len(y_true)

    if both_classes:
        roc_auc = float(roc_auc_score(y_true, y_proba))
        pr_auc = float(average_precision_score(y_true, y_proba))
    else:
        # Undefined, not zero: with one class present there is no pair to
        # rank and no precision-recall curve to integrate.
        roc_auc = float("nan")
        pr_auc = float("nan")
        logger.warning(
            "Test block contains a single class (%d positives of %d); "
            "ROC-AUC and PR-AUC are undefined and reported as NaN.",
            n_positive,
            len(y_true),
        )

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1],
    ).ravel()

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, np.clip(y_proba, 0.0, 1.0))),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n": int(len(y_true)),
        "n_positive": n_positive,
        "positive_rate": float(y_true.mean()),
    }


def format_confusion_matrix(metrics: Dict[str, float]) -> str:
    """Render the confusion-matrix cells of *metrics* as a small table."""
    return (
        "                 pred 0   pred 1\n"
        f"    actual 0   {metrics['tn']:>7d}  {metrics['fp']:>7d}\n"
        f"    actual 1   {metrics['fn']:>7d}  {metrics['tp']:>7d}"
    )
