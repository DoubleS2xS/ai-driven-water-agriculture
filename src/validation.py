"""Temporal validation protocol: rolling-origin CV and chronological holdout.

Every splitter here preserves time order.  Nothing is shuffled, and a
fold's test block always lies strictly after its training block.  This is
not a stylistic preference: the target is an hourly relay state that
persists in blocks (median irrigation episode ≈ 2 h, longest 117 h), so a
shuffled *k*-fold would place hours from the *same* episode in both train
and test and report a score that no deployed system could reproduce.

Two protocols are provided:

**Rolling-origin cross-validation** (:func:`rolling_origin_splits`) — the
primary protocol.  Five expanding-window folds, each testing on a block
that immediately follows its training data.  Reporting five folds rather
than one split gives the dispersion needed to say whether a difference
between two models is real.

**Chronological holdout** (:func:`chronological_holdout_split`) — a
single 80/20 ordered split, retained as a secondary result for
comparability with the earlier revision of this pipeline and with papers
that report only a holdout number.

Both return positional index arrays into a frame that is assumed to be
sorted ascending by time.
"""

from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.config import ValidationConfig

logger = logging.getLogger(__name__)

Split = Tuple[np.ndarray, np.ndarray]


# ======================================================================
# Splitters
# ======================================================================

def rolling_origin_splits(
    n_samples: int,
    config: ValidationConfig | None = None,
) -> List[Split]:
    """Build expanding-origin cross-validation folds.

    Delegates to :class:`sklearn.model_selection.TimeSeriesSplit`, whose
    semantics are exactly the rolling-origin protocol: with
    ``n_splits = k`` the series is cut into ``k + 1`` contiguous blocks,
    fold *i* tests on block *i + 1* and trains on every block before it.
    Using the library implementation rather than a bespoke loop keeps the
    protocol recognisable to reviewers and free of off-by-one surprises.

    Parameters
    ----------
    n_samples : int
        Number of rows in the (time-ordered) design matrix.
    config : ValidationConfig, optional
        Protocol settings; defaults to :class:`ValidationConfig`.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        ``(train_idx, test_idx)`` positional index arrays, oldest fold
        first.

    Raises
    ------
    ValueError
        If *n_samples* is too small to form the requested folds.
    """
    cfg = config or ValidationConfig()

    if cfg.n_folds < 2:
        raise ValueError(
            f"n_folds must be >= 2 to estimate dispersion, got {cfg.n_folds}."
        )
    if n_samples < cfg.n_folds + 1:
        raise ValueError(
            f"Need at least n_folds + 1 = {cfg.n_folds + 1} rows to build "
            f"{cfg.n_folds} rolling-origin folds, got {n_samples}."
        )

    test_size = n_samples // (cfg.n_folds + 1)
    max_train_size = None if cfg.expanding else test_size

    splitter = TimeSeriesSplit(
        n_splits=cfg.n_folds,
        gap=cfg.gap_hours,
        max_train_size=max_train_size,
    )
    splits = [
        (train_idx, test_idx)
        for train_idx, test_idx in splitter.split(np.arange(n_samples))
    ]

    logger.info(
        "Rolling-origin CV: %d %s folds over %d rows (gap=%d)",
        cfg.n_folds,
        "expanding" if cfg.expanding else "sliding",
        n_samples,
        cfg.gap_hours,
    )
    return splits


def chronological_holdout_split(
    n_samples: int,
    config: ValidationConfig | None = None,
) -> Split:
    """Build a single ordered train/test split (no shuffle).

    Parameters
    ----------
    n_samples : int
        Number of rows in the time-ordered design matrix.
    config : ValidationConfig, optional
        Supplies
        :attr:`~src.config.ValidationConfig.holdout_train_fraction`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(train_idx, test_idx)``; the test block is the most recent
        rows.

    Raises
    ------
    ValueError
        If the split would leave either side empty.
    """
    cfg = config or ValidationConfig()

    split_at = int(n_samples * cfg.holdout_train_fraction)
    if split_at <= 0 or split_at >= n_samples:
        raise ValueError(
            f"holdout_train_fraction={cfg.holdout_train_fraction} leaves an "
            f"empty side for n_samples={n_samples}."
        )

    train_idx = np.arange(split_at)
    test_idx = np.arange(split_at, n_samples)

    logger.info(
        "Chronological holdout: train=%d rows, test=%d rows (%.0f/%.0f)",
        len(train_idx),
        len(test_idx),
        cfg.holdout_train_fraction * 100,
        (1 - cfg.holdout_train_fraction) * 100,
    )
    return train_idx, test_idx


# ======================================================================
# Fold reporting
# ======================================================================

def describe_folds(
    splits: Sequence[Split],
    y: pd.Series,
    timestamps: pd.Series | None = None,
    config: ValidationConfig | None = None,
) -> pd.DataFrame:
    """Summarise each fold's size, period and class balance.

    Produces the fold table for the paper's methods section and flags
    folds whose test block holds too few positives for AUC-style metrics
    to be stable.

    Parameters
    ----------
    splits : Sequence[tuple[np.ndarray, np.ndarray]]
        Output of :func:`rolling_origin_splits`.
    y : pd.Series
        Binary target aligned to the design matrix.
    timestamps : pd.Series, optional
        UTC timestamps aligned to the design matrix.  When given, the
        covered period of each block is reported.
    config : ValidationConfig, optional
        Supplies
        :attr:`~src.config.ValidationConfig.min_test_positives`.

    Returns
    -------
    pd.DataFrame
        One row per fold.
    """
    cfg = config or ValidationConfig()
    y_arr = np.asarray(y)

    rows = []
    for i, (train_idx, test_idx) in enumerate(splits, start=1):
        n_pos_test = int(y_arr[test_idx].sum())
        n_pos_train = int(y_arr[train_idx].sum())
        row = {
            "fold": i,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "train_positives": n_pos_train,
            "test_positives": n_pos_test,
            "train_positive_rate": float(y_arr[train_idx].mean()),
            "test_positive_rate": float(y_arr[test_idx].mean()),
            # A fold is only informative if the model had enough positives
            # to learn from AND enough to be scored against. Early folds of
            # a seasonal series routinely fail the first condition.
            "sufficient_positives": (
                n_pos_test >= cfg.min_test_positives
                and n_pos_train >= cfg.min_test_positives
            ),
        }
        if timestamps is not None:
            ts = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True))
            row["train_start"] = ts.iloc[train_idx[0]]
            row["train_end"] = ts.iloc[train_idx[-1]]
            row["test_start"] = ts.iloc[test_idx[0]]
            row["test_end"] = ts.iloc[test_idx[-1]]
        rows.append(row)

    table = pd.DataFrame(rows)

    thin = table.loc[~table["sufficient_positives"], "fold"].tolist()
    if thin:
        logger.warning(
            "Folds %s have fewer than %d positives in train or test; their "
            "F1 and AUC are unstable and must be read with that caveat.",
            thin,
            cfg.min_test_positives,
        )

    # Seasonal series drift in class balance, which makes a mean across
    # folds a poor summary. Quantify the drift rather than leaving the
    # reader to infer it from the per-fold table.
    spread = (
        table["test_positive_rate"].max() - table["test_positive_rate"].min()
    )
    if spread > 0.25:
        logger.warning(
            "Test-set positive rate ranges from %.3f to %.3f across folds "
            "(spread %.3f). The target regime is non-stationary, so folds "
            "are not exchangeable: a mean ± SD across them describes a "
            "mixture of regimes, not repeated draws from one. Report the "
            "per-fold table alongside any aggregate.",
            table["test_positive_rate"].min(),
            table["test_positive_rate"].max(),
            spread,
        )
    return table


# ======================================================================
# Guards
# ======================================================================

def assert_splits_are_ordered(splits: Sequence[Split]) -> None:
    """Assert no fold's test block precedes or overlaps its training data.

    A cheap invariant that would catch an accidental shuffle, a swapped
    return value, or a splitter replaced by one with different
    semantics — any of which would silently inflate every metric.

    Raises
    ------
    ValueError
        Naming the first offending fold.
    """
    for i, (train_idx, test_idx) in enumerate(splits, start=1):
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError(f"Fold {i} has an empty train or test block.")

        overlap = np.intersect1d(train_idx, test_idx)
        if overlap.size:
            raise ValueError(
                f"Fold {i}: train and test overlap on {overlap.size} rows "
                f"(first: {overlap[0]}). The same hour cannot be used for "
                f"both fitting and scoring."
            )
        if test_idx.min() <= train_idx.max():
            raise ValueError(
                f"Fold {i}: test block starts at row {test_idx.min()} which "
                f"is not after the last training row {train_idx.max()}. "
                f"Temporal order has been violated — every metric computed "
                f"from these folds would be optimistically biased."
            )
