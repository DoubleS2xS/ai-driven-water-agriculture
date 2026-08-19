"""Publication figures for the manuscript.

Generates every figure to a single enforced style, derived from the
supplied journal guidance:

* no outer frame — only the left and bottom axis lines are drawn
* no legend box outline
* no figure titles — the caption carries the description
* circular markers on every curve
* one font family and one size scheme across all figures
* no bold text anywhere inside a figure
* no capital letters except at the start of a label
* vector output (PDF) plus a 600 dpi raster preview

Run as ``python -m src.figures``.  Output lands in
``data/outputs/figures/``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Enforced style ────────────────────────────────────────────────────

FONT = "DejaVu Sans"
FS_LABEL = 9
FS_TICK = 8
FS_ANNOT = 7.5
LINEWIDTH = 1.3
MARKERSIZE = 4.0
AXIS_LW = 0.8

#: One column of a two-column layout, in inches.
W_SINGLE = 3.5
#: Full text width across two columns.
W_DOUBLE = 7.16

OUT = Path("data/outputs/figures")

#: Colour-blind-safe qualitative palette, consistent across all figures.
PALETTE: Dict[str, str] = {
    "moisture_threshold": "#0072B2",
    "logistic": "#D55E00",
    "lightgbm": "#009E73",
    "xgboost": "#CC79A7",
    "persistence": "#56B4E9",
    "majority": "#999999",
}

#: Display names — lower case, per the guidance.
LABEL: Dict[str, str] = {
    "moisture_threshold": "moisture threshold",
    "logistic": "logistic regression",
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
    "persistence": "persistence",
    "majority": "majority class",
}

ORDER: List[str] = [
    "moisture_threshold", "logistic", "lightgbm",
    "xgboost", "persistence", "majority",
]


def apply_style() -> None:
    """Install the global rcParams shared by every figure."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT],
        "font.size": FS_LABEL,
        "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_ANNOT,
        "axes.linewidth": AXIS_LW,
        "lines.linewidth": LINEWIDTH,
        "lines.markersize": MARKERSIZE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,          # no legend box
        "figure.autolayout": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,               # embed as TrueType, editable
        "ps.fonttype": 42,
        "text.usetex": False,
    })


def bare(ax: plt.Axes) -> plt.Axes:
    """Strip an axes to X and Y lines only."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(AXIS_LW)
    ax.tick_params(width=AXIS_LW, length=3)
    ax.set_title("")
    return ax


def blank(ax: plt.Axes) -> plt.Axes:
    """Remove every axis decoration — for schematic figures."""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("")
    return ax


def save(fig: plt.Figure, name: str) -> None:
    """Write vector and raster copies."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=600)
    plt.close(fig)
    logger.info("wrote %s", name)


# ══════════════════════════════════════════════════════════════════════
# Figure 1 — landscape of prediction targets and validation rigour
# ══════════════════════════════════════════════════════════════════════

def figure_01() -> None:
    """Map prior work onto target discreteness and validation rigour."""
    # x: prediction target, 0 = soil state … 3 = irrigation event
    # y: validation rigour, 0 = single split … 2 = rolling origin
    works = [
        # (label, x, y, uses_xai)
        ("[35]", 0.0, 0.6, False),
        ("[43]", 0.0, 0.4, False),
        ("[44]", 0.0, 0.5, False),
        ("[40]", 0.0, 0.9, False),
        ("[41]", 0.0, 0.8, False),
        ("[42]", 0.0, 0.7, False),
        ("[47]", 0.0, 0.3, False),
        ("[50]", 0.0, 0.2, False),
        ("[48]", 0.15, 1.9, False),
        ("[51]", 1.0, 0.8, True),
        ("[55]", 1.0, 1.1, True),
        ("[45]", 1.1, 0.6, False),
        ("[36]", 2.0, 1.0, False),
        ("[37]", 2.0, 0.7, False),
        ("[54]", 2.05, 0.35, True),
        ("[38]", 2.8, 0.9, False),
        ("[39]", 2.75, 0.5, False),
    ]
    fig, ax = plt.subplots(figsize=(W_DOUBLE, 3.3))
    bare(ax)

    for label, x, y, xai in works:
        ax.scatter(
            x, y, s=46,
            facecolor=PALETTE["persistence"] if xai else "white",
            edgecolor=PALETTE["logistic"] if xai else "#555555",
            linewidth=1.0, zorder=3,
        )
        ax.annotate(
            label, (x, y), xytext=(0, 7), textcoords="offset points",
            ha="center", fontsize=FS_ANNOT, color="#333333",
        )

    ax.scatter(
        3.0, 2.0, s=140, marker="*",
        facecolor=PALETTE["moisture_threshold"],
        edgecolor=PALETTE["moisture_threshold"], zorder=4,
    )
    ax.annotate(
        "this study", (3.0, 2.0), xytext=(-6, 10),
        textcoords="offset points", ha="right",
        fontsize=FS_ANNOT, color=PALETTE["moisture_threshold"],
    )

    ax.set_xlim(-0.35, 3.35)
    ax.set_ylim(-0.15, 2.35)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([
        "soil state", "crop coefficient\nor stress index",
        "irrigation volume\nor depth", "irrigation event",
    ])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["single split", "holdout", "rolling origin"])
    ax.set_xlabel("prediction target")
    ax.set_ylabel("validation protocol")

    # legend without a box
    ax.scatter([], [], s=46, facecolor="white", edgecolor="#555555",
               label="no attribution analysis")
    ax.scatter([], [], s=46, facecolor=PALETTE["persistence"],
               edgecolor=PALETTE["logistic"], label="attribution applied")
    ax.legend(loc="upper left", handletextpad=0.4, borderpad=0.0)

    save(fig, "figure_01_landscape")


# ══════════════════════════════════════════════════════════════════════
# Figure 2 — causal structure of the irrigation decision
# ══════════════════════════════════════════════════════════════════════

def _box(ax, x, y, w, h, text, fc="white", ec="#555555", fs=FS_ANNOT):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                           linewidth=0.9, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, zorder=3)


def _arrow(ax, x1, y1, x2, y2, color="#555555", style="-|>", dashed=False):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=9,
        linewidth=0.9, color=color, zorder=1,
        linestyle="--" if dashed else "-",
        shrinkA=1, shrinkB=1,
    ))


def figure_02() -> None:
    """Physical causal chain and two alternative feature designs."""
    fig, ax = plt.subplots(figsize=(W_DOUBLE, 3.2))
    blank(ax)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.6)

    # physical chain
    ax.text(0.05, 5.15, "physical process", fontsize=FS_LABEL, va="bottom")
    chain = [
        (0.2, "operator\ndecision"), (2.4, "valve\nopens"),
        (4.6, "volume\ndelivered"), (6.8, "soil moisture\nrises"),
    ]
    for x, t in chain:
        _box(ax, x, 4.20, 1.9, 0.80, t)
    for x in (2.1, 4.3, 6.5):
        _arrow(ax, x, 4.60, x + 0.3, 4.60)

    # synchronous design
    ax.text(0.05, 3.62, "synchronous design — inference from consequence",
            fontsize=FS_LABEL, va="bottom", color=PALETTE["xgboost"])
    _box(ax, 6.8, 2.62, 1.9, 0.72, "moisture (t)", ec=PALETTE["xgboost"])
    _box(ax, 0.2, 2.62, 1.9, 0.72, "event (t)", ec=PALETTE["xgboost"])
    _arrow(ax, 6.75, 2.98, 2.2, 2.98, color=PALETTE["xgboost"])
    _arrow(ax, 7.75, 4.16, 7.75, 3.40, color=PALETTE["xgboost"], dashed=True)
    ax.text(4.5, 3.12, "predicts", fontsize=FS_ANNOT,
            color=PALETTE["xgboost"], ha="center")

    # causal design
    ax.text(0.05, 1.92, "causal design used here",
            fontsize=FS_LABEL, va="bottom",
            color=PALETTE["moisture_threshold"])
    for i, lag in enumerate(("t − 1", "t − 2", "t − 3", "…", "t − 24")):
        _box(ax, 4.30 + i * 1.14, 0.92, 0.98, 0.70, f"moisture\n({lag})",
             ec=PALETTE["moisture_threshold"], fs=6.6)
    _box(ax, 0.2, 0.92, 1.9, 0.70, "event (t)",
         ec=PALETTE["moisture_threshold"])
    _arrow(ax, 4.25, 1.27, 2.2, 1.27, color=PALETTE["moisture_threshold"])
    ax.text(3.22, 1.45, "predicts", fontsize=FS_ANNOT,
            color=PALETTE["moisture_threshold"], ha="center")

    # excluded — placed clear of every box and arrow
    ax.text(0.2, 0.30, "excluded at every lag:   moisture (t),   flow (t − k)",
            fontsize=FS_ANNOT, color="#8a8a8a", ha="left", va="center")

    save(fig, "figure_02_causal_structure")


# ══════════════════════════════════════════════════════════════════════
# Figure 3 — data pipeline and validation protocol
# ══════════════════════════════════════════════════════════════════════

def figure_03() -> None:
    """Pipeline with reconciliation, causal features and fold-internal fitting."""
    fig, ax = plt.subplots(figsize=(W_DOUBLE, 4.1))
    blank(ax)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)

    src = [(0.15, 5.3, "moisture\nprobe"), (2.05, 5.3, "relay\nstate"),
           (3.95, 5.3, "flow\nmeter"), (5.85, 5.3, "reanalysis\nweather")]
    for x, y, t in src:
        _box(ax, x, y, 1.7, 0.72, t)
        _arrow(ax, x + 0.85, y, x + 0.85, y - 0.42)

    _box(ax, 0.15, 4.15, 7.4, 0.72,
         "timezone reconciliation  →  utc", ec=PALETTE["logistic"])
    _box(ax, 7.85, 4.15, 2.0, 0.72,
         "diurnal check\nraise if peak off", ec=PALETTE["xgboost"], fs=6.8)
    _arrow(ax, 7.55, 4.51, 7.85, 4.51, color=PALETTE["xgboost"])
    _arrow(ax, 3.85, 4.15, 3.85, 3.73)

    _box(ax, 0.15, 3.0, 7.4, 0.72,
         "hourly resample  —  mean, max for valve, last for flow")
    _arrow(ax, 3.85, 3.0, 3.85, 2.58)

    _box(ax, 0.15, 1.85, 7.4, 0.72,
         "causal features  —  shift by one row, then lag, diff, window",
         ec=PALETTE["moisture_threshold"])
    _box(ax, 7.85, 1.85, 2.0, 0.72,
         "two look-ahead\ntests", ec=PALETTE["xgboost"], fs=6.8)
    _arrow(ax, 7.55, 2.21, 7.85, 2.21, color=PALETTE["xgboost"])
    _arrow(ax, 3.85, 1.85, 3.85, 1.43)

    _box(ax, 0.15, 0.7, 7.4, 0.72,
         "rolling-origin folds  —  preprocessing fitted inside train block",
         ec=PALETTE["lightgbm"])

    ax.text(9.9, 0.35, "verification gate", fontsize=FS_ANNOT,
            color=PALETTE["xgboost"], ha="right")

    save(fig, "figure_03_pipeline")


# ══════════════════════════════════════════════════════════════════════
# Figure 4 — moisture trace, episodes and fold boundaries
# ══════════════════════════════════════════════════════════════════════

def figure_04() -> None:
    """Soil moisture with irrigation episodes and fold boundaries."""
    df = pd.read_csv("data/processed/merged_hourly.csv",
                     parse_dates=["timestamp"])
    folds = pd.read_csv("data/outputs/folds.csv",
                        parse_dates=["test_start", "test_end"])

    fig, ax = plt.subplots(figsize=(W_DOUBLE, 2.8))
    bare(ax)

    ax.plot(df.timestamp, df.soil_moisture, color="#444444",
            linewidth=0.9, zorder=2)

    on = df.irrigation_event.fillna(0).to_numpy() > 0
    lo, hi = float(np.nanmin(df.soil_moisture)), float(np.nanmax(df.soil_moisture))
    band = lo - 1.2
    ax.fill_between(df.timestamp, band, band + 0.6, where=on,
                    color=PALETTE["moisture_threshold"], linewidth=0,
                    zorder=3)

    for _, f in folds.iterrows():
        ax.axvline(f.test_start, color="#bbbbbb", linewidth=0.7,
                   linestyle="--", zorder=1)
        mid = f.test_start + (f.test_end - f.test_start) / 2
        ax.annotate(f"fold {int(f.fold)}", (mid, hi + 0.4),
                    ha="center", fontsize=FS_ANNOT, color="#666666")

    # annotate the dominant episode
    ax.annotate(
        "117 h episode", xy=(pd.Timestamp("2022-08-30 12:00"), band + 0.3),
        xytext=(pd.Timestamp("2022-08-05"), lo + 1.0),
        fontsize=FS_ANNOT, color=PALETTE["logistic"],
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["logistic"],
                        linewidth=0.8, shrinkA=2, shrinkB=2),
    )

    ax.set_ylim(band - 0.5, hi + 1.3)
    ax.set_xlabel("date, utc")
    ax.set_ylabel("soil moisture, %")
    ax.text(df.timestamp.iloc[8], band + 0.15, "irrigating hours",
            fontsize=FS_ANNOT, color=PALETTE["moisture_threshold"],
            va="center")

    save(fig, "figure_04_moisture_episodes")


# ══════════════════════════════════════════════════════════════════════
# Figure 5 — precision-recall performance by fold
# ══════════════════════════════════════════════════════════════════════

def figure_05() -> None:
    """PR-AUC by fold for all models, with the no-skill reference."""
    p = pd.read_csv("data/outputs/per_fold_metrics.csv")
    m = p[p.protocol == "main"]
    folds = pd.read_csv("data/outputs/folds.csv")

    fig, ax = plt.subplots(figsize=(W_DOUBLE, 3.0))
    bare(ax)

    # The majority-class curve coincides with the no-skill reference by
    # construction — average precision of a constant predictor equals the
    # positive rate — so it is drawn once, dashed, and labelled as both.
    for name in ORDER:
        s = m[m.model == name].sort_values("fold")
        if s.empty:
            continue
        is_ref = name == "majority"
        ax.plot(s.fold, s.pr_auc, marker="o", color=PALETTE[name],
                label="majority class, equals no skill" if is_ref
                else LABEL[name],
                linestyle="--" if is_ref else "-",
                markerfacecolor="white", markeredgewidth=1.1)

    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("fold")
    ax.set_ylabel("area under precision-recall curve")
    ax.set_ylim(-0.03, 1.06)
    ax.legend(ncol=3, loc="lower right", handletextpad=0.5,
              columnspacing=1.2, borderpad=0.0)

    save(fig, "figure_05_pr_auc_by_fold")


# ══════════════════════════════════════════════════════════════════════
# Figure 6 — recall against decision threshold, onset task
# ══════════════════════════════════════════════════════════════════════

def _onset_fold_predictions() -> Tuple[List[np.ndarray],
                                       Dict[str, List[np.ndarray]],
                                       List[float]]:
    """Return per-fold onset labels, probabilities and rule-based recall.

    Aggregation is per fold and then averaged, matching Table 6 exactly.
    Pooling predictions across folds would give different numbers,
    because folds carry very different positive rates.

    The moisture-threshold baseline is handled separately: its
    ``predict_proba`` is a monotone transform of distance from the fitted
    cut-point, not a calibrated probability, so sweeping a threshold over
    it does not correspond to its decision rule.  Its recall is therefore
    taken from ``predict`` and reported as a single operating point.
    """
    from src.onset import build_onset_design_matrix
    from src.validation import rolling_origin_splits
    from src.evaluate_pipeline import make_model_factory

    df = pd.read_csv("data/processed/merged_hourly.csv",
                     parse_dates=["timestamp"])
    X, y, _timestamps, _blocks = build_onset_design_matrix(df)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    splits = rolling_origin_splits(len(X))

    y_true: List[np.ndarray] = []
    probs: Dict[str, List[np.ndarray]] = {
        k: [] for k in ("logistic", "xgboost", "lightgbm")
    }
    rule_recall: List[float] = []

    for split in splits:
        tr, te = (np.asarray(split[0]), np.asarray(split[1]))
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        Xte, yte = X.iloc[te], y.iloc[te]
        yte_arr = yte.to_numpy()
        y_true.append(yte_arr)

        rule = make_model_factory("moisture_threshold")(42)
        rule.fit(Xtr, ytr)
        pred = np.asarray(rule.predict(Xte))
        rule_recall.append(
            float(((pred == 1) & (yte_arr == 1)).sum()
                  / max(yte_arr.sum(), 1))
        )

        for name in ("logistic", "xgboost", "lightgbm"):
            est = make_model_factory(name)(42)
            est.fit(Xtr, ytr)
            probs[name].append(np.asarray(est.predict_proba(Xte))[:, 1])

    return y_true, probs, rule_recall


def figure_06() -> None:
    """Recall against decision threshold for the learned models.

    The moisture-threshold rule appears as a horizontal reference at its
    own operating point, since it exposes no tunable probability.
    """
    y_true, probs, rule_recall = _onset_fold_predictions()
    grid = np.linspace(0.01, 0.99, 99)

    def mean_recall(model: str, t: float) -> float:
        vals = [
            float(((p >= t) & (y == 1)).sum() / max(y.sum(), 1))
            for p, y in zip(probs[model], y_true)
        ]
        return float(np.mean(vals))

    fig, ax = plt.subplots(figsize=(W_DOUBLE, 3.1))
    bare(ax)

    rule = float(np.mean(rule_recall))
    ax.axhline(rule, color=PALETTE["moisture_threshold"],
               linewidth=LINEWIDTH, linestyle="-", zorder=2)
    ax.annotate(
        f"moisture threshold, fixed rule — {rule:.2f}",
        (0.015, rule), xytext=(0, 6), textcoords="offset points",
        fontsize=FS_ANNOT, color=PALETTE["moisture_threshold"],
    )

    offsets = {"xgboost": 9, "logistic": -2, "lightgbm": -13}
    for name in ("xgboost", "logistic", "lightgbm"):
        rec = [mean_recall(name, t) for t in grid]
        ax.plot(grid, rec, color=PALETTE[name], label=LABEL[name],
                marker="o", markevery=14, markerfacecolor="white",
                markeredgewidth=1.1, zorder=3)
        at_half = mean_recall(name, 0.5)
        ax.scatter([0.5], [at_half], s=32, color=PALETTE[name], zorder=5)
        ax.annotate(f"{at_half:.2f}", (0.5, at_half),
                    xytext=(7, offsets[name]), textcoords="offset points",
                    fontsize=FS_ANNOT, color=PALETTE[name], va="center")

    ax.axvline(0.5, color="#bbbbbb", linewidth=0.7, linestyle="--", zorder=1)
    ax.annotate("default operating point", (0.5, 1.03), xytext=(5, 0),
                textcoords="offset points", fontsize=FS_ANNOT,
                color="#666666", va="center")

    ax.set_xlabel("decision threshold")
    ax.set_ylabel("recall on irrigation onset")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.03, 1.10)
    ax.legend(loc="center right", handletextpad=0.5, borderpad=0.0)

    save(fig, "figure_06_recall_vs_threshold")


# ══════════════════════════════════════════════════════════════════════
# Figure 7 — feature ablation
# ══════════════════════════════════════════════════════════════════════

def figure_07() -> None:
    """PR-AUC by feature set for both estimators."""
    a = pd.read_csv("data/outputs/ablation.csv")
    sets = ["A", "B", "C", "D", "E"]
    desc = ["a\nmoisture lags", "b\n+ weather", "c\n+ calendar",
            "d\n+ valve lags", "e\nweather only"]

    fig, ax = plt.subplots(figsize=(W_DOUBLE, 2.9))
    bare(ax)

    x = np.arange(len(sets))
    width = 0.34
    for i, (model, colour) in enumerate([
        ("xgboost", PALETTE["xgboost"]),
        ("logistic", PALETTE["logistic"]),
    ]):
        vals = [float(a[(a.feature_set == s) & (a.model == model)].pr_auc_mean.iloc[0])
                for s in sets]
        ax.bar(x + (i - 0.5) * width, vals, width, color=colour,
               edgecolor="none", label=LABEL[model])

    ax.axhline(0.2361, color="#888888", linestyle=":", linewidth=1.0)
    ax.annotate("no skill", (len(sets) - 0.55, 0.2361), xytext=(0, 4),
                textcoords="offset points", fontsize=FS_ANNOT,
                color="#666666")

    ax.set_xticks(x)
    ax.set_xticklabels(desc)
    ax.set_xlabel("feature set")
    ax.set_ylabel("area under precision-recall curve")
    ax.set_ylim(0, 0.8)
    ax.legend(loc="upper right", handletextpad=0.5, borderpad=0.0)

    save(fig, "figure_07_ablation")


# ══════════════════════════════════════════════════════════════════════
# Figure 8 — attribution ranking
# ══════════════════════════════════════════════════════════════════════

def figure_08() -> None:
    """Mean absolute attribution by feature, styled to the house rules."""
    fi = pd.read_csv("data/outputs/feature_importance.csv").head(10)
    fi = fi.iloc[::-1]

    block_colour = {
        "moisture": PALETTE["moisture_threshold"],
        "irrigation": PALETTE["logistic"],
        "weather": PALETTE["lightgbm"],
        "calendar": PALETTE["persistence"],
    }
    colours = [block_colour.get(b, "#999999") for b in fi.block]
    labels = [f.replace("_", " ") for f in fi.feature]

    fig, ax = plt.subplots(figsize=(W_DOUBLE, 3.0))
    bare(ax)

    ax.barh(np.arange(len(fi)), fi.mean_abs_shap, color=colours,
            edgecolor="none", height=0.62)
    ax.set_yticks(np.arange(len(fi)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("mean absolute attribution")
    ax.set_ylabel("")

    for b, c in block_colour.items():
        if b in set(fi.block):
            ax.barh([], [], color=c, label=b)
    ax.legend(loc="lower right", handletextpad=0.5, borderpad=0.0)

    save(fig, "figure_08_attribution")


# ══════════════════════════════════════════════════════════════════════
# Figure 9 — local attribution for the confident false alarm
# ══════════════════════════════════════════════════════════════════════

def figure_09() -> None:
    """Waterfall for the highest-probability false alarm.

    The only figure that used to come out of ``shap.waterfall_plot``, in
    the library's own fonts, colours and proportions. It is redrawn here
    through :meth:`src.models.explanation.SHAPExplainer.plot_local_decision`
    so it matches the rest of the set, and its axis labels carry the
    **training-fold medians** — without which a raw value such as
    ``soil_moisture_lag1h = 71.02`` reads as wet when it is in fact near
    the dry extreme of what that fold was trained on.

    The fold and the instance are taken from ``shap_instances.json`` so
    the figure depicts exactly the hour the manuscript names, rather than
    re-deriving a selection that could drift from it.
    """
    from src.evaluate_pipeline import (
        build_design_matrix,
        load_modeling_frame,
        make_model_factory,
    )
    from src.models.explanation import SHAPExplainer
    from src.validation import rolling_origin_splits

    manifest_path = Path("data/outputs/shap_instances.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    instance = manifest["instances"].get("confident_false_positive")
    if instance is None:
        logger.warning(
            "no confident false positive in the explained fold; "
            "figure 9 skipped"
        )
        return

    df = load_modeling_frame()
    X, y, _timestamps, _blocks = build_design_matrix(df)
    splits = rolling_origin_splits(len(X))
    train_idx, test_idx = splits[int(manifest["fold"]) - 1]

    predictor = make_model_factory(str(manifest["model"]))(
        int(manifest["seed"])
    )
    predictor.fit(X.iloc[train_idx], y.iloc[train_idx])

    OUT.mkdir(parents=True, exist_ok=True)
    SHAPExplainer(predictor.model).plot_local_decision(
        X.iloc[test_idx].reset_index(drop=True),
        index=int(instance["position"]),
        save_path=str(OUT / "figure_09_false_alarm.png"),
        dpi=600,
        training_medians=X.iloc[train_idx].median(),
    )
    logger.info("wrote figure_09_false_alarm")


# ══════════════════════════════════════════════════════════════════════
# Figure 10 — effect of validation protocol
# ══════════════════════════════════════════════════════════════════════

def figure_10() -> None:
    """Paired comparison of PR-AUC under two validation protocols."""
    single = {
        "xgboost": 0.9627, "logistic": 0.9563, "lightgbm": 0.9414,
        "moisture_threshold": 0.8016, "persistence": 0.7399,
        "majority": 0.3194,
    }
    r = pd.read_csv("data/outputs/results_summary.csv")
    rolling = (r[r.metric == "pr_auc"]
               .set_index("model")["mean"].to_dict())

    fig, ax = plt.subplots(figsize=(W_SINGLE + 0.9, 3.1))
    bare(ax)

    for name in ORDER:
        if name not in single or name not in rolling:
            continue
        ax.plot([0, 1], [single[name], rolling[name]], marker="o",
                color=PALETTE[name], markerfacecolor="white",
                markeredgewidth=1.1, label=LABEL[name])
        ax.annotate(LABEL[name], (1, rolling[name]), xytext=(6, 0),
                    textcoords="offset points", fontsize=FS_ANNOT,
                    color=PALETTE[name], va="center")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["single split", "rolling origin"])
    ax.set_xlim(-0.12, 1.75)
    ax.set_ylim(0.2, 1.02)
    ax.set_ylabel("area under precision-recall curve")
    ax.set_xlabel("validation protocol")

    save(fig, "figure_10_protocol_effect")


# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    apply_style()
    for fn in (figure_01, figure_02, figure_03, figure_04,
               figure_05, figure_06, figure_07, figure_08, figure_09,
               figure_10):
        fn()
    logger.info("figures written to %s", OUT)


if __name__ == "__main__":
    main()
