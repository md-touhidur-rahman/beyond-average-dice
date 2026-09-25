#!/usr/bin/env python3

"""
Publication figures for frozen ground-truth-free reliability analyses.

Scientific inputs and definitions are unchanged.
This script only controls manuscript visualization.

Requires:
    Python 3
    NumPy
    Matplotlib
"""

from pathlib import Path
import csv
import json

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

SCRIPT = Path(__file__).resolve()
REMEDY = SCRIPT.parents[1]

HELDOUT = REMEDY / "results" / "heldout_evaluation"
BRAIN600 = REMEDY / "results" / "brain600_stress"
OUT = REMEDY / "figures"
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_json(path):
    with open(path) as f:
        return json.load(f)


def arr(rows, key, dtype=float):
    return np.asarray([dtype(r[key]) for r in rows])


def precision_recall_curve_manual(y_true, score):
    """Precision-recall curve; higher score means greater risk."""
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)

    order = np.argsort(-score, kind="mergesort")
    y = y_true[order]
    s = score[order]

    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)

    distinct = np.where(np.diff(s))[0]
    idx = np.r_[distinct, y.size - 1]

    tp = tp[idx]
    fp = fp[idx]

    precision = tp / (tp + fp)
    recall = tp / np.sum(y_true)

    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]

    return precision, recall


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    ax.tick_params(
        axis="both",
        labelsize=9,
        width=0.8,
        length=3,
    )


def add_bar_labels(ax, bars, values, offset):
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{100 * value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )


# ============================================================
# Global manuscript typography
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# Load frozen Brain400 held-out data
# ============================================================

held_rows = read_csv(
    HELDOUT / "heldout_slice_results.csv"
)

held_cov = read_csv(
    HELDOUT / "heldout_risk_coverage.csv"
)

held_summary = read_json(
    HELDOUT / "heldout_summary.json"
)

held_y = arr(
    held_rows,
    "catastrophic_slice",
    int,
)

held_risk = arr(
    held_rows,
    "risk",
    float,
)

held_precision, held_recall = precision_recall_curve_manual(
    held_y,
    held_risk,
)

held_prevalence = float(np.mean(held_y))

primary = held_summary["primary"]
secondary = held_summary["secondary"]
ci = held_summary["bootstrap_ci95"]

frozen_threshold = held_summary["frozen_method"]["threshold"]


# ============================================================
# FIGURE 10
# Frozen Brain400 held-out evaluation
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(11.6, 3.45),
)

fig.subplots_adjust(
    left=0.065,
    right=0.985,
    bottom=0.22,
    top=0.86,
    wspace=0.42,
)


# ------------------------------------------------------------
# A. Precision-recall
# ------------------------------------------------------------

ax = axes[0]

ax.plot(
    held_recall,
    held_precision,
    linewidth=2.0,
)

# prevalence reference only -- no label on top of curve
ax.axhline(
    held_prevalence,
    linestyle="--",
    linewidth=1.0,
)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")

ax.set_title(
    "A   Held-out discrimination",
    loc="left",
    fontweight="bold",
    pad=9,
)

lo, hi = ci["auprc"]

# Single compact annotation in an empty region
ax.text(
    0.05,
    0.08,
    f"AUPRC = {primary['auprc']:.3f}\n"
    f"95% CI: {lo:.3f}–{hi:.3f}",
    transform=ax.transAxes,
    fontsize=9,
    ha="left",
    va="bottom",
)

style_axis(ax)


# ------------------------------------------------------------
# B. Frozen operating point
# ------------------------------------------------------------

ax = axes[1]

values = np.asarray([
    secondary["sensitivity"],
    secondary["flagged_fraction"],
    secondary["fraction_P14_failures_captured"],
])

labels = [
    "Catastrophic\nsensitivity",
    "Slices\nflagged",
    "P14 failures\ncaptured",
]

x = np.arange(len(values))

bars = ax.bar(
    x,
    values,
    width=0.56,
)

ax.set_xticks(x)
ax.set_xticklabels(labels)

# Deliberate headroom for labels
ax.set_ylim(0, 1.08)

ax.set_ylabel("Fraction")

ax.set_title(
    "B   Frozen operating point",
    loc="left",
    fontweight="bold",
    pad=9,
)

add_bar_labels(
    ax,
    bars,
    values,
    offset=0.025,
)

# Short subtitle positioned outside plotting area
ax.text(
    0.5,
    1.015,
    rf"Frozen $\tau$ = {frozen_threshold:.3f}",
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=8.5,
)

style_axis(ax)


# ------------------------------------------------------------
# C. Selective reliability
# ------------------------------------------------------------

ax = axes[2]

values = np.asarray([
    secondary["overall_P14_failure_rate_lt_.10"],
    secondary["retained_P14_failure_rate_lt_.10"],
    secondary["flagged_P14_failure_rate_lt_.10"],
])

labels = [
    "Overall",
    "Retained",
    "Flagged",
]

x = np.arange(len(values))

bars = ax.bar(
    x,
    values,
    width=0.56,
)

ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.set_ylim(0, 0.50)

ax.set_ylabel("P14 failure rate (Dice < 0.10)")

ax.set_title(
    "C   Selective reliability",
    loc="left",
    fontweight="bold",
    pad=9,
)

add_bar_labels(
    ax,
    bars,
    values,
    offset=0.012,
)

# NO extra explanatory text inside this panel.
# Catastrophic retained/flagged prevalence belongs in caption.

style_axis(ax)


fig10 = OUT / "Fig10_gtfree_heldout_reliability.pdf"

fig.savefig(
    fig10,
    format="pdf",
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Load frozen Brain600 transfer data
# ============================================================

brain_rows = read_csv(
    BRAIN600 / "brain600_stress_results.csv"
)

brain_cov = read_csv(
    BRAIN600 / "brain600_risk_coverage.csv"
)

brain_summary = read_json(
    BRAIN600 / "brain600_stress_summary.json"
)

brain_y = arr(
    brain_rows,
    "catastrophic_slice",
    int,
)

brain_risk = arr(
    brain_rows,
    "risk_1_minus_mean_pairwise_iou",
    float,
)

brain_precision, brain_recall = precision_recall_curve_manual(
    brain_y,
    brain_risk,
)

brain_prevalence = float(np.mean(brain_y))


# ============================================================
# Risk-coverage data
# ============================================================

held_coverage = arr(
    held_cov,
    "coverage",
)

held_failure = arr(
    held_cov,
    "retained_P14_failure_rate_lt_.10",
)

brain_coverage = arr(
    brain_cov,
    "coverage",
)

brain_failure = arr(
    brain_cov,
    "retained_H0_failure_rate_lt_.10",
)


# ============================================================
# Brain600 frozen-threshold quantities
# ============================================================

n = int(brain_summary["n"])
flagged_n = int(brain_summary["flagged_n"])
retained_n = n - flagged_n

flagged_cat = (
    brain_summary["tp"] / flagged_n
)

retained_cat = (
    brain_summary["fn"] / retained_n
)


# ============================================================
# FIGURE 11
# Selective reliability and no-retuning Brain600 transfer
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(11.6, 3.45),
)

fig.subplots_adjust(
    left=0.065,
    right=0.985,
    bottom=0.22,
    top=0.86,
    wspace=0.42,
)


# ------------------------------------------------------------
# A. Brain600 transfer discrimination
# ------------------------------------------------------------

ax = axes[0]

ax.plot(
    brain_recall,
    brain_precision,
    linewidth=2.0,
)

ax.axhline(
    brain_prevalence,
    linestyle="--",
    linewidth=1.0,
)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")

ax.set_title(
    "A   Brain600 transfer",
    loc="left",
    fontweight="bold",
    pad=9,
)

ax.text(
    0.05,
    0.08,
    f"AUPRC = {brain_summary['auprc']:.3f}\n"
    f"AUROC = {brain_summary['auroc']:.3f}",
    transform=ax.transAxes,
    fontsize=9,
    ha="left",
    va="bottom",
)

style_axis(ax)


# ------------------------------------------------------------
# B. Risk-coverage
# ------------------------------------------------------------

ax = axes[1]

ax.plot(
    held_coverage,
    held_failure,
    marker="o",
    markersize=4.5,
    linewidth=1.7,
    label="Brain400 held-out",
)

ax.plot(
    brain_coverage,
    brain_failure,
    marker="s",
    markersize=4.5,
    linewidth=1.7,
    label="Brain600",
)

ax.set_xlabel("Retained coverage")

ax.set_ylabel(
    "Retained failure rate\n(Dice < 0.10)"
)

ax.set_title(
    "B   Risk–coverage",
    loc="left",
    fontweight="bold",
    pad=9,
)

# High coverage on left, decreasing coverage to right
ax.set_xlim(
    max(
        np.max(held_coverage),
        np.max(brain_coverage),
    ) + 0.015,
    min(
        np.min(held_coverage),
        np.min(brain_coverage),
    ) - 0.015,
)

ax.legend(
    frameon=False,
    loc="upper left",
)

style_axis(ax)


# ------------------------------------------------------------
# C. Frozen-threshold enrichment
# ------------------------------------------------------------

ax = axes[2]

values = np.asarray([
    brain_summary["catastrophic_prevalence"],
    retained_cat,
    flagged_cat,
])

labels = [
    "Overall",
    "Retained",
    "Flagged",
]

x = np.arange(len(values))

bars = ax.bar(
    x,
    values,
    width=0.56,
)

ax.set_xticks(x)
ax.set_xticklabels(labels)

# Plenty of room above 32.5% bar.
ax.set_ylim(0, 0.40)

ax.set_ylabel("Catastrophic prevalence")

ax.set_title(
    "C   Frozen-threshold enrichment",
    loc="left",
    fontweight="bold",
    pad=9,
)

add_bar_labels(
    ax,
    bars,
    values,
    offset=0.010,
)

# IMPORTANT:
# No "27/38 captured" annotation here.
# No "no retuning" annotation here.
# Those belong in the caption.

style_axis(ax)


fig11 = OUT / "Fig11_selective_reliability_transfer.pdf"

fig.savefig(
    fig11,
    format="pdf",
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Provenance / numerical verification
# ============================================================

print()
print("============================================================")
print("PUBLICATION RISK FIGURES GENERATED")
print("============================================================")

print(f"Figure 10: {fig10}")
print(f"Figure 11: {fig11}")

print()
print("Brain400 held-out:")
print(f"  N                    = {len(held_y)}")
print(f"  catastrophic         = {int(held_y.sum())}")
print(f"  prevalence           = {held_prevalence:.12f}")
print(f"  AUPRC                = {primary['auprc']:.12f}")
print(f"  AUPRC CI             = [{lo:.12f}, {hi:.12f}]")
print(f"  frozen threshold     = {frozen_threshold:.12f}")
print(f"  sensitivity          = {secondary['sensitivity']:.12f}")
print(f"  flagged fraction     = {secondary['flagged_fraction']:.12f}")
print(
    f"  catastrophic capture = "
    f"{secondary['fraction_catastrophic_captured']:.12f}"
)
print(
    f"  P14 failure capture   = "
    f"{secondary['fraction_P14_failures_captured']:.12f}"
)
print(
    f"  retained cat prev     = "
    f"{secondary['retained_catastrophic_prevalence']:.12f}"
)
print(
    f"  flagged cat prev      = "
    f"{secondary['flagged_catastrophic_prevalence']:.12f}"
)

print()
print("Brain600 transfer:")
print(f"  N                    = {n}")
print(
    f"  catastrophic         = "
    f"{brain_summary['catastrophic_slices']}"
)
print(f"  prevalence           = {brain_prevalence:.12f}")
print(f"  AUROC                = {brain_summary['auroc']:.12f}")
print(f"  AUPRC                = {brain_summary['auprc']:.12f}")
print(f"  flagged              = {flagged_n}")
print(f"  retained             = {retained_n}")
print(
    f"  captured             = "
    f"{brain_summary['catastrophic_captured']}"
)
print(f"  flagged cat prev     = {flagged_cat:.12f}")
print(f"  retained cat prev    = {retained_cat:.12f}")

print()
print("Scientific inputs unchanged.")
print("Only manuscript visualization was revised.")
print("============================================================")
