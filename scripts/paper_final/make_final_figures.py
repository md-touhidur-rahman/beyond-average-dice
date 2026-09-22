#!/usr/bin/env python3

from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter


# ============================================================
# PATHS
# ============================================================

ROOT = Path("/home/hpc/rlvl/rlvl178v/prl_medclipsam")

B600 = ROOT / "beyond-average-dice/results/brain600_language_ladder"
B400 = ROOT / "beyond-average-dice/results/brain_p20"

OUT = ROOT / "beyond-average-dice/paper_results/final_publication"
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# PUBLICATION STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.0,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.2,

    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,

    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "figure.facecolor": "white",
    "axes.facecolor": "white",

    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})


# Restrained palette.
# Colors encode experimental groups, not decoration.
COL_MINOR = "#4C78A8"
COL_ABLATION = "#E45756"
COL_NEUTRAL = "#6B6B6B"
COL_LIGHT = "#B8B8B8"
COL_DARK = "#222222"

COND_ORDER = ["H1", "H2", "L3", "L4", "L5"]

COND_LABELS = {
    "H1": "H1\nverb",
    "H2": "H2\nintro",
    "L3": "L3\nreform.",
    "L4": "L4\ncompress.",
    "L5": "L5\ndetail ablation",
}


# ============================================================
# HELPERS
# ============================================================

def panel_label(ax, label):
    ax.text(
        -0.14, 1.05, label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, stem):
    pdf = OUT / f"{stem}.pdf"
    png = OUT / f"{stem}.png"

    fig.savefig(pdf)
    fig.savefig(png, dpi=600)

    plt.close(fig)

    print("WROTE:", pdf)
    print("WROTE:", png)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Required source file missing:\n{path}")
    return path


# ============================================================
# FIGURE 1
#
# Controlled Brain600 language experiment
#
# Panel A:
# H0-relative mean absolute Dice change + bootstrap CI.
#
# Panel B:
# fraction of cases exceeding absolute-change thresholds.
#
# Important:
# Conditions are categorical.
# NO connecting line is drawn.
# ============================================================

def make_brain600_controlled():

    summary_path = require_file(B600 / "h0_relative_summary.csv")
    ci_path = require_file(B600 / "bootstrap_95ci.csv")

    summary = pd.read_csv(summary_path)
    ci = pd.read_csv(ci_path)

    print("\nBrain600 summary columns:")
    print(summary.columns.tolist())

    print("\nBrain600 bootstrap columns:")
    print(ci.columns.tolist())

    # --------------------------------------------------------
    # Resolve condition column
    # --------------------------------------------------------

    def find_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        raise RuntimeError(
            "Could not find any of these columns:\n"
            f"{candidates}\n"
            f"Available columns:\n{df.columns.tolist()}"
        )

    cond_col = find_col(
        summary,
        ["condition", "cond", "prompt_condition"]
    )

    mean_abs_col = find_col(
        summary,
        [
            "mean_abs_delta",
            "mean_abs_change",
            "mean_absolute_change",
            "mean_abs_dice_change",
        ]
    )

    gt10_col = find_col(
        summary,
        [
            "prop_abs_gt_0.10",
            "prop_abs_gt_10",
            "prop_gt_0.10",
            "abs_gt_0.10",
            "frac_abs_gt_0.10",
        ]
    )

    gt20_col = find_col(
        summary,
        [
            "prop_abs_gt_0.20",
            "prop_abs_gt_20",
            "prop_gt_0.20",
            "abs_gt_0.20",
            "frac_abs_gt_0.20",
        ]
    )

    gt50_col = find_col(
        summary,
        [
            "prop_abs_gt_0.50",
            "prop_abs_gt_50",
            "prop_gt_0.50",
            "abs_gt_0.50",
            "frac_abs_gt_0.50",
        ]
    )

    # --------------------------------------------------------
    # CI table
    # --------------------------------------------------------

    ci_cond_col = find_col(
        ci,
        ["condition", "cond", "prompt_condition"]
    )

    metric_col = find_col(
        ci,
        ["metric", "measure", "statistic"]
    )

    low_col = find_col(
        ci,
        ["ci_low", "low", "lower", "lower_95"]
    )

    high_col = find_col(
        ci,
        ["ci_high", "high", "upper", "upper_95"]
    )

    # --------------------------------------------------------
    # Filter exact conditions
    # --------------------------------------------------------

    summary[cond_col] = summary[cond_col].astype(str)
    ci[ci_cond_col] = ci[ci_cond_col].astype(str)

    sub = (
        summary[summary[cond_col].isin(COND_ORDER)]
        .set_index(cond_col)
        .loc[COND_ORDER]
        .reset_index()
    )

    # Identify mean-absolute CI rows robustly.
    ci_metric = ci[metric_col].astype(str).str.lower()

    mask = (
        ci_metric.str.contains("mean")
        & ci_metric.str.contains("abs")
    )

    ci_abs = ci[mask].copy()

    if len(ci_abs) != len(COND_ORDER):
        print("\nCandidate CI rows:")
        print(ci[[ci_cond_col, metric_col, low_col, high_col]].to_string(index=False))
        raise RuntimeError(
            "Expected exactly five mean-absolute-change CI rows "
            f"for H1/H2/L3/L4/L5; found {len(ci_abs)}."
        )

    ci_abs = (
        ci_abs[ci_abs[ci_cond_col].isin(COND_ORDER)]
        .set_index(ci_cond_col)
        .loc[COND_ORDER]
        .reset_index()
    )

    means = sub[mean_abs_col].to_numpy(float)
    lows = ci_abs[low_col].to_numpy(float)
    highs = ci_abs[high_col].to_numpy(float)

    if not np.all((lows <= means) & (means <= highs)):
        raise RuntimeError("Bootstrap CI does not contain point estimate.")

    # Frozen-value integrity checks.
    expected = np.array([
        0.020296,
        0.029997,
        0.033596,
        0.080111,
        0.072436,
    ])

    if not np.allclose(means, expected, atol=5e-6):
        raise RuntimeError(
            "Brain600 mean absolute changes do not match frozen record.\n"
            f"Observed: {means}\n"
            f"Expected: {expected}"
        )

    # --------------------------------------------------------
    # Export exact plotting data
    # --------------------------------------------------------

    plotdata = pd.DataFrame({
        "condition": COND_ORDER,
        "mean_abs_delta_dice": means,
        "ci_low": lows,
        "ci_high": highs,
        "prop_abs_delta_gt_0.10": sub[gt10_col].to_numpy(float),
        "prop_abs_delta_gt_0.20": sub[gt20_col].to_numpy(float),
        "prop_abs_delta_gt_0.50": sub[gt50_col].to_numpy(float),
    })

    plotdata.to_csv(
        OUT / "fig06_brain600_controlled_language_data.csv",
        index=False
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1, 2,
        figsize=(7.05, 2.65),
        gridspec_kw={"wspace": 0.32}
    )

    # ---------------- A ----------------

    ax = axes[0]
    x = np.arange(len(COND_ORDER))

    colors = [
        COL_MINOR,
        COL_MINOR,
        COL_MINOR,
        COL_ABLATION,
        COL_ABLATION,
    ]

    yerr = np.vstack([
        means - lows,
        highs - means
    ])

    ax.bar(
        x,
        means,
        width=0.68,
        color=colors,
        edgecolor=COL_DARK,
        linewidth=0.45,
        zorder=2,
    )

    ax.errorbar(
        x,
        means,
        yerr=yerr,
        fmt="none",
        ecolor=COL_DARK,
        elinewidth=0.8,
        capsize=2.3,
        capthick=0.8,
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABELS[c] for c in COND_ORDER])

    ax.set_ylabel(r"Mean $|\Delta \mathrm{Dice}|$ vs H0")
    ax.set_ylim(0, max(highs) * 1.18)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))

    clean_axis(ax)
    panel_label(ax, "a")

    # ---------------- B ----------------

    ax = axes[1]

    thresholds = [
        ("|ΔDice| > 0.10", "prop_abs_delta_gt_0.10"),
        ("|ΔDice| > 0.20", "prop_abs_delta_gt_0.20"),
        ("|ΔDice| > 0.50", "prop_abs_delta_gt_0.50"),
    ]

    width = 0.22
    offsets = [-width, 0, width]

    for (label, col), off in zip(thresholds, offsets):
        vals = plotdata[col].to_numpy(float)

        ax.bar(
            x + off,
            vals,
            width=width,
            label=label,
            edgecolor=COL_DARK,
            linewidth=0.35,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABELS[c] for c in COND_ORDER])
    ax.set_ylabel("Cases exceeding threshold")
    ax.set_ylim(0, 0.23)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))

    ax.legend(
        frameon=False,
        loc="upper left",
        handlelength=1.2,
        borderaxespad=0,
    )

    clean_axis(ax)
    panel_label(ax, "b")

    save(fig, "fig06_brain600_controlled_language")


# ============================================================
# FIGURE 2
#
# Brain400 catastrophic failure component decomposition
#
# Panel A:
# localization-IoU distribution among failure-success pairs.
#
# Panel B:
# selected-mask divergence vs retrospective candidate-oracle
# divergence by localization stratum.
#
# This is descriptive decomposition, NOT causal mediation.
# ============================================================

def make_component_decomposition():

    dist_path = require_file(
        B400 / "failure_success_localization_distribution.csv"
    )

    decomp_path = require_file(
        B400 / "failure_success_pair_decomposition.csv"
    )

    dist = pd.read_csv(dist_path)
    decomp = pd.read_csv(decomp_path)

    print("\nFailure-success distribution columns:")
    print(dist.columns.tolist())

    print("\nFailure-success decomposition columns:")
    print(decomp.columns.tolist())

    def find_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        raise RuntimeError(
            f"Could not resolve {candidates}\n"
            f"Columns: {df.columns.tolist()}"
        )

    stratum_dist = find_col(
        dist,
        ["stratum", "box_iou_stratum", "iou_stratum"]
    )

    n_col = find_col(
        dist,
        ["n", "count", "N"]
    )

    stratum_dec = find_col(
        decomp,
        ["stratum", "box_iou_stratum", "iou_stratum"]
    )

    selected_col = find_col(
        decomp,
        [
            "selected_mean_abs_delta",
            "mean_selected_abs_delta",
            "selected_mean_abs_ddice",
        ]
    )

    oracle_col = find_col(
        decomp,
        [
            "oracle_mean_abs_delta",
            "mean_oracle_abs_delta",
            "oracle_candidate_mean_abs_delta",
        ]
    )

    strata = [
        "<.10",
        ".10-.25",
        ".25-.50",
        ".50-.75",
        ".75-.90",
        ">=.90",
    ]

    # Normalize labels for matching.
    def norm(x):
        return (
            str(x)
            .replace(" ", "")
            .replace("0.", ".")
        )

    dist["_norm"] = dist[stratum_dist].map(norm)
    decomp["_norm"] = decomp[stratum_dec].map(norm)

    wanted = [norm(x) for x in strata]

    dist_idx = dist.set_index("_norm")
    dec_idx = decomp.set_index("_norm")

    missing_d = [x for x in wanted if x not in dist_idx.index]
    missing_c = [x for x in wanted if x not in dec_idx.index]

    if missing_d or missing_c:
        print("\nDistribution:")
        print(dist.to_string(index=False))

        print("\nDecomposition:")
        print(decomp.to_string(index=False))

        raise RuntimeError(
            f"Missing strata. distribution={missing_d}, "
            f"decomposition={missing_c}"
        )

    counts = np.array([
        float(dist_idx.loc[x, n_col])
        for x in wanted
    ])

    selected = np.array([
        float(dec_idx.loc[x, selected_col])
        for x in wanted
    ])

    oracle = np.array([
        float(dec_idx.loc[x, oracle_col])
        for x in wanted
    ])

    # Frozen integrity.
    expected_counts = np.array([
        4520, 253, 315, 190, 95, 25
    ])

    if not np.array_equal(counts.astype(int), expected_counts):
        raise RuntimeError(
            "Failure-success stratum counts disagree with frozen record.\n"
            f"Observed: {counts}\n"
            f"Expected: {expected_counts}"
        )

    if counts.sum() != 5398:
        raise RuntimeError(
            f"Expected 5,398 pairs; got {counts.sum()}"
        )

    # Export exact plotting data.
    plotdata = pd.DataFrame({
        "box_iou_stratum": strata,
        "n_pairs": counts.astype(int),
        "fraction_pairs": counts / counts.sum(),
        "selected_mean_abs_delta_dice": selected,
        "oracle_candidate_mean_abs_delta_dice": oracle,
    })

    plotdata.to_csv(
        OUT / "fig05_failure_component_decomposition_data.csv",
        index=False
    )

    fig, axes = plt.subplots(
        1, 2,
        figsize=(7.05, 2.65),
        gridspec_kw={"wspace": 0.34}
    )

    x = np.arange(len(strata))

    # ---------------- A ----------------

    ax = axes[0]

    frac = counts / counts.sum()

    bars = ax.bar(
        x,
        frac,
        width=0.68,
        color=COL_NEUTRAL,
        edgecolor=COL_DARK,
        linewidth=0.45,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(strata, rotation=30, ha="right")
    ax.set_xlabel("Box IoU stratum")
    ax.set_ylabel("Failure-success pairs")
    ax.set_ylim(0, 0.90)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))

    # Only annotate important extremes to avoid clutter.
    ax.text(
        x[0],
        frac[0] + 0.025,
        f"{frac[0]*100:.1f}%\n(n={int(counts[0]):,})",
        ha="center",
        va="bottom",
        fontsize=6.8,
    )

    ax.text(
        x[-1],
        frac[-1] + 0.025,
        f"{frac[-1]*100:.2f}%\n(n={int(counts[-1])})",
        ha="center",
        va="bottom",
        fontsize=6.8,
    )

    clean_axis(ax)
    panel_label(ax, "a")

    # ---------------- B ----------------

    ax = axes[1]

    width = 0.34

    ax.bar(
        x - width/2,
        selected,
        width=width,
        label="Selected mask",
        color=COL_NEUTRAL,
        edgecolor=COL_DARK,
        linewidth=0.4,
    )

    ax.bar(
        x + width/2,
        oracle,
        width=width,
        label="Candidate oracle",
        color=COL_LIGHT,
        edgecolor=COL_DARK,
        linewidth=0.4,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(strata, rotation=30, ha="right")
    ax.set_xlabel("Box IoU stratum")
    ax.set_ylabel(r"Mean $|\Delta \mathrm{Dice}|$")
    ax.set_ylim(0, 0.90)

    ax.legend(
        frameon=False,
        loc="upper right",
        handlelength=1.2,
    )

    # Explicit rare-stratum denominator.
    ax.text(
        0.98,
        0.04,
        r"$\geq$0.90: 25 pairs" + "\n6 images / 6 patients",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.8,
    )

    clean_axis(ax)
    panel_label(ax, "b")

    save(fig, "fig05_failure_component_decomposition")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("FINAL PUBLICATION FIGURE GENERATION")
    print("=" * 70)

    make_brain600_controlled()
    make_component_decomposition()

    print("\nDONE")
    print("Output directory:")
    print(OUT)
