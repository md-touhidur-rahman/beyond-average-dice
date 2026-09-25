from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# ============================================================
# PATHS
# ============================================================

ROOT = Path("beyond-average-dice/remedy")
BASE = ROOT / "results/embedding_trajectory"
OUT = ROOT / "figures/paper"
OUT.mkdir(parents=True, exist_ok=True)

CASE_FILE = BASE / "dose_response_case_by_alpha.csv"
TRAJ_FILE = BASE / "trajectory_analysis_1200.csv"

# Frozen-input hashes.
EXPECTED = {
    CASE_FILE:
        "23f3841742e544a668520429e9762144aa7a879dc456b7de84a2c4587ff15087",
    TRAJ_FILE:
        "83897c70552062f39f8c6d36c86eb1323b3db774b7e551226d40fe3db9fa6c46",
}

for path, expected in EXPECTED.items():
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    assert got == expected, (
        f"FROZEN INPUT HASH MISMATCH\n"
        f"{path}\nexpected={expected}\ngot={got}"
    )

case = pd.read_csv(CASE_FILE)
traj = pd.read_csv(TRAJ_FILE)

# ============================================================
# PUBLICATION STYLE
# Matches paper_final/make_final_figures.py
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
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

# Restrained, color-blind-friendly palette.
PRIMARY = "#0072B2"
SECONDARY = "#D55E00"
TERTIARY = "#009E73"
CASE_COLOR = "#9A9A9A"
ZERO_COLOR = "#555555"

alphas = np.array([0.00, 0.25, 0.50, 0.75, 1.00])

# ============================================================
# HELPERS
# ============================================================

def panel_label(ax, label):
    ax.text(
        -0.18, 1.07, label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    ax.grid(False)


def dose_panel(ax, column, ylabel, color):
    # Individual development cases.
    for _, g in case.groupby("filename", sort=False):
        g = g.sort_values("alpha")
        ax.plot(
            g["alpha"],
            g[column],
            color=CASE_COLOR,
            linewidth=0.55,
            alpha=0.55,
            zorder=1,
        )

    # Across-case mean and SEM.
    summary = (
        case.groupby("alpha")[column]
        .agg(["mean", "std", "count"])
        .reindex(alphas)
    )

    mean = summary["mean"].to_numpy()
    sem = (
        summary["std"] /
        np.sqrt(summary["count"])
    ).to_numpy()

    ax.fill_between(
        alphas,
        mean - sem,
        mean + sem,
        color=color,
        alpha=0.16,
        linewidth=0,
        zorder=2,
    )

    ax.plot(
        alphas,
        mean,
        color=color,
        linewidth=1.7,
        marker="o",
        markersize=3.8,
        markeredgewidth=0,
        zorder=3,
    )

    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks(alphas)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2g"))
    ax.set_xlabel("Embedding interpolation fraction, α")
    ax.set_ylabel(ylabel)

    clean_axis(ax)


# ============================================================
# FIGURE
# ============================================================

fig = plt.figure(figsize=(7.05, 5.0))

gs = fig.add_gridspec(
    nrows=2,
    ncols=2,
    height_ratios=[1.0, 0.92],
    hspace=0.62,
    wspace=0.38,
)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])

# ------------------------------------------------------------
# A. PRIMARY ENDPOINT
# ------------------------------------------------------------

dose_panel(
    ax1,
    "corr_distance_from_A",
    "Correlation distance from endpoint A",
    PRIMARY,
)

panel_label(ax1, "A")

# ------------------------------------------------------------
# B. SECONDARY CONTINUOUS ENDPOINT
# ------------------------------------------------------------

dose_panel(
    ax2,
    "heatmap_mae_from_A",
    "Heatmap MAE from endpoint A",
    SECONDARY,
)

panel_label(ax2, "B")

# ------------------------------------------------------------
# C. TRAJECTORY ORDERING DISTRIBUTIONS
# ------------------------------------------------------------

metrics = [
    ("corr_A_spearman", "Correlation\ndistance", PRIMARY),
    ("mae_A_spearman", "Heatmap\nMAE", SECONDARY),
    ("box_A_spearman", "Bounding-box\ninstability", TERTIARY),
]

rng = np.random.RandomState(20260923)

for xpos, (column, label, color) in enumerate(metrics, start=1):
    values = traj[column].dropna().to_numpy()

    # Boxplot gives distributional summary.
    bp = ax3.boxplot(
        [values],
        positions=[xpos],
        widths=0.46,
        patch_artist=True,
        showfliers=False,
        medianprops={
            "color": ZERO_COLOR,
            "linewidth": 1.0,
        },
        boxprops={
            "facecolor": color,
            "alpha": 0.24,
            "edgecolor": color,
            "linewidth": 0.8,
        },
        whiskerprops={
            "color": color,
            "linewidth": 0.8,
        },
        capprops={
            "color": color,
            "linewidth": 0.8,
        },
    )

    # Deterministic subsample of points for visibility only.
    # Distributional statistics still use all 1,200 trajectories.
    n_show = min(180, len(values))
    idx = rng.choice(
        len(values),
        size=n_show,
        replace=False,
    )
    shown = values[idx]

    jitter = rng.normal(
        loc=0.0,
        scale=0.045,
        size=n_show,
    )

    ax3.scatter(
        np.full(n_show, xpos) + jitter,
        shown,
        s=4.5,
        color=color,
        alpha=0.22,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )

ax3.axhline(
    0,
    color=ZERO_COLOR,
    linewidth=0.6,
    linestyle="--",
    alpha=0.7,
)

ax3.set_xlim(0.35, 3.65)
ax3.set_ylim(-0.08, 1.04)
ax3.set_xticks([1, 2, 3])
ax3.set_xticklabels(
    [
        "Correlation distance",
        "Heatmap MAE",
        "Bounding-box instability",
    ],
    rotation=0,
)
ax3.set_ylabel("Trajectory-level Spearman ρ")

clean_axis(ax3)
panel_label(ax3, "C")

# ============================================================
# EXPORT
# ============================================================

fig.subplots_adjust(
    left=0.09,
    right=0.985,
    bottom=0.10,
    top=0.97,
)

stem = OUT / "fig_embedding_interpolation_mechanism"

fig.savefig(stem.with_suffix(".pdf"))
fig.savefig(stem.with_suffix(".svg"))
fig.savefig(stem.with_suffix(".png"), dpi=600)

plt.close(fig)

print("=== FIGURE CREATED ===")
for ext in [".pdf", ".svg", ".png"]:
    p = stem.with_suffix(ext)
    print(
        p,
        p.stat().st_size,
        "bytes",
    )

print()
print("=== INPUT HASHES VERIFIED ===")
for p, expected in EXPECTED.items():
    print(expected, p)

print()
print("NOTE:")
print("Panels A/B: thin lines = 12 development cases.")
print("Thick line = across-case mean; band = case-level SEM.")
print("Panel C statistics use all 1200 trajectories.")
print("Displayed points are deterministic subsamples for legibility.")
