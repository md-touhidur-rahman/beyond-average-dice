from pathlib import Path
import csv
import hashlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# ============================================================
# PATHS + FROZEN INPUT VERIFICATION
# ============================================================

ROOT = Path("beyond-average-dice/remedy")
BASE = ROOT / "results/embedding_trajectory"
OUT = ROOT / "figures/paper"
OUT.mkdir(parents=True, exist_ok=True)

CASE_FILE = BASE / "dose_response_case_by_alpha.csv"
TRAJ_FILE = BASE / "trajectory_analysis_1200.csv"

EXPECTED = {
    CASE_FILE:
        "23f3841742e544a668520429e9762144aa7a879dc456b7de84a2c4587ff15087",
    TRAJ_FILE:
        "83897c70552062f39f8c6d36c86eb1323b3db774b7e551226d40fe3db9fa6c46",
}

for path, expected in EXPECTED.items():
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != expected:
        raise RuntimeError(
            f"FROZEN INPUT HASH MISMATCH\n"
            f"{path}\nexpected={expected}\ngot={got}"
        )

# ============================================================
# READ CSV — NO PANDAS
# ============================================================

def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

case = read_csv(CASE_FILE)
traj = read_csv(TRAJ_FILE)

alphas = np.array([0.0, 0.25, 0.50, 0.75, 1.0])

# Unique development cases
filenames = sorted(set(r["filename"] for r in case))

# ============================================================
# STYLE
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
    "savefig.pad_inches": 0.04,
})

PRIMARY = "#0072B2"
SECONDARY = "#D55E00"
TERTIARY = "#009E73"
CASE_COLOR = "#9A9A9A"
ZERO_COLOR = "#555555"

# ============================================================
# HELPERS
# ============================================================

def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")


def panel_label(ax, label):
    ax.text(
        -0.12, 1.06, label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )


def case_values(column):
    """
    Return one 5-dose trajectory per development case.
    """
    curves = []

    for fn in filenames:
        rows = [r for r in case if r["filename"] == fn]

        vals = []
        ok = True

        for a in alphas:
            matches = [
                r for r in rows
                if abs(float(r["alpha"]) - a) < 1e-9
            ]

            if len(matches) != 1:
                ok = False
                break

            vals.append(float(matches[0][column]))

        if ok:
            curves.append(vals)

    return np.asarray(curves, dtype=float)


def dose_panel(ax, column, ylabel, color):
    Y = case_values(column)

    # Individual development cases
    for row in Y:
        ax.plot(
            alphas,
            row,
            color=CASE_COLOR,
            linewidth=0.65,
            alpha=0.45,
            zorder=1,
        )

    # Across-case mean and case-level SEM
    mean = np.mean(Y, axis=0)
    sem = np.std(Y, axis=0, ddof=1) / np.sqrt(Y.shape[0])

    ax.fill_between(
        alphas,
        mean - sem,
        mean + sem,
        color=color,
        alpha=0.15,
        linewidth=0,
        zorder=2,
    )

    ax.plot(
        alphas,
        mean,
        color=color,
        linewidth=1.8,
        marker="o",
        markersize=4.0,
        markeredgewidth=0,
        zorder=3,
    )

    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks(alphas)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2g"))

    ax.set_xlabel("Embedding interpolation fraction, α")
    ax.set_ylabel(ylabel)

    clean_axis(ax)

    return Y


# ============================================================
# FIGURE
#
# A/B top row
# C full-width bottom row
# ============================================================

fig = plt.figure(figsize=(7.05, 5.05))

gs = fig.add_gridspec(
    2, 2,
    height_ratios=[1.0, 0.78],
    hspace=0.42,
    wspace=0.38,
)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])

# ------------------------------------------------------------
# A. PRIMARY CONTINUOUS ENDPOINT
# ------------------------------------------------------------

Y1 = dose_panel(
    ax1,
    "corr_distance_from_A",
    "Correlation distance from endpoint A",
    PRIMARY,
)

panel_label(ax1, "A")

# ------------------------------------------------------------
# B. SECONDARY CONTINUOUS ENDPOINT
# ------------------------------------------------------------

Y2 = dose_panel(
    ax2,
    "heatmap_mae_from_A",
    "Heatmap MAE from endpoint A",
    SECONDARY,
)

panel_label(ax2, "B")

# ------------------------------------------------------------
# C. TRAJECTORY-LEVEL ORDERING
# Statistics use ALL trajectories.
# Displayed scatter points are deterministic subsamples only.
# ------------------------------------------------------------

metrics = [
    ("corr_A_spearman", "Correlation distance", PRIMARY),
    ("mae_A_spearman", "Heatmap MAE", SECONDARY),
    ("box_A_spearman", "Bounding-box instability", TERTIARY),
]

rng = np.random.RandomState(20260923)

for xpos, (column, label, color) in enumerate(metrics, start=1):

    values = np.asarray(
        [
            float(r[column])
            for r in traj
            if r[column] not in ("", None)
        ],
        dtype=float,
    )

    ax3.boxplot(
        [values],
        positions=[xpos],
        widths=0.42,
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

    # Deterministic visual subsample.
    # Summary statistics above use all 1,200 trajectories.
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
        s=5.0,
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
ax3.set_xticklabels([
    "Correlation distance",
    "Heatmap MAE",
    "Bounding-box instability",
])

ax3.set_ylabel("Trajectory-level Spearman ρ")

clean_axis(ax3)
panel_label(ax3, "C")

# ============================================================
# FINAL LAYOUT
# ============================================================

fig.subplots_adjust(
    left=0.10,
    right=0.985,
    bottom=0.095,
    top=0.965,
)

# ============================================================
# EXPORT
# ============================================================

stem = OUT / "fig_embedding_interpolation_mechanism_REVISED"

fig.savefig(stem.with_suffix(".pdf"))
fig.savefig(stem.with_suffix(".svg"))
fig.savefig(stem.with_suffix(".png"), dpi=600)

plt.close(fig)

# ============================================================
# VERIFICATION OUTPUT
# ============================================================

print()
print("=" * 62)
print("REVISED MECHANISM FIGURE GENERATED")
print("=" * 62)

print(f"Development cases : {len(filenames)}")
print(f"Trajectories      : {len(traj)}")

print()
print("Primary endpoint means:")
print(np.mean(Y1, axis=0))

print()
print("Heatmap MAE means:")
print(np.mean(Y2, axis=0))

print()
print("FROZEN INPUT HASHES VERIFIED:")
for path, expected in EXPECTED.items():
    print(expected, path)

print()
print("OUTPUTS:")
for ext in [".pdf", ".svg", ".png"]:
    p = stem.with_suffix(ext)
    print(f"{p}   {p.stat().st_size} bytes")

print()
print("Scientific inputs unchanged.")
print("Only manuscript visualization/layout changed.")
print("=" * 62)
