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

EXPECTED = {
    CASE_FILE:
        "23f3841742e544a668520429e9762144aa7a879dc456b7de84a2c4587ff15087",
    TRAJ_FILE:
        "83897c70552062f39f8c6d36c86eb1323b3db774b7e551226d40fe3db9fa6c46",
}


# ============================================================
# VERIFY FROZEN INPUTS
# ============================================================

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


for path, expected in EXPECTED.items():
    got = sha256(path)
    assert got == expected, (
        f"FROZEN INPUT HASH MISMATCH\n"
        f"path={path}\n"
        f"expected={expected}\n"
        f"got={got}"
    )


# ============================================================
# PUBLICATION STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,

    "axes.linewidth": 0.75,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,

    "legend.frameon": False,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def save_all(fig, stem):
    pdf = OUT / f"{stem}.pdf"
    svg = OUT / f"{stem}.svg"
    png = OUT / f"{stem}.png"

    fig.savefig(pdf)
    fig.savefig(svg)
    fig.savefig(png, dpi=600)

    print(f"{pdf} {pdf.stat().st_size} bytes")
    print(f"{svg} {svg.stat().st_size} bytes")
    print(f"{png} {png.stat().st_size} bytes")


# ============================================================
# LOAD DATA
# ============================================================

case = pd.read_csv(CASE_FILE)
traj = pd.read_csv(TRAJ_FILE)

required_case = {
    "filename",
    "alpha",
    "corr_distance_from_A",
    "heatmap_mae_from_A",
}

required_traj = {
    "corr_A_spearman",
    "mae_A_spearman",
    "box_A_spearman",
}

assert required_case.issubset(case.columns), (
    f"Missing case columns: {required_case - set(case.columns)}"
)
assert required_traj.issubset(traj.columns), (
    f"Missing trajectory columns: "
    f"{required_traj - set(traj.columns)}"
)

alphas = np.array([0.00, 0.25, 0.50, 0.75, 1.00])

assert sorted(case["alpha"].unique().tolist()) == alphas.tolist()
assert case["filename"].nunique() == 12
assert len(traj) == 1200


# ============================================================
# FIGURE 1
# CORRELATION DISTANCE FROM ENDPOINT A
# ============================================================

fig, ax = plt.subplots(figsize=(3.45, 2.75))

pivot = (
    case.pivot(
        index="filename",
        columns="alpha",
        values="corr_distance_from_A",
    )
    .reindex(columns=alphas)
)

# Individual development cases.
for _, row in pivot.iterrows():
    ax.plot(
        alphas,
        row.values,
        color="0.72",
        linewidth=0.75,
        zorder=1,
    )

mean = pivot.mean(axis=0).values
sem = pivot.sem(axis=0).values

ax.fill_between(
    alphas,
    mean - sem,
    mean + sem,
    alpha=0.16,
    zorder=2,
)

ax.plot(
    alphas,
    mean,
    marker="o",
    markersize=4.2,
    linewidth=1.8,
    zorder=3,
)

ax.set_xlabel("Embedding interpolation fraction, α")
ax.set_ylabel("Correlation distance from endpoint A")

ax.set_xticks(alphas)
ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

save_all(
    fig,
    "fig_embedding_interpolation_correlation_distance",
)

plt.close(fig)


# ============================================================
# FIGURE 2
# HEATMAP MAE FROM ENDPOINT A
# ============================================================

fig, ax = plt.subplots(figsize=(3.45, 2.75))

pivot = (
    case.pivot(
        index="filename",
        columns="alpha",
        values="heatmap_mae_from_A",
    )
    .reindex(columns=alphas)
)

for _, row in pivot.iterrows():
    ax.plot(
        alphas,
        row.values,
        color="0.72",
        linewidth=0.75,
        zorder=1,
    )

mean = pivot.mean(axis=0).values
sem = pivot.sem(axis=0).values

ax.fill_between(
    alphas,
    mean - sem,
    mean + sem,
    alpha=0.16,
    zorder=2,
)

ax.plot(
    alphas,
    mean,
    marker="o",
    markersize=4.2,
    linewidth=1.8,
    zorder=3,
)

ax.set_xlabel("Embedding interpolation fraction, α")
ax.set_ylabel("Heatmap MAE from endpoint A")

ax.set_xticks(alphas)
ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

save_all(
    fig,
    "fig_embedding_interpolation_heatmap_mae",
)

plt.close(fig)


# ============================================================
# FIGURE 3
# TRAJECTORY-LEVEL ORDERING
# ============================================================

fig, ax = plt.subplots(figsize=(4.8, 3.15))

groups = [
    traj["corr_A_spearman"].to_numpy(),
    traj["mae_A_spearman"].to_numpy(),
    traj["box_A_spearman"].to_numpy(),
]

labels = [
    "Correlation distance",
    "Heatmap MAE",
    "Bounding-box instability",
]

positions = np.arange(1, 4)

# Boxplots summarize all 1200 trajectories.
bp = ax.boxplot(
    groups,
    positions=positions,
    widths=0.42,
    patch_artist=True,
    showfliers=False,
    medianprops={
        "linewidth": 1.2,
        "color": "0.25",
    },
    boxprops={
        "linewidth": 0.9,
    },
    whiskerprops={
        "linewidth": 0.9,
    },
    capprops={
        "linewidth": 0.9,
    },
)

# Keep fill subtle.
for box in bp["boxes"]:
    box.set_alpha(0.20)

# Deterministic point display for legibility.
# Statistics remain based on all 1200 trajectories.
rng = np.random.default_rng(20260923)

for xpos, values in zip(positions, groups):
    n_display = min(160, len(values))

    idx = np.linspace(
        0,
        len(values) - 1,
        n_display,
        dtype=int,
    )

    shown = values[idx]

    jitter = rng.uniform(
        -0.075,
        0.075,
        size=len(shown),
    )

    ax.scatter(
        np.full(len(shown), xpos) + jitter,
        shown,
        s=8,
        alpha=0.20,
        linewidths=0,
        zorder=1,
    )

ax.axhline(
    0,
    linestyle="--",
    linewidth=0.75,
    color="0.5",
    zorder=0,
)

ax.set_ylabel("Spearman ρ across interpolation doses")

ax.set_xticks(positions)
ax.set_xticklabels(
    labels,
    rotation=0,
    ha="center",
)

# Enough horizontal separation for the full scientific wording.
ax.set_xlim(0.45, 3.55)
ax.set_ylim(-0.08, 1.045)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.subplots_adjust(
    left=0.15,
    right=0.98,
    top=0.97,
    bottom=0.20,
)

save_all(
    fig,
    "fig_embedding_interpolation_trajectory_ordering",
)

plt.close(fig)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=== THREE SEPARATE PUBLICATION FIGURES CREATED ===")
print()
print("Figure 1:")
print("  Correlation distance from endpoint A")
print()
print("Figure 2:")
print("  Heatmap MAE from endpoint A")
print()
print("Figure 3:")
print("  Spearman rho across interpolation doses")
print("  Categories:")
print("    Correlation distance")
print("    Heatmap MAE")
print("    Bounding-box instability")
print()
print("INPUT HASHES VERIFIED:")
for path in EXPECTED:
    print(sha256(path), path)
