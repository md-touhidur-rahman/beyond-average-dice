#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path("/home/hpc/rlvl/rlvl178v/prl_medclipsam")
OUT = ROOT / "beyond-average-dice/paper_results/FINAL_FIGURES"
OUT.mkdir(parents=True, exist_ok=True)

B600_CASE = ROOT / "beyond-average-dice/results/brain600_language_ladder/case_metrics.csv"
B600_CI   = ROOT / "beyond-average-dice/results/brain600_language_ladder/bootstrap_95ci.csv"

B400_PAIR = ROOT / "beyond-average-dice/results/brain_p20/failure_success_pair_decomposition.csv"
B400_SUM  = ROOT / "beyond-average-dice/results/brain_p20/failure_success_final_decomposition.csv"

# ============================================================
# PUBLICATION STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.labelsize": 9,
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
    "savefig.pad_inches": 0.03,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def panel(ax, s):
    ax.text(
        -0.13, 1.04, s,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom"
    )

def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=600)
    plt.close(fig)
    print("WROTE", name)

# ============================================================
# LOAD + VALIDATE RAW RECORDS
# ============================================================

b600 = pd.read_csv(B600_CASE)
ci600 = pd.read_csv(B600_CI)

b400 = pd.read_csv(B400_PAIR)
sum400 = pd.read_csv(B400_SUM)

assert len(b600) == 600
assert b600.case.nunique() == 600
assert len(b400) == 5398

conds = ["H1", "H2", "L3", "L4", "L5"]

# ============================================================
# FIGURE 1
# Brain600 controlled reformulation/information ablation
# ============================================================

rows = []

for c in conds:
    d = (b600[f"dice_{c}"] - b600["dice_H0"]).abs()

    cirow = ci600[
        (ci600.scope == c) &
        (ci600.metric == "mean_abs_delta")
    ]

    assert len(cirow) == 1

    rows.append({
        "condition": c,
        "mean_abs_delta": d.mean(),
        "median_abs_delta": d.median(),
        "ci_low": cirow.ci_low.iloc[0],
        "ci_high": cirow.ci_high.iloc[0],
        "gt_010": (d > .10).mean(),
        "gt_020": (d > .20).mean(),
        "gt_050": (d > .50).mean(),
    })

f1 = pd.DataFrame(rows)

# raw-derived integrity checks
assert np.isclose(f1.loc[0,"mean_abs_delta"], .020296, atol=1e-6)
assert np.isclose(f1.loc[3,"mean_abs_delta"], .080111, atol=1e-6)
assert np.isclose(f1.loc[4,"mean_abs_delta"], .072436, atol=1e-6)

f1.to_csv(OUT / "Fig1_data.csv", index=False)

fig, axes = plt.subplots(
    1, 2,
    figsize=(7.05, 2.75),
    gridspec_kw={"wspace": .32}
)

x = np.arange(5)

# Panel a
ax = axes[0]

means = f1.mean_abs_delta.to_numpy()
lo = f1.ci_low.to_numpy()
hi = f1.ci_high.to_numpy()

ax.errorbar(
    x, means,
    yerr=np.vstack([means-lo, hi-means]),
    fmt="o",
    markersize=5,
    capsize=3,
    linewidth=1.1
)

ax.set_xticks(x)
ax.set_xticklabels(conds)
ax.set_ylabel(r"Mean $|\Delta\mathrm{Dice}|$ from H0")
ax.set_xlabel("Language condition")
ax.set_ylim(0, .105)

clean(ax)
panel(ax, "a")

# Panel b
ax = axes[1]

w = .23

ax.bar(x-w, f1.gt_010*100, width=w, label=r"$|\Delta Dice|>0.10$")
ax.bar(x,   f1.gt_020*100, width=w, label=r"$|\Delta Dice|>0.20$")
ax.bar(x+w, f1.gt_050*100, width=w, label=r"$|\Delta Dice|>0.50$")

ax.set_xticks(x)
ax.set_xticklabels(conds)
ax.set_xlabel("Language condition")
ax.set_ylabel("Cases (%)")
ax.set_ylim(0, 22)

ax.legend(loc="upper left")
clean(ax)
panel(ax, "b")

save(fig, "Fig1_controlled_language")

# ============================================================
# FIGURE 2
# Brain600 within-case range survival
# ============================================================

D = b600[
    ["dice_H0","dice_H1","dice_H2","dice_L3","dice_L4","dice_L5"]
].to_numpy()

ranges = D.max(axis=1) - D.min(axis=1)

assert np.isclose(ranges.mean(), .132872, atol=1e-6)
assert np.isclose(np.median(ranges), .036751, atol=1e-6)
assert np.isclose(ranges.max(), .977128, atol=1e-6)
assert (ranges > .50).sum() == 48

f2 = pd.DataFrame({
    "case": b600.case,
    "range_H0_L5": ranges
}).sort_values("range_H0_L5")

f2.to_csv(OUT / "Fig2_data.csv", index=False)

# P(range > x), including endpoint correctly
xx = np.sort(ranges)
yy = np.array([(ranges > v).mean() for v in xx])

fig, ax = plt.subplots(figsize=(3.45, 2.75))

ax.step(xx, yy*100, where="post", linewidth=1.4)

for t in [.10, .20, .50]:
    ax.axvline(t, linestyle="--", linewidth=.7)

ax.set_xlim(0, 1)
ax.set_ylim(0, 100)
ax.set_xlabel("Within-case Dice range across H0–L5")
ax.set_ylabel("Cases exceeding range (%)")

clean(ax)

save(fig, "Fig2_brain600_range_tail")

# ============================================================
# FIGURE 3
# Brain400 catastrophic-pair localization distribution
# ============================================================

thresholds = [
    ("<0.10", (b400.box_iou < .10)),
    ("<0.25", (b400.box_iou < .25)),
    ("<0.50", (b400.box_iou < .50)),
    (r"$\geq$0.75", (b400.box_iou >= .75)),
    (r"$\geq$0.90", (b400.box_iou >= .90)),
]

f3 = pd.DataFrame([
    {
        "criterion": label,
        "n": int(mask.sum()),
        "rate": float(mask.mean())
    }
    for label, mask in thresholds
])

assert f3.n.tolist() == [4520,4773,5088,120,25]

f3.to_csv(OUT / "Fig3_data.csv", index=False)

fig, ax = plt.subplots(figsize=(3.45, 2.75))

x = np.arange(len(f3))

bars = ax.bar(
    x,
    f3.rate*100,
    width=.65
)

ax.set_xticks(x)
ax.set_xticklabels(f3.criterion)
ax.set_xlabel("Bounding-box IoU criterion")
ax.set_ylabel("Failure-success pairs (%)")
ax.set_ylim(0,100)

for bar, rate in zip(bars, f3.rate):
    ax.text(
        bar.get_x()+bar.get_width()/2,
        bar.get_height()+2,
        f"{100*rate:.1f}",
        ha="center",
        va="bottom",
        fontsize=7
    )

clean(ax)

save(fig, "Fig3_catastrophic_localization")

# ============================================================
# FIGURE 4
# Full Brain400 failure-component decomposition
#
# Point estimates from raw 5,398 pair records.
# CIs from original frozen patient-cluster bootstrap.
# ============================================================

order = [
    "<0.10",
    "0.10–0.25",
    "0.25–0.50",
    "0.50–0.75",
    "0.75–0.90",
    "≥0.90",
]

rows = []

for s in order:
    d = b400[b400.stratum == s]

    frozen = sum400[sum400.stratum == s]
    assert len(frozen) == 1

    rows.append({
        "stratum": s,
        "N_pairs": len(d),
        "N_images": d.filename.nunique(),
        "N_patients": d.patient_id.nunique(),

        "selected_diff": d.selected_diff.mean(),
        "selected_low": frozen.selected_diff_ci_low.iloc[0],
        "selected_high": frozen.selected_diff_ci_high.iloc[0],

        "oracle_diff": d.oracle_diff.mean(),
        "oracle_low": frozen.oracle_diff_ci_low.iloc[0],
        "oracle_high": frozen.oracle_diff_ci_high.iloc[0],

        "oracle50": d.failure_has_oracle_50.mean(),
        "diff_idx": d.different_selected_idx.mean(),
        "gap20": d.failure_selection_loss_20.mean(),
    })

f4 = pd.DataFrame(rows)

assert f4.N_pairs.tolist() == [4520,253,315,190,95,25]

last = f4.iloc[-1]
assert last.N_images == 6
assert last.N_patients == 6
assert np.isclose(last.oracle50, .88)
assert np.isclose(last.diff_idx, .96)
assert np.isclose(last.gap20, 1.0)

f4.to_csv(OUT / "Fig4_data.csv", index=False)

fig, axes = plt.subplots(
    1, 2,
    figsize=(7.05, 2.85),
    gridspec_kw={"wspace": .34}
)

x = np.arange(6)

# Panel a
ax = axes[0]

w = .34

sel = f4.selected_diff.to_numpy()
ora = f4.oracle_diff.to_numpy()

selerr = np.vstack([
    sel-f4.selected_low.to_numpy(),
    f4.selected_high.to_numpy()-sel
])

oraerr = np.vstack([
    ora-f4.oracle_low.to_numpy(),
    f4.oracle_high.to_numpy()-ora
])

ax.bar(
    x-w/2, sel, width=w,
    yerr=selerr,
    capsize=2,
    label="Selected mask"
)

ax.bar(
    x+w/2, ora, width=w,
    yerr=oraerr,
    capsize=2,
    label="Candidate oracle"
)

ax.set_xticks(x)
ax.set_xticklabels(order, rotation=30, ha="right")
ax.set_ylabel(r"Mean $|\Delta\mathrm{Dice}|$")
ax.set_xlabel("Box IoU stratum")
ax.set_ylim(0, .9)
ax.legend(loc="upper right")

clean(ax)
panel(ax, "a")

# Panel b
ax = axes[1]

w = .24

ax.bar(
    x-w,
    f4.oracle50*100,
    width=w,
    label=r"Failure candidate Dice $\geq0.50$"
)

ax.bar(
    x,
    f4.diff_idx*100,
    width=w,
    label="Different selected candidate"
)

ax.bar(
    x+w,
    f4.gap20*100,
    width=w,
    label=r"Failure selection gap $>0.20$"
)

ax.set_xticks(x)
ax.set_xticklabels(order, rotation=30, ha="right")
ax.set_xlabel("Box IoU stratum")
ax.set_ylabel("Pairs (%)")
ax.set_ylim(0,105)

ax.legend(
    loc="upper left",
    fontsize=6.7
)

# rare high-overlap denominator
ax.text(
    .98, .04,
    r"$\geq0.90$: 25 pairs"+"\n6 images / 6 patients",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=7
)

clean(ax)
panel(ax, "b")

save(fig, "Fig4_failure_component_decomposition")

# ============================================================
# FIGURE 5
# Brain600 localization agreement vs Dice change
# Case-level scatter, not aggregated/artificial points
# ============================================================

records = []

for c in conds:
    delta = (b600[f"dice_{c}"] - b600.dice_H0).abs()

    for case, iou, dd in zip(
        b600.case,
        b600[f"box_iou_{c}"],
        delta
    ):
        records.append({
            "case": case,
            "condition": c,
            "box_iou_H0": iou,
            "abs_delta_dice": dd
        })

f5 = pd.DataFrame(records)
assert len(f5) == 3000

f5.to_csv(OUT / "Fig5_data.csv", index=False)

fig, ax = plt.subplots(figsize=(3.45,2.8))

ax.scatter(
    f5.box_iou_H0,
    f5.abs_delta_dice,
    s=7,
    alpha=.18,
    edgecolors="none"
)

ax.set_xlim(-.02,1.02)
ax.set_ylim(-.02,1.02)
ax.set_xlabel("Bounding-box IoU with H0")
ax.set_ylabel(r"$|\Delta\mathrm{Dice}|$ from H0")

clean(ax)

save(fig, "Fig5_brain600_localization_association")

# ============================================================
# PROVENANCE + HASHES
# ============================================================

manifest = pd.DataFrame([
    ["Fig1_controlled_language",
     str(B600_CASE),
     "600 images; raw case Dice; frozen original bootstrap CI"],

    ["Fig2_brain600_range_tail",
     str(B600_CASE),
     "600 images; raw six-condition Dice"],

    ["Fig3_catastrophic_localization",
     str(B400_PAIR),
     "5398 catastrophic failure-success prompt pairs"],

    ["Fig4_failure_component_decomposition",
     str(B400_PAIR),
     "raw pair point estimates + original frozen patient-cluster CI"],

    ["Fig5_brain600_localization_association",
     str(B600_CASE),
     "3000 H0-relative image-condition observations"],
], columns=["figure","source","provenance"])

manifest.to_csv(OUT / "FIGURE_PROVENANCE.csv", index=False)

print()
print("="*72)
print("FINAL EMPIRICAL FIGURES GENERATED")
print("OUTPUT:", OUT)
print("="*72)
