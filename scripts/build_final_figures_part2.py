#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/home/hpc/rlvl/rlvl178v/prl_medclipsam")
OUT = ROOT / "beyond-average-dice/paper_results/FINAL_FIGURES"
OUT.mkdir(parents=True, exist_ok=True)

BREAST_CASE = ROOT / "beyond-average-dice/results/breast_p20/breast97_p20_case_metrics.csv"
BRAIN_CASE  = ROOT / "beyond-average-dice/results/brain_p20/brain400_p20_case_metrics.csv"

BREAST_RAW = ROOT / "beyond-average-dice/results/breast_p20/breast97_p20.csv"
BRAIN_RAW  = ROOT / "beyond-average-dice/results/brain_p20/brain400_p20.csv"

B600 = ROOT / "beyond-average-dice/results/brain600_language_ladder/case_metrics.csv"
T3D  = ROOT / "beyond-average-dice/results/text3dsam/text3dsam_amos30_liver_language.csv"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": .75,
    "xtick.major.width": .7,
    "ytick.major.width": .7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": .03,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def panel(ax, s):
    ax.text(
        -.13, 1.04, s,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom"
    )

def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=600)
    plt.close(fig)
    print("WROTE", stem)

# ============================================================
# LOAD CASE-LEVEL DATA
# ============================================================

breast = pd.read_csv(BREAST_CASE)
brain  = pd.read_csv(BRAIN_CASE)
b600   = pd.read_csv(B600)
t3d    = pd.read_csv(T3D)

assert len(breast) == 97
assert len(brain) == 400
assert len(b600) == 600
assert len(t3d) == 30

# Genuine case-level quantities
B_range = breast["prompt_range"].to_numpy()
B_pair  = breast["mean_pairwise_abs"].to_numpy()
B_oracle_case = breast["prompt_oracle"].to_numpy()

R_range = brain["prompt_range"].to_numpy()
R_pair  = brain["mean_pairwise_abs_dice"].to_numpy()
R_oracle_case = brain["prompt_oracle_dice"].to_numpy()

# ------------------------------------------------------------
# Frozen-record integrity checks
# ------------------------------------------------------------

assert np.isclose(B_pair.mean(), .104712673, atol=2e-6)
assert np.isclose(B_range.mean(), .340558194, atol=2e-6)
assert np.isclose((B_range > .50).mean(), 34/97, atol=1e-12)
assert np.isclose(B_oracle_case.mean(), .603951313, atol=2e-6)

assert np.isclose(R_pair.mean(), .111660177, atol=2e-6)
assert np.isclose(R_range.mean(), .385860607, atol=2e-6)
assert np.isclose((R_range > .50).mean(), 162/400, atol=1e-12)
assert np.isclose(R_oracle_case.mean(), .564146509, atol=2e-6)

print("P20 CASE-LEVEL VALIDATION PASSED")

# ============================================================
# FIGURE 6
# Broad P20 case-level sensitivity
# ============================================================

f6 = pd.concat([
    pd.DataFrame({
        "dataset": "Breast97",
        "case": breast["filename"],
        "case_range": B_range,
        "mean_pairwise_abs": B_pair
    }),
    pd.DataFrame({
        "dataset": "Brain400",
        "case": brain["filename"],
        "case_range": R_range,
        "mean_pairwise_abs": R_pair
    })
], ignore_index=True)

f6.to_csv(OUT / "Fig6_data.csv", index=False)

fig, axes = plt.subplots(
    1, 2,
    figsize=(7.05, 2.75),
    gridspec_kw={"wspace": .32}
)

# a — survival
ax = axes[0]

grid = np.linspace(0, 1, 501)

for label, vals in [
    ("Breast97", B_range),
    ("Brain400", R_range)
]:
    survival = np.array([(vals > t).mean() for t in grid])
    ax.plot(grid, survival * 100, linewidth=1.5, label=label)

ax.axvline(.50, linestyle="--", linewidth=.7)

ax.set_xlim(0, 1)
ax.set_ylim(0, 100)
ax.set_xlabel("Within-case Dice range across P20")
ax.set_ylabel("Cases exceeding range (%)")
ax.legend()

clean(ax)
panel(ax, "a")

# b — pairwise sensitivity
ax = axes[1]

ax.boxplot(
    [B_pair, R_pair],
    labels=["Breast97", "Brain400"],
    widths=.50,
    showfliers=True,
    medianprops={"linewidth":1.2},
    boxprops={"linewidth":1.0},
    whiskerprops={"linewidth":.9},
    capprops={"linewidth":.9},
    flierprops={"markersize":2.5, "alpha":.35}
)

ax.set_ylabel(r"Mean pairwise $|\Delta\mathrm{Dice}|$")
ax.set_xlabel("Evaluation cohort")

clean(ax)
panel(ax, "b")

save(fig, "Fig6_broad_P20_case_sensitivity")

# ============================================================
# LOAD RAW P20 MATRICES FOR FIXED-PROMPT RECONSTRUCTION
# ============================================================

braw = pd.read_csv(BREAST_RAW)
rraw = pd.read_csv(BRAIN_RAW)

print()
print("BREAST RAW:", braw.shape)
print("BREAST RAW COLS:", list(braw.columns))
print()
print("BRAIN RAW:", rraw.shape)
print("BRAIN RAW COLS:", list(rraw.columns))

def reconstruct_fixed(raw, case_metrics, expected_n, expected_fixed,
                      expected_oracle, expected_prompt, name):

    # ---- Long format: filename / prompt / dice
    lower = {c.lower(): c for c in raw.columns}

    filename_col = lower.get("filename")

    prompt_col = None
    for candidate in ["prompt", "prompt_id", "condition"]:
        if candidate in lower:
            prompt_col = lower[candidate]
            break

    dice_col = None
    for candidate in ["dice", "dice_score"]:
        if candidate in lower:
            dice_col = lower[candidate]
            break

    if filename_col and prompt_col and dice_col:
        pivot = raw.pivot(
            index=filename_col,
            columns=prompt_col,
            values=dice_col
        )

    else:
        # ---- Wide format
        filename_col = (
            "filename" if "filename" in raw.columns
            else raw.columns[0]
        )

        dice_cols = []

        for c in raw.columns:
            if c == filename_col:
                continue

            cl = c.lower()

            # P01 ... P20
            direct_prompt = (
                len(c) == 3 and
                c[0].upper() == "P" and
                c[1:].isdigit()
            )

            # dice_P01 etc.
            dice_prompt = (
                "dice" in cl and
                "p" in cl and
                pd.api.types.is_numeric_dtype(raw[c])
            )

            if direct_prompt or dice_prompt:
                if pd.api.types.is_numeric_dtype(raw[c]):
                    dice_cols.append(c)

        if len(dice_cols) != 20:
            raise RuntimeError(
                f"{name}: expected 20 prompt Dice columns, found "
                f"{len(dice_cols)}.\nColumns={list(raw.columns)}"
            )

        pivot = raw.set_index(filename_col)[dice_cols]

    assert pivot.shape[0] == expected_n, (
        f"{name}: expected {expected_n} cases, got {pivot.shape[0]}"
    )
    assert pivot.shape[1] == 20, (
        f"{name}: expected 20 prompts, got {pivot.shape[1]}"
    )

    # Cohort-level globally best fixed prompt
    means = pivot.mean(axis=0)
    best_col = means.idxmax()

    # normalize label e.g. dice_P13 -> P13
    best_label = str(best_col)
    if "P" in best_label.upper():
        pos = best_label.upper().rfind("P")
        best_label = best_label[pos:pos+3].upper()

    fixed = pivot.iloc[:, means.values.argmax()].to_numpy()
    oracle = pivot.max(axis=1).to_numpy()

    assert np.isclose(fixed.mean(), expected_fixed, atol=2e-6), (
        f"{name}: fixed mean {fixed.mean()} != {expected_fixed}"
    )

    assert np.isclose(oracle.mean(), expected_oracle, atol=2e-6), (
        f"{name}: oracle mean {oracle.mean()} != {expected_oracle}"
    )

    assert best_label == expected_prompt, (
        f"{name}: best fixed {best_label}, expected {expected_prompt}"
    )

    # Independent oracle cross-check against case-metric table
    case_oracle_col = (
        "prompt_oracle"
        if "prompt_oracle" in case_metrics.columns
        else "prompt_oracle_dice"
    )

    assert np.isclose(
        oracle.mean(),
        case_metrics[case_oracle_col].mean(),
        atol=2e-6
    )

    print(
        name,
        "best fixed =", best_label,
        "fixed mean =", fixed.mean(),
        "oracle =", oracle.mean()
    )

    return best_label, fixed, oracle

B_best, B_fixed, B_oracle = reconstruct_fixed(
    braw, breast,
    97,
    .499668543,
    .603951313,
    "P13",
    "Breast97"
)

R_best, R_fixed, R_oracle = reconstruct_fixed(
    rraw, brain,
    400,
    .476014,
    .564146509,
    "P14",
    "Brain400"
)

# ============================================================
# Brain600 fixed vs oracle
# ============================================================

b600_cols = [
    "dice_H0","dice_H1","dice_H2",
    "dice_L3","dice_L4","dice_L5"
]

D600 = b600[b600_cols]

means600 = D600.mean(axis=0)
best600 = means600.idxmax()

assert best600 == "dice_H0"

B600_fixed = D600[best600].to_numpy()
B600_oracle = D600.max(axis=1).to_numpy()

assert np.isclose(B600_fixed.mean(), .753830, atol=2e-6)
assert np.isclose(B600_oracle.mean(), .792847, atol=2e-6)

# ============================================================
# FIGURE 7
# Fixed condition vs retrospective oracle
# ============================================================

recover = pd.DataFrame([
    {
        "regime": "Breast97 P20",
        "n": 97,
        "best_condition": B_best,
        "best_fixed": B_fixed.mean(),
        "oracle": B_oracle.mean(),
        "advantage": (B_oracle-B_fixed).mean()
    },
    {
        "regime": "Brain400 P20",
        "n": 400,
        "best_condition": R_best,
        "best_fixed": R_fixed.mean(),
        "oracle": R_oracle.mean(),
        "advantage": (R_oracle-R_fixed).mean()
    },
    {
        "regime": "Brain600 H0-L5",
        "n": 600,
        "best_condition": "H0",
        "best_fixed": B600_fixed.mean(),
        "oracle": B600_oracle.mean(),
        "advantage": (B600_oracle-B600_fixed).mean()
    }
])

assert np.isclose(recover.iloc[0].advantage, .104282769, atol=2e-6)
assert np.isclose(recover.iloc[1].advantage, .088132485, atol=2e-6)
assert np.isclose(recover.iloc[2].advantage, .039017, atol=2e-6)

recover.to_csv(OUT / "Fig7_data.csv", index=False)

fig, ax = plt.subplots(figsize=(4.4, 2.85))

x = np.arange(3)
w = .34

ax.bar(
    x-w/2,
    recover.best_fixed,
    width=w,
    label="Best fixed condition"
)

ax.bar(
    x+w/2,
    recover.oracle,
    width=w,
    label="Retrospective oracle"
)

ax.set_xticks(x)
ax.set_xticklabels([
    "Breast97\nP20",
    "Brain400\nP20",
    "Brain600\nH0–L5"
])

ax.set_ylabel("Mean Dice")
ax.set_ylim(0, .9)
ax.legend(loc="upper left")

clean(ax)

save(fig, "Fig7_fixed_vs_oracle")

# ============================================================
# TEXT3DSAM — RAW 30-CASE VALIDATION
# ============================================================

t3d_dice_cols = ["dice_P0","dice_P1","dice_P2","dice_P3"]

assert all(c in t3d.columns for c in t3d_dice_cols)

DT = t3d[t3d_dice_cols].to_numpy()

t_range_raw = DT.max(axis=1) - DT.min(axis=1)
t_oracle_raw = DT.max(axis=1)

# Cross-check supplied derived columns
assert np.allclose(
    t_range_raw,
    t3d["range_dice"].to_numpy(),
    atol=1e-12
)

assert np.allclose(
    t_oracle_raw,
    t3d["oracle_dice"].to_numpy(),
    atol=1e-12
)

assert np.isclose(t_range_raw.mean(), .000389620, atol=2e-6)
assert np.isclose(t_range_raw.max(), .00121674, atol=2e-6)

print("TEXT3DSAM RAW VALIDATION PASSED")

# ============================================================
# FIGURE 8
# Cross-regime boundary
#
# IMPORTANT: descriptive boundary comparison only.
# Not an architecture ranking / causal comparison.
# ============================================================

range600 = (
    D600.max(axis=1).to_numpy() -
    D600.min(axis=1).to_numpy()
)

boundary = pd.DataFrame([
    {
        "regime": "Breast97 P20",
        "n": 97,
        "mean_case_range": B_range.mean()
    },
    {
        "regime": "Brain400 P20",
        "n": 400,
        "mean_case_range": R_range.mean()
    },
    {
        "regime": "Brain600 H0-L5",
        "n": 600,
        "mean_case_range": range600.mean()
    },
    {
        "regime": "Text3DSAM liver",
        "n": 30,
        "mean_case_range": t_range_raw.mean()
    }
])

assert np.isclose(
    boundary.iloc[0].mean_case_range,
    .340558194,
    atol=2e-6
)

assert np.isclose(
    boundary.iloc[1].mean_case_range,
    .385860607,
    atol=2e-6
)

assert np.isclose(
    boundary.iloc[2].mean_case_range,
    .132872,
    atol=2e-6
)

assert np.isclose(
    boundary.iloc[3].mean_case_range,
    .000389620,
    atol=2e-6
)

boundary.to_csv(OUT / "Fig8_data.csv", index=False)

fig, ax = plt.subplots(figsize=(4.7, 2.9))

x = np.arange(4)

bars = ax.bar(
    x,
    boundary.mean_case_range,
    width=.62
)

ax.set_xticks(x)
ax.set_xticklabels([
    "Breast97\nP20",
    "Brain400\nP20",
    "Brain600\nH0–L5",
    "Text3DSAM\nliver"
])

ax.set_ylabel("Mean within-case Dice range")
ax.set_ylim(0, .43)

# Text3DSAM bar is tiny, so label exact value
ax.text(
    3,
    boundary.iloc[3].mean_case_range + .012,
    f'{boundary.iloc[3].mean_case_range:.4f}',
    ha="center",
    va="bottom",
    fontsize=7
)

clean(ax)

save(fig, "Fig8_regime_boundary")

# ============================================================
# PROVENANCE
# ============================================================

prov = pd.DataFrame([
    {
        "figure":"Fig6_broad_P20_case_sensitivity",
        "source":str(BREAST_CASE),
        "role":"raw case-level Breast97 P20 statistics"
    },
    {
        "figure":"Fig6_broad_P20_case_sensitivity",
        "source":str(BRAIN_CASE),
        "role":"raw case-level Brain400 P20 statistics"
    },
    {
        "figure":"Fig7_fixed_vs_oracle",
        "source":str(BREAST_RAW),
        "role":"prompt-wise Breast97 Dice; fixed prompt selected from cohort means"
    },
    {
        "figure":"Fig7_fixed_vs_oracle",
        "source":str(BRAIN_RAW),
        "role":"prompt-wise Brain400 Dice; fixed prompt selected from cohort means"
    },
    {
        "figure":"Fig7_fixed_vs_oracle",
        "source":str(B600),
        "role":"prompt-wise H0-L5 Brain600 Dice"
    },
    {
        "figure":"Fig8_regime_boundary",
        "source":str(T3D),
        "role":"raw P0-P3 Text3DSAM case Dice"
    },
])

prov.to_csv(
    OUT / "FIGURE_PROVENANCE_PART2.csv",
    index=False
)

print()
print("="*76)
print("PART 2 COMPLETE")
print("="*76)
print(recover.to_string(index=False))
print()
print(boundary.to_string(index=False))
print()
print("OUTPUT:", OUT)
