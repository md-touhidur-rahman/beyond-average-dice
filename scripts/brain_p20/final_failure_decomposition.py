import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path

ROOT = Path("/home/hpc/rlvl/rlvl178v/prl_medclipsam")
BASE = ROOT / "MedCLIP-SAM/part2/brain400_p20_controlled"
OUT  = ROOT / "beyond-average-dice/results/brain_p20"
FIG  = ROOT / "paper_results/final_separate"

OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

d = pd.read_csv(BASE / "failure_success_pair_decomposition.csv")

# ----------------------------------------------------------
# Recover patient identifier from Brain400 filename.
# IMPORTANT: use the same filename convention as Brain400.
# Expected filenames resemble patient/slice identifiers.
# ----------------------------------------------------------
def patient_id(filename):
    m = re.search(r"pid-([^_\.]+)", str(filename))
    if m is None:
        raise ValueError(f"Could not parse patient ID: {filename}")
    return m.group(1)

d["patient_id"] = d["filename"].map(patient_id)

print("=== CLUSTER STRUCTURE ===")
print("pairs:", len(d))
print("images:", d.filename.nunique())
print("derived patients:", d.patient_id.nunique())
print()

# We expect the underlying Brain400 cohort to contain 174 patients,
# although only patients contributing failure-success pairs appear here.
# Do NOT require 174 because this table contains only 132 affected images.

# ----------------------------------------------------------
# Fixed localization strata
# ----------------------------------------------------------
labels = [
    "<0.10",
    "0.10–0.25",
    "0.25–0.50",
    "0.50–0.75",
    "0.75–0.90",
    "≥0.90",
]

edges = [-np.inf, .10, .25, .50, .75, .90, np.inf]

d["stratum"] = pd.cut(
    d["box_iou"],
    bins=edges,
    labels=labels,
    right=False
)

# ----------------------------------------------------------
# Descriptive statistics
# ----------------------------------------------------------
def summarize(x):
    return pd.Series({
        "N_pairs": len(x),
        "N_images": x.filename.nunique(),
        "N_patients": x.patient_id.nunique(),

        "mean_box_iou": x.box_iou.mean(),

        "selected_diff_mean": x.selected_diff.mean(),
        "selected_diff_median": x.selected_diff.median(),

        "oracle_diff_mean": x.oracle_diff.mean(),
        "oracle_diff_median": x.oracle_diff.median(),

        "mean_reduction": (
            x.selected_diff - x.oracle_diff
        ).mean(),

        "different_selected_idx_rate":
            x.different_selected_idx.mean(),

        "fail_oracle_ge50_rate":
            x.failure_has_oracle_50.mean(),

        "fail_selection_gap_gt20_rate":
            x.failure_selection_loss_20.mean(),
    })

summary = (
    d.groupby("stratum", observed=False)
     .apply(summarize)
     .reset_index()
)

# ----------------------------------------------------------
# Patient-cluster bootstrap
#
# Estimand:
# descriptive pair-level mean among failure-success pairs.
# Patients are resampled as clusters.
# ----------------------------------------------------------
rng = np.random.default_rng(42)
patients = d.patient_id.unique()
B = 10000

boot_rows = []

for b in range(B):
    sampled = rng.choice(
        patients,
        size=len(patients),
        replace=True
    )

    # Preserve multiplicity when a patient is sampled repeatedly.
    pieces = []

    for j, pid in enumerate(sampled):
        z = d[d.patient_id == pid].copy()
        z["_bootstrap_cluster"] = j
        pieces.append(z)

    bd = pd.concat(pieces, ignore_index=True)

    for lab in labels:
        x = bd[bd.stratum == lab]

        if len(x) == 0:
            continue

        boot_rows.append({
            "bootstrap": b,
            "stratum": lab,
            "selected_diff": x.selected_diff.mean(),
            "oracle_diff": x.oracle_diff.mean(),
            "reduction": (
                x.selected_diff - x.oracle_diff
            ).mean(),
            "oracle_ge50":
                x.failure_has_oracle_50.mean(),
            "selection_gap_gt20":
                x.failure_selection_loss_20.mean(),
        })

boot = pd.DataFrame(boot_rows)

def ci_for(metric):
    q = (
        boot.groupby("stratum", observed=False)[metric]
            .quantile([.025, .975])
            .unstack()
            .reset_index()
    )
    q.columns = [
        "stratum",
        f"{metric}_ci_low",
        f"{metric}_ci_high"
    ]
    return q

for metric in [
    "selected_diff",
    "oracle_diff",
    "reduction",
    "oracle_ge50",
    "selection_gap_gt20",
]:
    summary = summary.merge(
        ci_for(metric),
        on="stratum",
        how="left"
    )

# ----------------------------------------------------------
# Overall localization distribution
# ----------------------------------------------------------
localization = pd.DataFrame({
    "metric": [
        "box_iou_lt_010",
        "box_iou_lt_025",
        "box_iou_lt_050",
        "box_iou_ge_075",
        "box_iou_ge_090",
    ],
    "N": [
        int((d.box_iou < .10).sum()),
        int((d.box_iou < .25).sum()),
        int((d.box_iou < .50).sum()),
        int((d.box_iou >= .75).sum()),
        int((d.box_iou >= .90).sum()),
    ]
})

localization["rate"] = localization.N / len(d)

# ----------------------------------------------------------
# Save publication tables
# ----------------------------------------------------------
summary.to_csv(
    OUT / "failure_success_final_decomposition.csv",
    index=False
)

localization.to_csv(
    OUT / "failure_success_localization_distribution.csv",
    index=False
)

# Also preserve full decomposition in final results tree.
d.to_csv(
    OUT / "failure_success_pair_decomposition.csv",
    index=False
)

# ----------------------------------------------------------
# Figure: mechanism changes with localization agreement
# ----------------------------------------------------------
x = np.arange(len(labels))

sel = summary["selected_diff_mean"].to_numpy()
ora = summary["oracle_diff_mean"].to_numpy()

sel_lo = summary["selected_diff_ci_low"].to_numpy()
sel_hi = summary["selected_diff_ci_high"].to_numpy()
ora_lo = summary["oracle_diff_ci_low"].to_numpy()
ora_hi = summary["oracle_diff_ci_high"].to_numpy()

fig, ax = plt.subplots(figsize=(9.2, 5.8))

ax.errorbar(
    x - .06,
    sel,
    yerr=[sel-sel_lo, sel_hi-sel],
    marker="o",
    capsize=3,
    linewidth=2,
    label="Automatically selected SAM-B mask"
)

ax.errorbar(
    x + .06,
    ora,
    yerr=[ora-ora_lo, ora_hi-ora],
    marker="o",
    capsize=3,
    linewidth=2,
    label="Retrospective best SAM-B candidate"
)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel("Localization agreement (box IoU)")
ax.set_ylabel("Mean absolute Dice difference")
ax.set_ylim(0, 1.0)

ax.legend(frameon=False)

# Pair counts under each stratum.
for i, n in enumerate(summary.N_pairs.astype(int)):
    ax.text(
        i, .025,
        f"n={n:,}",
        ha="center",
        va="bottom",
        fontsize=9
    )

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

fig.savefig(
    FIG / "13_failure_success_component_decomposition.png",
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    FIG / "13_failure_success_component_decomposition.pdf",
    bbox_inches="tight"
)

plt.close(fig)

# ----------------------------------------------------------
# Console output: exact numbers we will use
# ----------------------------------------------------------
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

print("=== LOCALIZATION DISTRIBUTION ===")
print(localization.to_string(index=False))

print("\n=== FINAL DECOMPOSITION ===")

show = [
    "stratum",
    "N_pairs",
    "N_images",
    "N_patients",
    "selected_diff_mean",
    "selected_diff_ci_low",
    "selected_diff_ci_high",
    "oracle_diff_mean",
    "oracle_diff_ci_low",
    "oracle_diff_ci_high",
    "mean_reduction",
    "different_selected_idx_rate",
    "fail_oracle_ge50_rate",
    "fail_selection_gap_gt20_rate",
]

print(summary[show].to_string(index=False))

print("\nSaved figure:")
print(FIG / "13_failure_success_component_decomposition.png")

print("\nSaved tables:")
print(OUT / "failure_success_final_decomposition.csv")
print(OUT / "failure_success_localization_distribution.csv")
