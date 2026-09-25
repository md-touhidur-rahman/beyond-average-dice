import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

SEED = 42
B = 10000
THRESHOLD = 0.264446087201

ROOT = Path("..")
DATA = ROOT / "results/brain_p20/brain400_p20.csv"
PARTITION = Path("results/brain400_slice_partition.csv")
OUT = Path("results/heldout_evaluation")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
part = pd.read_csv(PARTITION)

# Partition supplies only identity/split information here.
df = df.merge(
    part[["filename", "patient_id", "split"]],
    on="filename",
    how="left",
    validate="one_to_one",
)

held = df[df["split"] == "heldout"].copy()

# Frozen partition guards.
assert len(held) == 132
assert held["patient_id"].nunique() == 53
assert set(held["split"]) == {"heldout"}

PROMPTS = [f"P{i:02d}" for i in range(1, 21)]
DICE_COLS = [f"{p}_dice" for p in PROMPTS]

def valid_box(b):
    return (
        np.all(np.isfinite(b))
        and b[2] > b[0]
        and b[3] > b[1]
    )

def box_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih

    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    union = aa + ab - inter
    return inter / union if union > 0 else 0.0

records = []

for _, row in held.iterrows():
    boxes = []

    for p in PROMPTS:
        b = np.array([
            row[f"{p}_x1"],
            row[f"{p}_y1"],
            row[f"{p}_x2"],
            row[f"{p}_y2"],
        ], dtype=float)

        if valid_box(b):
            boxes.append(b)

    K = len(boxes)

    if K < 2:
        risk = np.nan
    else:
        ious = []
        for i in range(K):
            for j in range(i + 1, K):
                ious.append(box_iou(boxes[i], boxes[j]))

        risk = 1.0 - float(np.mean(ious))

    dice = row[DICE_COLS].to_numpy(dtype=float)

    catastrophic = int(
        (dice < 0.10).any() and
        (dice >= 0.50).any()
    )

    p14 = float(row["P14_dice"])

    records.append({
        "filename": row["filename"],
        "patient_id": row["patient_id"],
        "n_valid_boxes": K,
        "risk": risk,
        "flagged": int(risk > THRESHOLD) if np.isfinite(risk) else np.nan,
        "catastrophic_slice": catastrophic,
        "P14_dice": p14,
        "P14_failure_lt_.10": int(p14 < 0.10),
    })

r = pd.DataFrame(records)

assert len(r) == 132
assert r["patient_id"].nunique() == 53
assert r["risk"].notna().all()

# Frozen-data invariant from partition creation.
assert int(r["catastrophic_slice"].sum()) == 43

r.to_csv(OUT / "heldout_slice_results.csv", index=False)

# --------------------------------------------------
# Frozen primary + secondary metrics
# --------------------------------------------------
y = r["catastrophic_slice"].to_numpy(int)
risk = r["risk"].to_numpy(float)
flag = r["flagged"].to_numpy(int)

auprc = float(average_precision_score(y, risk))
auroc = float(roc_auc_score(y, risk))

tp = int(((flag == 1) & (y == 1)).sum())
fp = int(((flag == 1) & (y == 0)).sum())
tn = int(((flag == 0) & (y == 0)).sum())
fn = int(((flag == 0) & (y == 1)).sum())

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
ppv = tp / (tp + fp) if tp + fp else np.nan
npv = tn / (tn + fn) if tn + fn else np.nan

flagged = r[r["flagged"] == 1]
retained = r[r["flagged"] == 0]

summary = {
    "heldout_guard": {
        "patients": int(r["patient_id"].nunique()),
        "slices": int(len(r)),
        "catastrophic_slices": int(r["catastrophic_slice"].sum()),
    },
    "frozen_method": {
        "risk": "1_minus_mean_pairwise_box_iou",
        "threshold": THRESHOLD,
        "baseline": "P14",
    },
    "primary": {
        "auprc": auprc,
    },
    "secondary": {
        "auroc": auroc,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "flagged_fraction": float(r["flagged"].mean()),

        "overall_P14_mean_dice":
            float(r["P14_dice"].mean()),

        "overall_P14_failure_rate_lt_.10":
            float(r["P14_failure_lt_.10"].mean()),

        "retained_P14_mean_dice":
            float(retained["P14_dice"].mean()),

        "retained_P14_median_dice":
            float(retained["P14_dice"].median()),

        "retained_P14_failure_rate_lt_.10":
            float(retained["P14_failure_lt_.10"].mean()),

        "flagged_P14_failure_rate_lt_.10":
            float(flagged["P14_failure_lt_.10"].mean()),

        "flagged_catastrophic_prevalence":
            float(flagged["catastrophic_slice"].mean()),

        "retained_catastrophic_prevalence":
            float(retained["catastrophic_slice"].mean()),

        "fraction_catastrophic_captured":
            float(
                flagged["catastrophic_slice"].sum()
                / r["catastrophic_slice"].sum()
            ),

        "fraction_P14_failures_captured":
            float(
                flagged["P14_failure_lt_.10"].sum()
                / r["P14_failure_lt_.10"].sum()
            ),
    }
}

# --------------------------------------------------
# Descriptive frozen risk-coverage fractions
# --------------------------------------------------
ordered = r.sort_values(
    ["risk", "filename"],
    ascending=[False, True]
).reset_index(drop=True)

risk_coverage = []

for reject_frac in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]:
    n_reject = int(np.ceil(len(ordered) * reject_frac))

    rej = ordered.iloc[:n_reject]
    ret = ordered.iloc[n_reject:]

    risk_coverage.append({
        "reject_fraction": reject_frac,
        "n_rejected": len(rej),
        "n_retained": len(ret),
        "coverage": len(ret) / len(ordered),

        "retained_P14_mean_dice":
            float(ret["P14_dice"].mean()),

        "retained_P14_median_dice":
            float(ret["P14_dice"].median()),

        "retained_P14_failure_rate_lt_.10":
            float(ret["P14_failure_lt_.10"].mean()),

        "retained_catastrophic_rate":
            float(ret["catastrophic_slice"].mean()),

        "rejected_P14_failure_rate_lt_.10":
            float(rej["P14_failure_lt_.10"].mean()),

        "rejected_catastrophic_rate":
            float(rej["catastrophic_slice"].mean()),
    })

rc = pd.DataFrame(risk_coverage)
rc.to_csv(OUT / "heldout_risk_coverage.csv", index=False)

# --------------------------------------------------
# Patient-cluster bootstrap
# --------------------------------------------------
rng = np.random.default_rng(SEED)
patients = r["patient_id"].unique()

groups = {
    pid: r[r["patient_id"] == pid]
    for pid in patients
}

boot_rows = []

for b in range(B):
    sampled = rng.choice(
        patients,
        size=len(patients),
        replace=True
    )

    # Give duplicated sampled patients independent cluster instances.
    chunks = []

    for draw_id, pid in enumerate(sampled):
        z = groups[pid].copy()
        z["_boot_cluster"] = draw_id
        chunks.append(z)

    bb = pd.concat(chunks, ignore_index=True)

    yy = bb["catastrophic_slice"].to_numpy(int)
    ss = bb["risk"].to_numpy(float)
    ff = bb["flagged"].to_numpy(int)

    # Very unlikely, but protect bootstrap samples with one class.
    b_auprc = (
        average_precision_score(yy, ss)
        if len(np.unique(yy)) == 2 else np.nan
    )

    b_auroc = (
        roc_auc_score(yy, ss)
        if len(np.unique(yy)) == 2 else np.nan
    )

    b_tp = ((ff == 1) & (yy == 1)).sum()
    b_fp = ((ff == 1) & (yy == 0)).sum()
    b_tn = ((ff == 0) & (yy == 0)).sum()
    b_fn = ((ff == 0) & (yy == 1)).sum()

    b_flagged = bb[bb["flagged"] == 1]
    b_retained = bb[bb["flagged"] == 0]

    boot_rows.append({
        "auprc": b_auprc,
        "auroc": b_auroc,

        "sensitivity":
            b_tp / (b_tp + b_fn)
            if b_tp + b_fn else np.nan,

        "specificity":
            b_tn / (b_tn + b_fp)
            if b_tn + b_fp else np.nan,

        "flagged_fraction":
            bb["flagged"].mean(),

        "retained_P14_mean_dice":
            b_retained["P14_dice"].mean()
            if len(b_retained) else np.nan,

        "retained_P14_failure_rate_lt_.10":
            b_retained["P14_failure_lt_.10"].mean()
            if len(b_retained) else np.nan,

        "flagged_P14_failure_rate_lt_.10":
            b_flagged["P14_failure_lt_.10"].mean()
            if len(b_flagged) else np.nan,

        "flagged_catastrophic_prevalence":
            b_flagged["catastrophic_slice"].mean()
            if len(b_flagged) else np.nan,

        "retained_catastrophic_prevalence":
            b_retained["catastrophic_slice"].mean()
            if len(b_retained) else np.nan,
    })

boot = pd.DataFrame(boot_rows)
boot.to_csv(OUT / "heldout_patient_cluster_bootstrap.csv", index=False)

def ci(col):
    x = boot[col].dropna().to_numpy()
    return [
        float(np.quantile(x, 0.025)),
        float(np.quantile(x, 0.975))
    ]

summary["bootstrap_ci95"] = {
    col: ci(col)
    for col in boot.columns
}

with open(OUT / "heldout_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# --------------------------------------------------
# Print frozen evaluation
# --------------------------------------------------
print("=== HELD-OUT EVALUATION ===")
print("patients:", r["patient_id"].nunique())
print("slices:", len(r))
print("catastrophic slices:", r["catastrophic_slice"].sum())

print("\n=== PRIMARY ===")
print("AUPRC:", f"{auprc:.6f}",
      "CI95:", summary["bootstrap_ci95"]["auprc"])

print("\n=== SECONDARY DETECTION ===")
print("AUROC:", f"{auroc:.6f}",
      "CI95:", summary["bootstrap_ci95"]["auroc"])
print("TP FP TN FN:", tp, fp, tn, fn)
print("sensitivity:", f"{sensitivity:.6f}",
      "CI95:", summary["bootstrap_ci95"]["sensitivity"])
print("specificity:", f"{specificity:.6f}",
      "CI95:", summary["bootstrap_ci95"]["specificity"])
print("PPV:", f"{ppv:.6f}")
print("NPV:", f"{npv:.6f}")
print("flagged fraction:", f"{r['flagged'].mean():.6f}",
      "CI95:", summary["bootstrap_ci95"]["flagged_fraction"])

print("\n=== SELECTIVE RELIABILITY ===")
print("overall P14 mean Dice:",
      f"{r['P14_dice'].mean():.6f}")
print("retained P14 mean Dice:",
      f"{retained['P14_dice'].mean():.6f}",
      "CI95:",
      summary["bootstrap_ci95"]["retained_P14_mean_dice"])

print("overall P14 Dice<.10:",
      f"{r['P14_failure_lt_.10'].mean():.6f}")
print("retained P14 Dice<.10:",
      f"{retained['P14_failure_lt_.10'].mean():.6f}",
      "CI95:",
      summary["bootstrap_ci95"]["retained_P14_failure_rate_lt_.10"])
print("flagged P14 Dice<.10:",
      f"{flagged['P14_failure_lt_.10'].mean():.6f}",
      "CI95:",
      summary["bootstrap_ci95"]["flagged_P14_failure_rate_lt_.10"])

print("\n=== CATASTROPHIC ENRICHMENT ===")
print("overall:", f"{r['catastrophic_slice'].mean():.6f}")
print("flagged:",
      f"{flagged['catastrophic_slice'].mean():.6f}",
      "CI95:",
      summary["bootstrap_ci95"]["flagged_catastrophic_prevalence"])
print("retained:",
      f"{retained['catastrophic_slice'].mean():.6f}",
      "CI95:",
      summary["bootstrap_ci95"]["retained_catastrophic_prevalence"])

print("\ncatastrophic captured:",
      int(flagged["catastrophic_slice"].sum()),
      "/",
      int(r["catastrophic_slice"].sum()),
      "=",
      f"{summary['secondary']['fraction_catastrophic_captured']:.6f}")

print("P14 failures captured:",
      int(flagged["P14_failure_lt_.10"].sum()),
      "/",
      int(r["P14_failure_lt_.10"].sum()),
      "=",
      f"{summary['secondary']['fraction_P14_failures_captured']:.6f}")

print("\n=== FROZEN DESCRIPTIVE RISK-COVERAGE ===")
print(
    rc.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\nHELD-OUT OPENED: YES")
print("POST-HOC METHOD TUNING ON THESE PATIENTS: PROHIBITED")
