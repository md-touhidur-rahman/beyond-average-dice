import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

SEED = 42
ROOT = Path("..")
DATA = ROOT / "results/brain_p20/brain400_p20.csv"
PARTITION = Path("results/brain400_slice_partition.csv")
OUT = Path("results/exp1_development")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
part = pd.read_csv(PARTITION)

df = df.merge(
    part[["filename", "patient_id", "split", "catastrophic_slice"]],
    on="filename",
    how="left",
    validate="one_to_one",
)

assert df["split"].notna().all()
dev = df[df["split"] == "development"].copy()

# Hard guard: this analysis may see development rows only.
assert len(dev) == 268
assert dev["patient_id"].nunique() == 121
assert set(dev["split"]) == {"development"}
assert int(dev["catastrophic_slice"].sum()) == 89

PROMPTS = [f"P{i:02d}" for i in range(1, 21)]

def box_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih

    aa = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    ab = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])

    union = aa + ab - inter
    return inter / union if union > 0 else 0.0

def valid_box(b):
    return (
        np.all(np.isfinite(b))
        and b[2] > b[0]
        and b[3] > b[1]
    )

records = []
prompt_records = []

for _, row in dev.iterrows():

    boxes = {}
    dice = {}

    for p in PROMPTS:
        b = np.array([
            row[f"{p}_x1"],
            row[f"{p}_y1"],
            row[f"{p}_x2"],
            row[f"{p}_y2"],
        ], dtype=float)

        if valid_box(b):
            boxes[p] = b

        dice[p] = float(row[f"{p}_dice"])

    valid_prompts = [p for p in PROMPTS if p in boxes]
    K = len(valid_prompts)

    if K >= 2:
        pair_ious = []

        agreements = {p: [] for p in valid_prompts}

        for i in range(K):
            for j in range(i + 1, K):
                p = valid_prompts[i]
                q = valid_prompts[j]
                v = box_iou(boxes[p], boxes[q])

                pair_ious.append(v)
                agreements[p].append(v)
                agreements[q].append(v)

        mean_pair_iou = float(np.mean(pair_ious))
        median_pair_iou = float(np.median(pair_ious))

        centers = np.array([
            [(boxes[p][0] + boxes[p][2]) / 2.0,
             (boxes[p][1] + boxes[p][3]) / 2.0]
            for p in valid_prompts
        ])

        widths = np.array([
            boxes[p][2] - boxes[p][0] for p in valid_prompts
        ])
        heights = np.array([
            boxes[p][3] - boxes[p][1] for p in valid_prompts
        ])
        areas = widths * heights

        # GT-free scale normalization using the localization hypotheses themselves.
        scale = np.sqrt(np.median(areas))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0

        centroid = centers.mean(axis=0)
        center_dispersion = float(
            np.mean(np.linalg.norm(centers - centroid, axis=1)) / scale
        )

        mean_area = float(np.mean(areas))
        area_dispersion = (
            float(np.std(areas) / mean_area)
            if mean_area > 0 else 0.0
        )

        per_prompt_agreement = {
            p: float(np.mean(agreements[p]))
            for p in valid_prompts
        }

    else:
        mean_pair_iou = np.nan
        median_pair_iou = np.nan
        center_dispersion = np.nan
        area_dispersion = np.nan
        per_prompt_agreement = {}

    # Medoid requires >=3 valid boxes per frozen protocol.
    medoid_prompt = None
    medoid_tie = False
    regret = np.nan
    oracle_prompt = None
    oracle_dice = np.nan
    medoid_dice = np.nan

    if K >= 3:
        max_agreement = max(per_prompt_agreement.values())

        tied = [
            p for p in valid_prompts
            if np.isclose(
                per_prompt_agreement[p],
                max_agreement,
                rtol=0.0,
                atol=1e-12
            )
        ]

        medoid_tie = len(tied) > 1

        # Lowest frozen prompt index.
        medoid_prompt = sorted(
            tied,
            key=lambda x: int(x[1:])
        )[0]

        oracle_prompt = max(
            PROMPTS,
            key=lambda p: dice[p]
        )

        oracle_dice = float(dice[oracle_prompt])
        medoid_dice = float(dice[medoid_prompt])
        regret = oracle_dice - medoid_dice

    records.append({
        "filename": row["filename"],
        "patient_id": row["patient_id"],
        "catastrophic_slice": int(row["catastrophic_slice"]),
        "n_valid_boxes": K,
        "mean_pair_iou": mean_pair_iou,
        "median_pair_iou": median_pair_iou,
        "signal_1_minus_mean_iou": (
            1.0 - mean_pair_iou if np.isfinite(mean_pair_iou) else np.nan
        ),
        "signal_1_minus_median_iou": (
            1.0 - median_pair_iou if np.isfinite(median_pair_iou) else np.nan
        ),
        "signal_center_dispersion": center_dispersion,
        "signal_area_dispersion": area_dispersion,
        "medoid_prompt": medoid_prompt,
        "medoid_tie": int(medoid_tie),
        "oracle_prompt": oracle_prompt,
        "oracle_dice": oracle_dice,
        "medoid_dice": medoid_dice,
        "medoid_regret": regret,
    })

    for p in valid_prompts:
        prompt_records.append({
            "filename": row["filename"],
            "patient_id": row["patient_id"],
            "catastrophic_slice": int(row["catastrophic_slice"]),
            "prompt": p,
            "agreement": per_prompt_agreement.get(p, np.nan),
            "dice": dice[p],
        })

case = pd.DataFrame(records)
prompt_df = pd.DataFrame(prompt_records)

case.to_csv(OUT / "exp1_case_signals.csv", index=False)
prompt_df.to_csv(OUT / "exp1_prompt_agreement.csv", index=False)

# --------------------------------------------------
# Experiment 1A
# --------------------------------------------------
candidates = [
    ("1_minus_mean_pairwise_iou", "signal_1_minus_mean_iou"),
    ("1_minus_median_pairwise_iou", "signal_1_minus_median_iou"),
    ("center_dispersion", "signal_center_dispersion"),
    ("area_dispersion", "signal_area_dispersion"),
]

metrics = []

for order, (name, col) in enumerate(candidates):
    z = case[["catastrophic_slice", col]].dropna()
    y = z["catastrophic_slice"].astype(int).to_numpy()
    s = z[col].to_numpy(float)

    ap = average_precision_score(y, s)
    auc = roc_auc_score(y, s)

    metrics.append({
        "candidate_order": order,
        "signal": name,
        "column": col,
        "n": len(z),
        "auprc": float(ap),
        "auroc": float(auc),
    })

metrics_df = pd.DataFrame(metrics).sort_values(
    ["auprc", "auroc", "candidate_order"],
    ascending=[False, False, True]
).reset_index(drop=True)

metrics_df.to_csv(OUT / "exp1a_signal_metrics.csv", index=False)

winner = metrics_df.iloc[0]
winner_col = winner["column"]

z = case[["catastrophic_slice", winner_col]].dropna()
y = z["catastrophic_slice"].astype(int).to_numpy()
s = z[winner_col].to_numpy(float)

fpr, tpr, thresholds = roc_curve(y, s)

eligible = np.where(tpr >= 0.80)[0]

if len(eligible):
    # Greatest specificity = smallest FPR.
    best_fpr = np.min(fpr[eligible])
    eligible2 = eligible[np.isclose(fpr[eligible], best_fpr)]

    # Deterministic tie break: highest threshold.
    idx = eligible2[np.argmax(thresholds[eligible2])]
    threshold_rule = "sensitivity>=0.80_then_max_specificity"
else:
    J = tpr - fpr
    idx = int(np.argmax(J))
    threshold_rule = "max_youden_J_fallback"

gate_threshold = float(thresholds[idx])
gate_sensitivity = float(tpr[idx])
gate_specificity = float(1.0 - fpr[idx])

# --------------------------------------------------
# Experiment 1B
# --------------------------------------------------
cat = case[
    (case["catastrophic_slice"] == 1) &
    case["medoid_regret"].notna()
].copy()

assert len(cat) > 0

mean_medoid_regret = float(cat["medoid_regret"].mean())

# Exact expected random-description regret:
# oracle Dice - mean Dice across the 20 descriptions.
random_regrets = []

for fname in cat["filename"]:
    row = dev.loc[dev["filename"] == fname].iloc[0]
    ds = np.array([float(row[f"{p}_dice"]) for p in PROMPTS])
    random_regrets.append(float(ds.max() - ds.mean()))

cat["random_expected_regret"] = random_regrets
mean_random_regret = float(cat["random_expected_regret"].mean())
mean_regret_difference = float(
    (cat["medoid_regret"] - cat["random_expected_regret"]).mean()
)

cat.to_csv(OUT / "exp1b_catastrophic_regret.csv", index=False)

top1_hit = float(
    (cat["medoid_prompt"] == cat["oracle_prompt"]).mean()
)

# --------------------------------------------------
# Patient-cluster bootstrap for primary 1B quantities
# --------------------------------------------------
rng = np.random.default_rng(SEED)
patients = cat["patient_id"].unique()
B = 10000

boot_medoid = np.empty(B)
boot_random = np.empty(B)
boot_diff = np.empty(B)

groups = {
    pid: cat[cat["patient_id"] == pid]
    for pid in patients
}

for b in range(B):
    sampled = rng.choice(patients, size=len(patients), replace=True)
    chunks = [groups[pid] for pid in sampled]
    bb = pd.concat(chunks, ignore_index=True)

    boot_medoid[b] = bb["medoid_regret"].mean()
    boot_random[b] = bb["random_expected_regret"].mean()
    boot_diff[b] = (
        bb["medoid_regret"] - bb["random_expected_regret"]
    ).mean()

def ci(x):
    return [
        float(np.quantile(x, 0.025)),
        float(np.quantile(x, 0.975)),
    ]

summary = {
    "data_guard": {
        "split": "development_only",
        "patients": int(dev["patient_id"].nunique()),
        "slices": int(len(dev)),
        "catastrophic_slices": int(dev["catastrophic_slice"].sum()),
    },
    "exp1a": {
        "selected_signal": str(winner["signal"]),
        "selected_column": str(winner_col),
        "auprc": float(winner["auprc"]),
        "auroc": float(winner["auroc"]),
        "gate_threshold": gate_threshold,
        "gate_rule": threshold_rule,
        "gate_sensitivity": gate_sensitivity,
        "gate_specificity": gate_specificity,
    },
    "exp1b": {
        "catastrophic_slices_evaluated": int(len(cat)),
        "mean_medoid_regret": mean_medoid_regret,
        "mean_medoid_regret_ci95": ci(boot_medoid),
        "mean_random_expected_regret": mean_random_regret,
        "mean_random_expected_regret_ci95": ci(boot_random),
        "mean_medoid_minus_random_regret": mean_regret_difference,
        "mean_medoid_minus_random_regret_ci95": ci(boot_diff),
        "medoid_top1_oracle_hit_rate": top1_hit,
        "medoid_ties_all_development": int(case["medoid_tie"].sum()),
        "cases_lt3_valid_boxes": int((case["n_valid_boxes"] < 3).sum()),
    },
}

with open(OUT / "exp1_development_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("=== DEVELOPMENT-ONLY GUARD ===")
print(summary["data_guard"])

print("\n=== EXPERIMENT 1A ===")
print(metrics_df[
    ["signal", "n", "auprc", "auroc"]
].to_string(index=False))

print("\nSELECTED SIGNAL:", summary["exp1a"]["selected_signal"])
print("AUPRC:", f'{summary["exp1a"]["auprc"]:.6f}')
print("AUROC:", f'{summary["exp1a"]["auroc"]:.6f}')
print("GATE:", f'{summary["exp1a"]["gate_threshold"]:.12f}')
print("SENSITIVITY:", f'{summary["exp1a"]["gate_sensitivity"]:.6f}')
print("SPECIFICITY:", f'{summary["exp1a"]["gate_specificity"]:.6f}')

print("\n=== EXPERIMENT 1B ===")
for k, v in summary["exp1b"].items():
    print(k, ":", v)

print("\nHELD-OUT OUTCOMES ACCESSED: NO")
