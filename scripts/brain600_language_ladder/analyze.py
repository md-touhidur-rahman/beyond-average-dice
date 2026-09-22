import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

ROOT = Path("/home/hpc/rlvl/rlvl178v/prl_medclipsam")
V2 = ROOT / "MedCLIP-SAMv2"
BASE = V2 / "parent_repro_brain600"
GT_DIR = ROOT / "authors_segmentation_data/data/brain_tumors/test_masks"
OUT = ROOT / "beyond-average-dice/results/brain600_language_ladder"
OUT.mkdir(parents=True, exist_ok=True)

CONDS = {
    "H0": {
        "sam": BASE / "sam",
        "coarse": BASE / "coarse",
    },
    "H1": {
        "sam": BASE / "highop/H1_verb/sam",
        "coarse": BASE / "highop/H1_verb/coarse",
    },
    "H2": {
        "sam": BASE / "highop/H2_intro/sam",
        "coarse": BASE / "highop/H2_intro/coarse",
    },
    "L3": {
        "sam": BASE / "language_ladder/L3_semantic/sam",
        "coarse": BASE / "language_ladder/L3_semantic/coarse",
    },
    "L4": {
        "sam": BASE / "language_ladder/L4_concise/sam",
        "coarse": BASE / "language_ladder/L4_concise/coarse",
    },
    "L5": {
        "sam": BASE / "language_ladder/L5_broad/sam",
        "coarse": BASE / "language_ladder/L5_broad/coarse",
    },
}

ORDER = list(CONDS)
RNG_SEED = 42
N_BOOT = 10000

def files_by_stem(directory):
    d = {}
    for p in sorted(directory.iterdir()):
        if p.is_file() and not p.name.startswith("."):
            if p.stem in d:
                raise RuntimeError(f"Duplicate stem {p.stem} in {directory}")
            d[p.stem] = p
    return d

def read_binary(path):
    x = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if x is None:
        raise RuntimeError(f"Cannot read {path}")
    if x.ndim == 3:
        x = x[..., 0]
    return x > 0

def dice(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return 2.0 * np.logical_and(a, b).sum() / denom

def iou(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return np.logical_and(a, b).sum() / union

def bbox_from_mask(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

def bbox_iou(a, b):
    if a is None and b is None:
        return 1.0
    if a is None or b is None:
        return 0.0

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1 + 1)
    ih = max(0, iy2 - iy1 + 1)
    inter = iw * ih

    area_a = (ax2 - ax1 + 1) * (ay2 - ay1 + 1)
    area_b = (bx2 - bx1 + 1) * (by2 - by1 + 1)
    union = area_a + area_b - inter

    return inter / union if union else 1.0

def ci(x):
    x = np.asarray(x, dtype=float)
    return (
        float(np.quantile(x, 0.025)),
        float(np.quantile(x, 0.975)),
    )

# --------------------------------------------------
# Resolve case identities by stem
# --------------------------------------------------

gt = files_by_stem(GT_DIR)
print("GT:", len(gt))

sam_maps = {}
coarse_maps = {}

for c in ORDER:
    sam_maps[c] = files_by_stem(CONDS[c]["sam"])
    coarse_maps[c] = files_by_stem(CONDS[c]["coarse"])
    print(
        c,
        "SAM =", len(sam_maps[c]),
        "COARSE =", len(coarse_maps[c])
    )

common = set(gt)

for c in ORDER:
    common &= set(sam_maps[c])
    common &= set(coarse_maps[c])

cases = sorted(common)

print("\nCOMMON CASES:", len(cases))

if len(cases) != 600:
    # Diagnose mismatched naming before silently continuing.
    print("\nExample GT stems:", list(sorted(gt))[:5])
    for c in ORDER:
        print(c, "SAM stems:", list(sorted(sam_maps[c]))[:5])
        print(c, "coarse stems:", list(sorted(coarse_maps[c]))[:5])
    raise RuntimeError(
        f"Expected exactly 600 common cases, found {len(cases)}. "
        "No results were analyzed."
    )

# --------------------------------------------------
# Dice + coarse localization boxes
# --------------------------------------------------

rows = []
boxes = {c: {} for c in ORDER}

for j, case in enumerate(cases):
    y = read_binary(gt[case])

    row = {"case": case}

    for c in ORDER:
        pred = read_binary(sam_maps[c][case])

        if pred.shape != y.shape:
            raise RuntimeError(
                f"Geometry mismatch {case} {c}: "
                f"GT={y.shape}, pred={pred.shape}"
            )

        row[f"dice_{c}"] = dice(y, pred)

        cmask = read_binary(coarse_maps[c][case])
        boxes[c][case] = bbox_from_mask(cmask)

    rows.append(row)

    if (j + 1) % 100 == 0:
        print("processed", j + 1, "/", len(cases))

df = pd.DataFrame(rows)

# --------------------------------------------------
# H0-relative outcomes
# --------------------------------------------------

for c in ORDER[1:]:
    df[f"signed_{c}"] = df[f"dice_{c}"] - df["dice_H0"]
    df[f"abs_{c}"] = np.abs(df[f"signed_{c}"])

    df[f"switch_{c}"] = (
        (np.minimum(df["dice_H0"], df[f"dice_{c}"]) < 0.10)
        &
        (np.maximum(df["dice_H0"], df[f"dice_{c}"]) >= 0.50)
    )

    df[f"box_iou_{c}"] = [
        bbox_iou(boxes["H0"][case], boxes[c][case])
        for case in cases
    ]

dice_cols = [f"dice_{c}" for c in ORDER]

df["range_H0_L5"] = df[dice_cols].max(axis=1) - df[dice_cols].min(axis=1)
df["oracle_H0_L5"] = df[dice_cols].max(axis=1)

df.to_csv(OUT / "case_metrics.csv", index=False)

# --------------------------------------------------
# Descriptive condition performance
# --------------------------------------------------

condition_rows = []

for c in ORDER:
    x = df[f"dice_{c}"].to_numpy()

    condition_rows.append({
        "condition": c,
        "n": len(x),
        "mean_dice": x.mean(),
        "median_dice": np.median(x),
        "std_dice": x.std(ddof=1),
        "dice_lt_010": np.mean(x < .10),
        "dice_ge_050": np.mean(x >= .50),
        "dice_ge_070": np.mean(x >= .70),
        "dice_ge_080": np.mean(x >= .80),
    })

condition_df = pd.DataFrame(condition_rows)
condition_df.to_csv(OUT / "condition_summary.csv", index=False)

# --------------------------------------------------
# H0-relative descriptive endpoints
# --------------------------------------------------

comparison_rows = []

for c in ORDER[1:]:
    signed = df[f"signed_{c}"].to_numpy()
    absolute = df[f"abs_{c}"].to_numpy()
    biou = df[f"box_iou_{c}"].to_numpy()

    comparison_rows.append({
        "condition": c,
        "n": len(df),
        "mean_abs_delta": absolute.mean(),
        "median_abs_delta": np.median(absolute),
        "mean_signed_delta": signed.mean(),
        "abs_gt_010": np.mean(absolute > .10),
        "abs_gt_020": np.mean(absolute > .20),
        "abs_gt_050": np.mean(absolute > .50),
        "failure_success_switch": df[f"switch_{c}"].mean(),
        "mean_box_iou_H0": biou.mean(),
        "median_box_iou_H0": np.median(biou),
        "pearson_boxiou_absdelta":
            np.corrcoef(biou, absolute)[0,1]
            if np.std(biou) > 0 and np.std(absolute) > 0 else np.nan,
        "spearman_boxiou_absdelta":
            pd.Series(biou).corr(pd.Series(absolute), method="spearman"),
    })

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv(OUT / "h0_relative_summary.csv", index=False)

# --------------------------------------------------
# All 15 pairwise condition comparisons
# --------------------------------------------------

pair_rows = []

for a, b in combinations(ORDER, 2):
    da = df[f"dice_{a}"].to_numpy()
    db = df[f"dice_{b}"].to_numpy()
    absolute = np.abs(da - db)

    pair_rows.append({
        "condition_a": a,
        "condition_b": b,
        "mean_abs_delta": absolute.mean(),
        "median_abs_delta": np.median(absolute),
        "abs_gt_010": np.mean(absolute > .10),
        "abs_gt_020": np.mean(absolute > .20),
        "abs_gt_050": np.mean(absolute > .50),
        "failure_success_switch": np.mean(
            (np.minimum(da,db) < .10) &
            (np.maximum(da,db) >= .50)
        ),
    })

pd.DataFrame(pair_rows).to_csv(
    OUT / "all_pairwise_summary.csv", index=False
)

# --------------------------------------------------
# Full six-condition sensitivity/recoverability
# --------------------------------------------------

means = {c: df[f"dice_{c}"].mean() for c in ORDER}
best_fixed = max(ORDER, key=lambda c: means[c])
best_fixed_mean = means[best_fixed]

oracle = df["oracle_H0_L5"].to_numpy()
oracle_mean = oracle.mean()
oracle_advantage = oracle_mean - best_fixed_mean

range_x = df["range_H0_L5"].to_numpy()

full_summary = pd.DataFrame([{
    "n": len(df),
    "mean_range": range_x.mean(),
    "median_range": np.median(range_x),
    "max_range": range_x.max(),
    "range_gt_001": np.mean(range_x > .01),
    "range_gt_005": np.mean(range_x > .05),
    "range_gt_010": np.mean(range_x > .10),
    "range_gt_020": np.mean(range_x > .20),
    "range_gt_030": np.mean(range_x > .30),
    "range_gt_050": np.mean(range_x > .50),
    "best_fixed_condition": best_fixed,
    "best_fixed_mean_dice": best_fixed_mean,
    "oracle_mean_dice": oracle_mean,
    "oracle_advantage": oracle_advantage,
}])

full_summary.to_csv(OUT / "six_condition_summary.csv", index=False)

# --------------------------------------------------
# Frozen image bootstrap: 10k, seed 42
# --------------------------------------------------

rng = np.random.default_rng(RNG_SEED)
n = len(df)

boot_comparison = {c: {
    "mean_abs_delta": [],
    "mean_signed_delta": [],
    "failure_success_switch": [],
} for c in ORDER[1:]}

boot_range = []
boot_oracle = []
boot_best_fixed = []
boot_advantage = []

dice_matrix = df[dice_cols].to_numpy()

print("\nBOOTSTRAP:", N_BOOT)

for r in range(N_BOOT):
    idx = rng.integers(0, n, n)

    sample = df.iloc[idx]

    for c in ORDER[1:]:
        boot_comparison[c]["mean_abs_delta"].append(
            sample[f"abs_{c}"].mean()
        )
        boot_comparison[c]["mean_signed_delta"].append(
            sample[f"signed_{c}"].mean()
        )
        boot_comparison[c]["failure_success_switch"].append(
            sample[f"switch_{c}"].mean()
        )

    mat = dice_matrix[idx]

    ranges = mat.max(axis=1) - mat.min(axis=1)
    oracle_b = mat.max(axis=1).mean()

    # Reselect best fixed condition in every bootstrap replicate.
    fixed_means = mat.mean(axis=0)
    fixed_b = fixed_means.max()

    boot_range.append(ranges.mean())
    boot_oracle.append(oracle_b)
    boot_best_fixed.append(fixed_b)
    boot_advantage.append(oracle_b - fixed_b)

bootstrap_rows = []

for c in ORDER[1:]:
    for metric, vals in boot_comparison[c].items():
        lo, hi = ci(vals)
        bootstrap_rows.append({
            "scope": c,
            "metric": metric,
            "ci_low": lo,
            "ci_high": hi,
            "n_boot": N_BOOT,
            "seed": RNG_SEED,
        })

for metric, vals in [
    ("mean_range_H0_L5", boot_range),
    ("oracle_mean_H0_L5", boot_oracle),
    ("best_fixed_mean_H0_L5", boot_best_fixed),
    ("oracle_advantage_H0_L5", boot_advantage),
]:
    lo, hi = ci(vals)
    bootstrap_rows.append({
        "scope": "H0_L5",
        "metric": metric,
        "ci_low": lo,
        "ci_high": hi,
        "n_boot": N_BOOT,
        "seed": RNG_SEED,
    })

bootstrap_df = pd.DataFrame(bootstrap_rows)
bootstrap_df.to_csv(OUT / "bootstrap_95ci.csv", index=False)

# --------------------------------------------------
# Compact frozen-result printout
# --------------------------------------------------

print("\n==========================================")
print("CONDITION PERFORMANCE")
print("==========================================")
print(condition_df.to_string(index=False))

print("\n==========================================")
print("H0-RELATIVE RESULTS")
print("==========================================")
print(comparison_df.to_string(index=False))

print("\n==========================================")
print("SIX-CONDITION RESULTS")
print("==========================================")
print(full_summary.to_string(index=False))

print("\n==========================================")
print("BOOTSTRAP 95% CIs")
print("==========================================")
print(bootstrap_df.to_string(index=False))

print("\n==========================================")
print("TOP 15 CASES BY SIX-CONDITION RANGE")
print("==========================================")
show = ["case"] + dice_cols + ["range_H0_L5", "oracle_H0_L5"]
print(
    df.sort_values("range_H0_L5", ascending=False)[show]
      .head(15)
      .to_string(index=False)
)

print("\nWROTE:", OUT)
