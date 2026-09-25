from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

ROOT = Path("/home/hpc/rlvl/rlvl178v/prl_medclipsam")

IN = ROOT / "beyond-average-dice/remedy/results/stochastic_factorial/factorial_boxes.csv"
OUTDIR = ROOT / "beyond-average-dice/remedy/results/stochastic_factorial"
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN)

assert len(df) == 1200
assert df["filename"].nunique() == 12
assert df["prompt_id"].nunique() == 20
assert df["seed"].nunique() == 5

BOX = ["x1", "y1", "x2", "y2"]

def box_iou(a, b):
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1 + 1.0)
    ih = max(0.0, iy2 - iy1 + 1.0)
    inter = iw * ih

    aa = max(0.0, ax2 - ax1 + 1.0) * max(0.0, ay2 - ay1 + 1.0)
    ba = max(0.0, bx2 - bx1 + 1.0) * max(0.0, by2 - by1 + 1.0)

    union = aa + ba - inter

    if union <= 0:
        return 1.0

    return inter / union


# ------------------------------------------------------------
# 1. Seed variation:
# hold case + prompt fixed, vary seed.
#
# 5 seeds -> C(5,2)=10 comparisons per prompt.
# 20 prompts -> 200 seed-pair comparisons per case.
# ------------------------------------------------------------

seed_rows = []

for (fn, region, risk, prompt), g in df.groupby(
    ["filename", "risk_region", "frozen_risk", "prompt_id"],
    sort=False
):
    g = g.sort_values("seed")

    assert len(g) == 5

    recs = g.to_dict("records")

    for a, b in combinations(recs, 2):
        iou = box_iou(
            [a[c] for c in BOX],
            [b[c] for c in BOX]
        )

        seed_rows.append({
            "filename": fn,
            "risk_region": region,
            "frozen_risk": risk,
            "prompt_id": prompt,
            "seed_a": a["seed"],
            "seed_b": b["seed"],
            "box_iou": iou,
            "instability_1_minus_iou": 1.0 - iou,
        })

seed_pairs = pd.DataFrame(seed_rows)

assert len(seed_pairs) == 12 * 20 * 10


# ------------------------------------------------------------
# 2. Language variation:
# hold case + seed fixed, vary prompt.
#
# 20 prompts -> C(20,2)=190 comparisons per seed.
# 5 seeds -> 950 language-pair comparisons per case.
# ------------------------------------------------------------

lang_rows = []

for (fn, region, risk, seed), g in df.groupby(
    ["filename", "risk_region", "frozen_risk", "seed"],
    sort=False
):
    g = g.sort_values("prompt_id")

    assert len(g) == 20

    recs = g.to_dict("records")

    for a, b in combinations(recs, 2):
        iou = box_iou(
            [a[c] for c in BOX],
            [b[c] for c in BOX]
        )

        lang_rows.append({
            "filename": fn,
            "risk_region": region,
            "frozen_risk": risk,
            "seed": seed,
            "prompt_a": a["prompt_id"],
            "prompt_b": b["prompt_id"],
            "box_iou": iou,
            "instability_1_minus_iou": 1.0 - iou,
        })

lang_pairs = pd.DataFrame(lang_rows)

assert len(lang_pairs) == 12 * 5 * 190


# ------------------------------------------------------------
# Case-level decomposition
# ------------------------------------------------------------

seed_case = (
    seed_pairs
    .groupby(["filename", "risk_region", "frozen_risk"])
    ["instability_1_minus_iou"]
    .agg(["mean", "median", "std", "min", "max"])
    .reset_index()
    .rename(columns={
        "mean": "seed_instability_mean",
        "median": "seed_instability_median",
        "std": "seed_instability_sd",
        "min": "seed_instability_min",
        "max": "seed_instability_max",
    })
)

lang_case = (
    lang_pairs
    .groupby(["filename", "risk_region", "frozen_risk"])
    ["instability_1_minus_iou"]
    .agg(["mean", "median", "std", "min", "max"])
    .reset_index()
    .rename(columns={
        "mean": "language_instability_mean",
        "median": "language_instability_median",
        "std": "language_instability_sd",
        "min": "language_instability_min",
        "max": "language_instability_max",
    })
)

case = seed_case.merge(
    lang_case,
    on=["filename", "risk_region", "frozen_risk"],
    validate="one_to_one"
)

case["language_minus_seed"] = (
    case["language_instability_mean"] -
    case["seed_instability_mean"]
)

case["language_to_seed_ratio"] = np.where(
    case["seed_instability_mean"] > 0,
    case["language_instability_mean"] /
    case["seed_instability_mean"],
    np.inf
)

case["language_fraction_of_combined"] = (
    case["language_instability_mean"] /
    (
        case["language_instability_mean"] +
        case["seed_instability_mean"]
    )
)


# ------------------------------------------------------------
# Region summary
# ------------------------------------------------------------

region = (
    case.groupby("risk_region", sort=False)
    .agg(
        n_cases=("filename", "size"),
        frozen_risk_mean=("frozen_risk", "mean"),
        seed_instability_mean=("seed_instability_mean", "mean"),
        language_instability_mean=("language_instability_mean", "mean"),
        language_minus_seed_mean=("language_minus_seed", "mean"),
        language_to_seed_ratio_mean=("language_to_seed_ratio", "mean"),
        language_fraction_mean=("language_fraction_of_combined", "mean"),
    )
    .reset_index()
)


# ------------------------------------------------------------
# Overall descriptive summary
# ------------------------------------------------------------

print("=== FACTORIAL STOCHASTICITY DECOMPOSITION ===")
print("cases:", len(case))
print("raw runs:", len(df))
print("seed-pair comparisons:", len(seed_pairs))
print("language-pair comparisons:", len(lang_pairs))

print("\n=== OVERALL CASE-LEVEL MEANS ===")

S = case["seed_instability_mean"].mean()
L = case["language_instability_mean"].mean()

print(f"mean seed instability:     {S:.6f}")
print(f"mean language instability: {L:.6f}")
print(f"language - seed:           {L-S:+.6f}")

if S > 0:
    print(f"language / seed ratio:     {L/S:.6f}")

print(
    "language fraction combined:",
    f"{L/(L+S):.6f}"
)

print("\n=== CASEWISE DOMINANCE ===")

n_lang = int(
    (case["language_instability_mean"] >
     case["seed_instability_mean"]).sum()
)

n_seed = int(
    (case["language_instability_mean"] <
     case["seed_instability_mean"]).sum()
)

n_equal = len(case) - n_lang - n_seed

print("language > seed:", n_lang, "/", len(case))
print("seed > language:", n_seed, "/", len(case))
print("equal:", n_equal)

print("\n=== BY RISK REGION ===")
print(region.to_string(index=False))

print("\n=== CASE TABLE ===")
print(
    case[
        [
            "risk_region",
            "filename",
            "frozen_risk",
            "seed_instability_mean",
            "language_instability_mean",
            "language_minus_seed",
            "language_to_seed_ratio",
            "language_fraction_of_combined",
        ]
    ].to_string(index=False)
)

seed_pairs.to_csv(
    OUTDIR / "seed_pairwise_instability.csv",
    index=False
)

lang_pairs.to_csv(
    OUTDIR / "language_pairwise_instability.csv",
    index=False
)

case.to_csv(
    OUTDIR / "case_stochasticity_decomposition.csv",
    index=False
)

region.to_csv(
    OUTDIR / "region_stochasticity_decomposition.csv",
    index=False
)

print("\nDEVELOPMENT DIAGNOSTIC ONLY: YES")
print("GROUND TRUTH USED: NO")
print("HELDOUT OUTCOMES ACCESSED: NO")
print("METHOD/THRESHOLD TUNING: NO")
