import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path

IN = Path("part2/brain400_p20_controlled/brain400_p20.csv")
OUT = Path("part2/brain400_p20_controlled")

df = pd.read_csv(IN)
P = [f"P{i:02d}" for i in range(1, 21)]

def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    aa = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    ba = max(0.0, bx2-bx1) * max(0.0, by2-by1)
    union = aa + ba - inter

    return inter / union if union > 0 else 0.0

rows = []

for case_idx, r in df.iterrows():
    for i, j in combinations(range(20), 2):
        p = P[i]
        q = P[j]

        d1 = float(r[f"{p}_dice"])
        d2 = float(r[f"{q}_dice"])

        b1 = (
            float(r[f"{p}_x1"]), float(r[f"{p}_y1"]),
            float(r[f"{p}_x2"]), float(r[f"{p}_y2"])
        )
        b2 = (
            float(r[f"{q}_x1"]), float(r[f"{q}_y1"]),
            float(r[f"{q}_x2"]), float(r[f"{q}_y2"])
        )

        biou = box_iou(b1, b2)

        rows.append({
            "filename": r["filename"],
            "p1": p,
            "p2": q,
            "dice1": d1,
            "dice2": d2,
            "abs_dice_diff": abs(d1-d2),
            "box_iou": biou,
        })

pairs = pd.DataFrame(rows)
pairs.to_csv(OUT/"brain400_p20_pairwise_localization.csv", index=False)

print("Pair observations:", len(pairs))
assert len(pairs) == 400 * 190

# -------------------------------------------------------
# Overall relationship
# -------------------------------------------------------
pearson = np.corrcoef(
    pairs["box_iou"], pairs["abs_dice_diff"]
)[0,1]

spearman = pairs[
    ["box_iou", "abs_dice_diff"]
].corr(method="spearman").iloc[0,1]

print("\n=== OVERALL RELATIONSHIP ===")
print("Pearson corr(box IoU, |Dice diff|):", pearson)
print("Spearman corr(box IoU, |Dice diff|):", spearman)

# -------------------------------------------------------
# Dice divergence conditional on localization similarity
# -------------------------------------------------------
bins = [
    ("boxIoU < .10", pairs.box_iou < .10),
    ("boxIoU < .25", pairs.box_iou < .25),
    ("boxIoU < .50", pairs.box_iou < .50),
    ("boxIoU >= .50", pairs.box_iou >= .50),
    ("boxIoU >= .75", pairs.box_iou >= .75),
    ("boxIoU >= .90", pairs.box_iou >= .90),
]

print("\n=== DICE DIVERGENCE BY BOX SIMILARITY ===")
for name, mask in bins:
    x = pairs.loc[mask, "abs_dice_diff"]
    if len(x):
        print(
            f"{name}: n={len(x)} "
            f"mean|dDice|={x.mean():.4f} "
            f"median={x.median():.4f} "
            f"P(|dDice|>.10)={(x>.10).mean():.3f} "
            f"P(|dDice|>.20)={(x>.20).mean():.3f} "
            f"P(|dDice|>.50)={(x>.50).mean():.3f}"
        )

# -------------------------------------------------------
# Localization similarity conditional on Dice divergence
# -------------------------------------------------------
print("\n=== BOX IoU BY DICE DIVERGENCE ===")
for t in [.01, .05, .10, .20, .50]:
    mask = pairs.abs_dice_diff > t
    x = pairs.loc[mask, "box_iou"]

    print(
        f"|dDice|>{t:.2f}: n={len(x)} "
        f"mean boxIoU={x.mean():.4f} "
        f"median={x.median():.4f} "
        f"P(boxIoU<.10)={(x<.10).mean():.3f} "
        f"P(boxIoU<.25)={(x<.25).mean():.3f} "
        f"P(boxIoU<.50)={(x<.50).mean():.3f}"
    )

# -------------------------------------------------------
# Explicit catastrophic switch definition
# One prompt fails (<.10), the other succeeds (>=.50)
# -------------------------------------------------------
cat = (
    ((pairs.dice1 < .10) & (pairs.dice2 >= .50)) |
    ((pairs.dice2 < .10) & (pairs.dice1 >= .50))
)

x = pairs.loc[cat, "box_iou"]

print("\n=== CATASTROPHIC FAILURE/SUCCESS PAIRS ===")
print("N:", len(x))
print("Fraction of all pairs:", cat.mean())
print("Mean box IoU:", x.mean())
print("Median box IoU:", x.median())
print("boxIoU < .10:", (x < .10).mean())
print("boxIoU < .25:", (x < .25).mean())
print("boxIoU < .50:", (x < .50).mean())
print("boxIoU >= .75:", (x >= .75).mean())

# -------------------------------------------------------
# P14 vs P15 specifically
# -------------------------------------------------------
p14 = pairs[(pairs.p1=="P14") & (pairs.p2=="P15")].copy()

print("\n=== P14 vs P15 LOCALIZATION ===")
print("N:", len(p14))
print("Mean box IoU:", p14.box_iou.mean())
print("Median box IoU:", p14.box_iou.median())

for t in [.10, .20, .50]:
    z = p14[p14.abs_dice_diff > t]
    print(
        f"|dDice|>{t:.2f}: n={len(z)}, "
        f"mean boxIoU={z.box_iou.mean():.4f}, "
        f"median={z.box_iou.median():.4f}, "
        f"P(boxIoU<.25)={(z.box_iou<.25).mean():.3f}"
    )

# -------------------------------------------------------
# Same/near-same boxes but large Dice differences:
# important counterexample check.
# -------------------------------------------------------
counter = pairs[
    (pairs.box_iou >= .90) &
    (pairs.abs_dice_diff > .20)
].copy()

print("\n=== COUNTEREXAMPLE CHECK ===")
print("boxIoU >= .90 AND |dDice| > .20:", len(counter))
print("Fraction of all pairs:", len(counter)/len(pairs))

counter.sort_values(
    "abs_dice_diff", ascending=False
).head(50).to_csv(
    OUT/"brain400_p20_localization_counterexamples.csv",
    index=False
)

print("\nSaved:")
print(OUT/"brain400_p20_pairwise_localization.csv")
print(OUT/"brain400_p20_localization_counterexamples.csv")
print("\nLOCALIZATION ANALYSIS COMPLETE")
