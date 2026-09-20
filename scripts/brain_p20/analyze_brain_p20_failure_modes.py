import numpy as np
import pandas as pd
from pathlib import Path

IN = Path("part2/brain400_p20_controlled/brain400_p20.csv")
OUT = Path("part2/brain400_p20_controlled")

df = pd.read_csv(IN)
P = [f"P{i:02d}" for i in range(1,21)]
D = df[[f"{p}_dice" for p in P]].to_numpy(float)

# --------------------------------------------------
# Per-case distribution decomposition
# --------------------------------------------------
case_min = D.min(1)
case_max = D.max(1)
case_med = np.median(D, axis=1)
case_q1 = np.quantile(D, .25, axis=1)
case_q3 = np.quantile(D, .75, axis=1)
case_range = case_max - case_min

n_zero = (D <= 1e-8).sum(1)
n_bad05 = (D < .05).sum(1)
n_bad10 = (D < .10).sum(1)
n_good50 = (D >= .50).sum(1)
n_good70 = (D >= .70).sum(1)

best_minus_med = case_max - case_med
med_minus_worst = case_med - case_min

res = pd.DataFrame({
    "filename": df.filename,
    "min": case_min,
    "q1": case_q1,
    "median": case_med,
    "q3": case_q3,
    "max": case_max,
    "range": case_range,
    "best_minus_median": best_minus_med,
    "median_minus_worst": med_minus_worst,
    "n_zero": n_zero,
    "n_dice_lt_05": n_bad05,
    "n_dice_lt_10": n_bad10,
    "n_dice_ge_50": n_good50,
    "n_dice_ge_70": n_good70,
})
res.to_csv(OUT/"brain400_p20_failure_modes.csv", index=False)

large = case_range > .50

print("=== LARGE-RANGE CASES ===")
print("N range > .50:", large.sum())
print("Mean # exact-zero prompts:", n_zero[large].mean())
print("Median # exact-zero prompts:", np.median(n_zero[large]))
print("Mean # Dice < .05 prompts:", n_bad05[large].mean())
print("Median # Dice < .05 prompts:", np.median(n_bad05[large]))
print("Mean # Dice >= .50 prompts:", n_good50[large].mean())
print("Median # Dice >= .50 prompts:", np.median(n_good50[large]))

print("\n=== SHAPE OF LARGE-RANGE DISTRIBUTIONS ===")
print("Mean best-median gap:", best_minus_med[large].mean())
print("Mean median-worst gap:", med_minus_worst[large].mean())
print("Median best-median gap:", np.median(best_minus_med[large]))
print("Median median-worst gap:", np.median(med_minus_worst[large]))

# A simple interpretable decomposition.
# Catastrophic-tail: median remains good but at least one prompt collapses.
cat_tail = large & (case_med >= .50) & (case_min < .10)

# Rescue-tail: median is poor but at least one prompt succeeds.
rescue_tail = large & (case_med < .20) & (case_max >= .50)

# Mixed: substantial bad and good prompt subsets.
mixed = large & (n_bad10 >= 3) & (n_good50 >= 3)

print("\n=== INTERPRETABLE LARGE-RANGE PATTERNS ===")
print(f"Catastrophic-tail (median>=.50, min<.10): {cat_tail.sum()}/{large.sum()}")
print(f"Rescue-tail (median<.20, max>=.50): {rescue_tail.sum()}/{large.sum()}")
print(f"Mixed (>=3 prompts <.10 AND >=3 prompts >=.50): {mixed.sum()}/{large.sum()}")

# --------------------------------------------------
# Prompt-wise failure profile
# --------------------------------------------------
print("\n=== PROMPT-WISE PROFILE ===")
print("prompt mean median zero% <.05% <.10% >=.50%")

for j,p in enumerate(P):
    x = D[:,j]
    print(
        f"{p} "
        f"{x.mean():.4f} "
        f"{np.median(x):.4f} "
        f"{100*np.mean(x<=1e-8):.1f}% "
        f"{100*np.mean(x<.05):.1f}% "
        f"{100*np.mean(x<.10):.1f}% "
        f"{100*np.mean(x>=.50):.1f}%"
    )

# --------------------------------------------------
# Direct P14 vs P15
# --------------------------------------------------
a = D[:,13]
b = D[:,14]
delta = a-b

print("\n=== P14 vs P15 ===")
print("P14 mean:", a.mean())
print("P15 mean:", b.mean())
print("Mean P14-P15:", delta.mean())
print("Median P14-P15:", np.median(delta))
print("Mean |difference|:", np.mean(np.abs(delta)))
print("|difference| > .10:", np.mean(np.abs(delta)>.10))
print("|difference| > .20:", np.mean(np.abs(delta)>.20))
print("|difference| > .50:", np.mean(np.abs(delta)>.50))
print("P14 > P15 by >.20:", np.mean(delta>.20))
print("P15 > P14 by >.20:", np.mean(delta<-.20))

# --------------------------------------------------
# Box diversity: how many distinct localization boxes
# each case receives across 20 prompts
# --------------------------------------------------
box_counts = []

for _,r in df.iterrows():
    boxes = []
    for p in P:
        boxes.append((
            r[f"{p}_x1"], r[f"{p}_y1"],
            r[f"{p}_x2"], r[f"{p}_y2"]
        ))
    box_counts.append(len(set(boxes)))

box_counts = np.asarray(box_counts)

print("\n=== LOCALIZATION BOX DIVERSITY ===")
print("Mean distinct boxes/case:", box_counts.mean())
print("Median distinct boxes/case:", np.median(box_counts))
print("Max distinct boxes/case:", box_counts.max())
print("Cases with 1 box:", np.sum(box_counts==1))
print("Cases with >=5 boxes:", np.sum(box_counts>=5))
print("Cases with >=10 boxes:", np.sum(box_counts>=10))

print("\nLarge-range cases:")
print("Mean distinct boxes:", box_counts[large].mean())
print("Median distinct boxes:", np.median(box_counts[large]))

print("\nNon-large-range cases:")
print("Mean distinct boxes:", box_counts[~large].mean())
print("Median distinct boxes:", np.median(box_counts[~large]))

print("\nSaved:", OUT/"brain400_p20_failure_modes.csv")
