import pandas as pd
import numpy as np
from itertools import combinations

CSV = "part2/breast97_p20_controlled/breast97_p20.csv"
OUT = "part2/breast97_p20_controlled/breast97_p20_summary.csv"

df = pd.read_csv(CSV)
prompts = [f"P{i:02d}" for i in range(1, 21)]
cols = [f"{p}_dice" for p in prompts]

assert len(df) == 97
assert all(c in df.columns for c in cols)
assert not df[cols].isna().any().any()

X = df[cols].to_numpy(float)
n, k = X.shape
assert (n, k) == (97, 20)

# ------------------------------------------------------------
# PRE-SPECIFIED CASE-LEVEL ENDPOINTS
# ------------------------------------------------------------

pairs = list(combinations(range(k), 2))
pair_abs = np.stack(
    [np.abs(X[:, i] - X[:, j]) for i, j in pairs],
    axis=1
)

case_pairwise = pair_abs.mean(axis=1)
case_range = X.max(axis=1) - X.min(axis=1)

prompt_means = X.mean(axis=0)

best_fixed_idx = int(np.argmax(prompt_means))
best_fixed_name = prompts[best_fixed_idx]
best_fixed_mean = float(prompt_means[best_fixed_idx])

oracle = X.max(axis=1)
oracle_mean = float(oracle.mean())
oracle_gain = oracle_mean - best_fixed_mean

print("=" * 72)
print("CONTROLLED BREAST P20 — PRIMARY ANALYSIS")
print("=" * 72)
print("N cases:", n)
print("N prompts:", k)
print("N prompt pairs:", len(pairs))

print("\nPRIMARY ENDPOINT")
print(
    "Mean per-case pairwise absolute Dice difference:",
    f"{case_pairwise.mean():.9f}"
)
print(
    "Median per-case pairwise absolute Dice difference:",
    f"{np.median(case_pairwise):.9f}"
)

print("\nSECONDARY: WITHIN-CASE RANGE")
print("Mean range:", f"{case_range.mean():.9f}")
print("Median range:", f"{np.median(case_range):.9f}")
print("Max range:", f"{case_range.max():.9f}")

for t in [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]:
    c = int((case_range > t).sum())
    print(
        f"range > {t:.2f}: {c}/{n} "
        f"({100*c/n:.1f}%)"
    )

print("\nPROMPT-WISE MEAN DICE")
for p, m in zip(prompts, prompt_means):
    print(f"{p}: {m:.9f}")

print("\nFIXED vs ORACLE")
print("Best fixed prompt:", best_fixed_name)
print("Best fixed mean Dice:", f"{best_fixed_mean:.9f}")
print("Prompt oracle mean Dice:", f"{oracle_mean:.9f}")
print("Oracle - best fixed:", f"{oracle_gain:.9f}")

# ------------------------------------------------------------
# BOOTSTRAP — CASE LEVEL, 10k, RNG42
# ------------------------------------------------------------

rng = np.random.default_rng(42)
B = 10000

boot_pair = np.empty(B)
boot_range = np.empty(B)
boot_oracle_gain = np.empty(B)

boot_thresholds = {
    t: np.empty(B)
    for t in [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]
}

# Important:
# Re-select best fixed prompt inside each bootstrap replicate.
# This estimates the oracle-vs-best-fixed quantity under resampling.
for b in range(B):
    idx = rng.integers(0, n, n)

    xb = X[idx]
    cp = case_pairwise[idx]
    cr = case_range[idx]

    boot_pair[b] = cp.mean()
    boot_range[b] = cr.mean()

    pm = xb.mean(axis=0)
    best_fixed_b = pm.max()
    oracle_b = xb.max(axis=1).mean()
    boot_oracle_gain[b] = oracle_b - best_fixed_b

    for t in boot_thresholds:
        boot_thresholds[t][b] = (cr > t).mean()

def ci(a):
    return np.percentile(a, [2.5, 97.5])

print("\nBOOTSTRAP 95% CIs — 10,000 CASE-LEVEL RESAMPLES")

lo, hi = ci(boot_pair)
print(
    "mean_pairwise_abs:",
    f"{case_pairwise.mean():.9f}",
    f"[{lo:.9f}, {hi:.9f}]"
)

lo, hi = ci(boot_range)
print(
    "mean_range:",
    f"{case_range.mean():.9f}",
    f"[{lo:.9f}, {hi:.9f}]"
)

for t, arr in boot_thresholds.items():
    point = (case_range > t).mean()
    lo, hi = ci(arr)
    print(
        f"range>{t:.2f}:",
        f"{point:.6f}",
        f"[{lo:.6f}, {hi:.6f}]"
    )

lo, hi = ci(boot_oracle_gain)
print(
    "oracle_over_best_fixed:",
    f"{oracle_gain:.9f}",
    f"[{lo:.9f}, {hi:.9f}]"
)

# ------------------------------------------------------------
# SAVE COMPACT SUMMARY
# ------------------------------------------------------------

rows = [
    ("N", n, np.nan, np.nan),
    ("mean_pairwise_abs", case_pairwise.mean(), *ci(boot_pair)),
    ("median_pairwise_abs", np.median(case_pairwise), np.nan, np.nan),
    ("mean_range", case_range.mean(), *ci(boot_range)),
    ("median_range", np.median(case_range), np.nan, np.nan),
    ("max_range", case_range.max(), np.nan, np.nan),
    ("best_fixed_mean", best_fixed_mean, np.nan, np.nan),
    ("oracle_mean", oracle_mean, np.nan, np.nan),
    ("oracle_over_best_fixed", oracle_gain, *ci(boot_oracle_gain)),
]

for t, arr in boot_thresholds.items():
    rows.append((
        f"range_gt_{t:.2f}",
        (case_range > t).mean(),
        *ci(arr)
    ))

summary = pd.DataFrame(
    rows,
    columns=["metric", "point", "ci_low", "ci_high"]
)
summary.to_csv(OUT, index=False)

# Save case-level quantities for later pre-specified visualization.
caseout = pd.DataFrame({
    "filename": df["filename"],
    "mean_pairwise_abs": case_pairwise,
    "prompt_range": case_range,
    "prompt_min": X.min(axis=1),
    "prompt_max": X.max(axis=1),
    "prompt_oracle": oracle,
})
caseout.to_csv(
    "part2/breast97_p20_controlled/breast97_p20_case_metrics.csv",
    index=False
)

print("\nSaved:", OUT)
print("Saved: part2/breast97_p20_controlled/breast97_p20_case_metrics.csv")
