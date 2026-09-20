import re
import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path

IN = Path("part2/brain400_p20_controlled/brain400_p20.csv")
OUT = Path("part2/brain400_p20_controlled")
OUT.mkdir(parents=True, exist_ok=True)

N_BOOT = 10000
SEED = 42

df = pd.read_csv(IN)

prompts = [f"P{i:02d}" for i in range(1, 21)]
dice_cols = [f"{p}_dice" for p in prompts]

assert len(df) == 400
assert all(c in df.columns for c in dice_cols)
assert not df[dice_cols].isna().any().any()

# ------------------------------------------------------------
# Patient identity
# ------------------------------------------------------------
def get_pid(filename):
    m = re.search(r"pid-([^_\.]+)", str(filename))
    if m is None:
        raise ValueError(f"Could not parse patient ID: {filename}")
    return m.group(1)

df["patient_id"] = df["filename"].map(get_pid)

print("Cases:", len(df))
print("Patients:", df["patient_id"].nunique())

# ------------------------------------------------------------
# Case-level sensitivity metrics
# ------------------------------------------------------------
D = df[dice_cols].to_numpy(dtype=float)

pair_idx = list(combinations(range(20), 2))
pair_abs = np.column_stack([
    np.abs(D[:, i] - D[:, j])
    for i, j in pair_idx
])

case_pairwise_abs = pair_abs.mean(axis=1)
case_range = D.max(axis=1) - D.min(axis=1)
case_oracle = D.max(axis=1)

case_metrics = pd.DataFrame({
    "filename": df["filename"],
    "patient_id": df["patient_id"],
    "mean_pairwise_abs_dice": case_pairwise_abs,
    "prompt_range": case_range,
    "prompt_oracle_dice": case_oracle,
})

case_metrics.to_csv(
    OUT / "brain400_p20_case_metrics.csv",
    index=False
)

# ------------------------------------------------------------
# Point estimates
# ------------------------------------------------------------
prompt_means = D.mean(axis=0)
best_idx = int(np.argmax(prompt_means))
best_prompt = prompts[best_idx]
best_fixed = float(prompt_means[best_idx])
oracle_mean = float(case_oracle.mean())
oracle_gap = oracle_mean - best_fixed

print("\n=== PROMPT MEAN DICE ===")
for p, x in zip(prompts, prompt_means):
    print(f"{p}: {x:.9f}")

print("\nBest fixed prompt:", best_prompt)
print(f"Best fixed Dice: {best_fixed:.9f}")
print(f"Prompt oracle Dice: {oracle_mean:.9f}")
print(f"Oracle - best fixed: {oracle_gap:.9f}")

print("\n=== CASE-LEVEL SENSITIVITY ===")
print(f"Mean pairwise |Dice difference|: {case_pairwise_abs.mean():.9f}")
print(f"Median pairwise |Dice difference|: {np.median(case_pairwise_abs):.9f}")
print(f"Mean within-case range: {case_range.mean():.9f}")
print(f"Median within-case range: {np.median(case_range):.9f}")
print(f"Maximum within-case range: {case_range.max():.9f}")

thresholds = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]
print("\n=== RANGE THRESHOLDS ===")
for t in thresholds:
    n = int((case_range > t).sum())
    print(f"range > {t:.2f}: {n}/400 = {n/400:.4%}")

# ------------------------------------------------------------
# Patient-cluster bootstrap
#
# Estimand remains slice-level:
# sample patients with replacement, then concatenate all slices
# belonging to each sampled patient. A patient selected twice has
# all of its slices represented twice.
#
# Best fixed prompt is reselected inside each bootstrap replicate.
# ------------------------------------------------------------
rng = np.random.default_rng(SEED)

patients = df["patient_id"].unique()
patient_rows = {
    pid: np.flatnonzero(df["patient_id"].to_numpy() == pid)
    for pid in patients
}

print("\nBootstrap patients:", len(patients))
assert len(patients) > 1

boot_pairwise = np.empty(N_BOOT)
boot_range = np.empty(N_BOOT)
boot_oracle = np.empty(N_BOOT)
boot_best_fixed = np.empty(N_BOOT)
boot_gap = np.empty(N_BOOT)
boot_thresholds = {t: np.empty(N_BOOT) for t in thresholds}

for b in range(N_BOOT):
    sampled = rng.choice(patients, size=len(patients), replace=True)
    idx = np.concatenate([patient_rows[p] for p in sampled])

    Db = D[idx]
    pairb = case_pairwise_abs[idx]
    rangeb = case_range[idx]

    boot_pairwise[b] = pairb.mean()
    boot_range[b] = rangeb.mean()

    pm = Db.mean(axis=0)
    bf = pm.max()
    om = Db.max(axis=1).mean()

    boot_best_fixed[b] = bf
    boot_oracle[b] = om
    boot_gap[b] = om - bf

    for t in thresholds:
        boot_thresholds[t][b] = np.mean(rangeb > t)

def ci(x):
    return np.percentile(x, [2.5, 97.5])

pair_ci = ci(boot_pairwise)
range_ci = ci(boot_range)
best_ci = ci(boot_best_fixed)
oracle_ci = ci(boot_oracle)
gap_ci = ci(boot_gap)

print("\n=== PATIENT-CLUSTER BOOTSTRAP 95% CI ===")
print(
    f"Mean pairwise abs: {case_pairwise_abs.mean():.9f} "
    f"[{pair_ci[0]:.9f}, {pair_ci[1]:.9f}]"
)
print(
    f"Mean range: {case_range.mean():.9f} "
    f"[{range_ci[0]:.9f}, {range_ci[1]:.9f}]"
)
print(
    f"Best fixed: {best_fixed:.9f} "
    f"[{best_ci[0]:.9f}, {best_ci[1]:.9f}]"
)
print(
    f"Oracle: {oracle_mean:.9f} "
    f"[{oracle_ci[0]:.9f}, {oracle_ci[1]:.9f}]"
)
print(
    f"Oracle gap: {oracle_gap:.9f} "
    f"[{gap_ci[0]:.9f}, {gap_ci[1]:.9f}]"
)

print("\n=== THRESHOLD 95% CI ===")
for t in thresholds:
    point = np.mean(case_range > t)
    lo, hi = ci(boot_thresholds[t])
    print(
        f"range > {t:.2f}: {point:.4%} "
        f"[{lo:.4%}, {hi:.4%}]"
    )

# ------------------------------------------------------------
# Save compact summary
# ------------------------------------------------------------
rows = []

def add(metric, estimate, lo=np.nan, hi=np.nan):
    rows.append({
        "metric": metric,
        "estimate": estimate,
        "ci_low": lo,
        "ci_high": hi
    })

add("n_cases", len(df))
add("n_patients", len(patients))

add(
    "mean_pairwise_abs_dice",
    case_pairwise_abs.mean(),
    *pair_ci
)
add(
    "median_case_pairwise_abs_dice",
    np.median(case_pairwise_abs)
)
add(
    "mean_within_case_range",
    case_range.mean(),
    *range_ci
)
add(
    "median_within_case_range",
    np.median(case_range)
)
add(
    "max_within_case_range",
    case_range.max()
)

for t in thresholds:
    lo, hi = ci(boot_thresholds[t])
    add(
        f"fraction_range_gt_{t:.2f}",
        np.mean(case_range > t),
        lo,
        hi
    )

for p, x in zip(prompts, prompt_means):
    add(f"{p}_mean_dice", x)

add("best_fixed_prompt_index", best_idx + 1)
add("best_fixed_dice", best_fixed, *best_ci)
add("prompt_oracle_dice", oracle_mean, *oracle_ci)
add("oracle_minus_best_fixed", oracle_gap, *gap_ci)

summary = pd.DataFrame(rows)
summary.to_csv(
    OUT / "brain400_p20_summary.csv",
    index=False
)

print("\nSaved:")
print(OUT / "brain400_p20_case_metrics.csv")
print(OUT / "brain400_p20_summary.csv")
print("\nANALYSIS COMPLETE")
