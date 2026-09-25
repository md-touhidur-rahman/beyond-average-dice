from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path("/home/hpc/rlvl/rlvl178v/prl_medclipsam")

BASE = (
    ROOT /
    "beyond-average-dice/remedy/results/embedding_trajectory"
)

SRC = BASE / "trajectory_mechanism.csv"

OUT_DOSE = BASE / "dose_response_by_alpha.csv"
OUT_CASE = BASE / "dose_response_case_by_alpha.csv"
OUT_TXT  = BASE / "dose_response_audit.txt"

# ------------------------------------------------------------
# Frozen-input gate
# ------------------------------------------------------------
EXPECTED_SRC_SHA256 = (
    "41ab34e7374dc7cea80c8b24e5add76e"
    "602622687736eba6c21ccd947eb70c4e"
)

got = hashlib.sha256(SRC.read_bytes()).hexdigest()

assert got == EXPECTED_SRC_SHA256, (
    "Refusing analysis: trajectory_mechanism.csv hash changed.\n"
    f"expected={EXPECTED_SRC_SHA256}\n"
    f"got={got}"
)

df = pd.read_csv(SRC)

required = [
    "filename",
    "risk_region",
    "frozen_risk",
    "pair_id",
    "seed",
    "alpha",
    "corr_distance_from_A",
    "corr_distance_from_B",
    "heatmap_mae_from_A",
    "heatmap_mae_from_B",
]

missing = [c for c in required if c not in df.columns]
assert not missing, missing

assert len(df) == 6000
assert df["filename"].nunique() == 12
assert df["pair_id"].nunique() == 20
assert df["seed"].nunique() == 5

alphas = sorted(df["alpha"].unique().tolist())

assert np.allclose(
    alphas,
    [0.0, 0.25, 0.5, 0.75, 1.0]
)

assert not df[required].isnull().any().any()

for c in [
    "corr_distance_from_A",
    "corr_distance_from_B",
    "heatmap_mae_from_A",
    "heatmap_mae_from_B",
]:
    assert np.isfinite(df[c]).all(), c

# ------------------------------------------------------------
# Raw descriptive dose summaries
#
# IMPORTANT:
# n=1200 at each dose is descriptive only.
# These observations are NOT independent inferential replicates.
# ------------------------------------------------------------
metrics = [
    "corr_distance_from_A",
    "corr_distance_from_B",
    "heatmap_mae_from_A",
    "heatmap_mae_from_B",
]

rows = []

for alpha, g in df.groupby("alpha", sort=True):

    assert len(g) == 1200

    row = {
        "alpha": float(alpha),
        "n_raw_observations": len(g),
    }

    for metric in metrics:
        x = g[metric].to_numpy(dtype=float)

        row[f"{metric}_mean"] = np.mean(x)
        row[f"{metric}_std"] = np.std(x, ddof=1)
        row[f"{metric}_median"] = np.median(x)
        row[f"{metric}_q25"] = np.quantile(x, 0.25)
        row[f"{metric}_q75"] = np.quantile(x, 0.75)
        row[f"{metric}_min"] = np.min(x)
        row[f"{metric}_max"] = np.max(x)

    rows.append(row)

dose = pd.DataFrame(rows)
dose.to_csv(OUT_DOSE, index=False)

# ------------------------------------------------------------
# Case-level dose summaries
#
# First average all prompt-pair x seed observations within each
# case and alpha. This produces 12 case-level values at each dose.
# ------------------------------------------------------------
case = (
    df.groupby(
        [
            "filename",
            "risk_region",
            "frozen_risk",
            "alpha",
        ],
        as_index=False,
    )[metrics]
    .mean()
)

assert len(case) == 12 * 5

for alpha, g in case.groupby("alpha"):
    assert len(g) == 12

case.to_csv(OUT_CASE, index=False)

# ------------------------------------------------------------
# Case-level directional inference
#
# Use the already-frozen primary trajectory analysis.
# Do NOT infer from 1200 raw trajectories.
# ------------------------------------------------------------
PRIMARY_CASE = BASE / "trajectory_analysis_12.csv"

EXPECTED_PRIMARY_CASE_SHA256 = (
    "ab4b0409004feabb29515f6b06ec203a"
    "497c229a9ab633996193fb8df2b959e8"
)

got_case = hashlib.sha256(
    PRIMARY_CASE.read_bytes()
).hexdigest()

assert got_case == EXPECTED_PRIMARY_CASE_SHA256, (
    "Refusing inference: frozen case analysis hash changed.\n"
    f"expected={EXPECTED_PRIMARY_CASE_SHA256}\n"
    f"got={got_case}"
)

pc = pd.read_csv(PRIMARY_CASE)

assert len(pc) == 12
assert "corr_A_spearman_mean" in pc.columns

vals = pc["corr_A_spearman_mean"].to_numpy(dtype=float)

n_pos = int(np.sum(vals > 0))
n_zero = int(np.sum(vals == 0))
n_neg = int(np.sum(vals < 0))

# Exact two-sided sign test.
# There are no zero-valued cases in the observed frozen result,
# but handle zeros explicitly rather than silently treating them
# as positive/negative.
nonzero = vals[vals != 0]

sign_result = binomtest(
    int(np.sum(nonzero > 0)),
    n=len(nonzero),
    p=0.5,
    alternative="two-sided",
)

# ------------------------------------------------------------
# Step increments: useful for response-shape inspection
#
# These are descriptive quantities, not newly declared endpoints.
# ------------------------------------------------------------
case_wide_A = case.pivot(
    index="filename",
    columns="alpha",
    values="corr_distance_from_A",
)

case_wide_MAE = case.pivot(
    index="filename",
    columns="alpha",
    values="heatmap_mae_from_A",
)

increments_A = case_wide_A.diff(axis=1).iloc[:, 1:]
increments_MAE = case_wide_MAE.diff(axis=1).iloc[:, 1:]

# ------------------------------------------------------------
# Human-readable audit
# ------------------------------------------------------------
lines = []

lines.append(
    "=== EMBEDDING INTERPOLATION DOSE-RESPONSE AUDIT ==="
)
lines.append("")
lines.append(
    "POST-HOC DESCRIPTIVE SHAPE AUDIT OF THE "
    "PRE-SPECIFIED MECHANISM RESULT"
)
lines.append("")
lines.append(
    "This audit does NOT redefine the frozen primary endpoint."
)
lines.append(
    "Its purpose is to expose the raw response magnitude at each "
    "of the five pre-specified interpolation doses."
)
lines.append("")

lines.append("=== RAW OBSERVATIONS BY DOSE ===")
lines.append(
    "1200 observations per dose; descriptive only, "
    "not independent inferential replicates."
)
lines.append("")

for _, r in dose.iterrows():
    lines.append(f"alpha = {r['alpha']:.2f}")

    lines.append(
        "  corr distance from A: "
        f"mean={r['corr_distance_from_A_mean']:.9f} "
        f"median={r['corr_distance_from_A_median']:.9f} "
        f"IQR=[{r['corr_distance_from_A_q25']:.9f}, "
        f"{r['corr_distance_from_A_q75']:.9f}]"
    )

    lines.append(
        "  corr distance from B: "
        f"mean={r['corr_distance_from_B_mean']:.9f} "
        f"median={r['corr_distance_from_B_median']:.9f} "
        f"IQR=[{r['corr_distance_from_B_q25']:.9f}, "
        f"{r['corr_distance_from_B_q75']:.9f}]"
    )

    lines.append(
        "  heatmap MAE from A:   "
        f"mean={r['heatmap_mae_from_A_mean']:.9f} "
        f"median={r['heatmap_mae_from_A_median']:.9f} "
        f"IQR=[{r['heatmap_mae_from_A_q25']:.9f}, "
        f"{r['heatmap_mae_from_A_q75']:.9f}]"
    )

    lines.append(
        "  heatmap MAE from B:   "
        f"mean={r['heatmap_mae_from_B_mean']:.9f} "
        f"median={r['heatmap_mae_from_B_median']:.9f} "
        f"IQR=[{r['heatmap_mae_from_B_q25']:.9f}, "
        f"{r['heatmap_mae_from_B_q75']:.9f}]"
    )

    lines.append("")

lines.append("=== CASE-LEVEL MEANS BY DOSE ===")
lines.append(
    "Each value below first averages the 20 prompt pairs x "
    "5 matched seeds within a case."
)
lines.append("")

case_dose = (
    case.groupby("alpha")[metrics]
    .agg(["mean", "median", "min", "max"])
)

lines.append(case_dose.to_string())
lines.append("")

lines.append(
    "=== CASE-LEVEL STEP INCREMENTS: CORR DISTANCE FROM A ==="
)
lines.append(
    "Mean change across the 12 case-level dose curves."
)

for alpha in increments_A.columns:
    x = increments_A[alpha].to_numpy()
    lines.append(
        f"to alpha={alpha:.2f}: "
        f"mean increment={np.mean(x):.9f}, "
        f"median={np.median(x):.9f}, "
        f"positive cases={np.sum(x > 0)}/12"
    )

lines.append("")

lines.append(
    "=== CASE-LEVEL STEP INCREMENTS: HEATMAP MAE FROM A ==="
)

for alpha in increments_MAE.columns:
    x = increments_MAE[alpha].to_numpy()
    lines.append(
        f"to alpha={alpha:.2f}: "
        f"mean increment={np.mean(x):.9f}, "
        f"median={np.median(x):.9f}, "
        f"positive cases={np.sum(x > 0)}/12"
    )

lines.append("")

lines.append("=== PRIMARY CASE-LEVEL DIRECTIONAL TEST ===")
lines.append(
    f"positive case means: {n_pos} / 12"
)
lines.append(
    f"zero case means:     {n_zero} / 12"
)
lines.append(
    f"negative case means: {n_neg} / 12"
)
lines.append(
    "exact two-sided sign-test p: "
    f"{sign_result.pvalue:.12g}"
)
lines.append("")

lines.append("=== INTERPRETIVE RESTRICTIONS ===")
lines.append(
    "The five interpolation doses provide five ordered observations "
    "per trajectory."
)
lines.append(
    "Accordingly, perfect Spearman ordering is a coarse five-point "
    "ordering criterion and is not interpreted as high-resolution "
    "evidence of linearity."
)
lines.append(
    "Response shape is assessed from the raw dose magnitudes and "
    "case-level dose curves, not from Spearman rho alone."
)
lines.append(
    "Endpoint progress is mathematically coupled to distances from "
    "A and B and is treated as a manipulation-validity / trajectory "
    "sanity check rather than independent mechanistic evidence."
)
lines.append("")

lines.append("DEVELOPMENT DIAGNOSTIC ONLY: YES")
lines.append("GROUND TRUTH USED: NO")
lines.append("DICE OUTCOMES USED: NO")
lines.append("HELDOUT OUTCOMES ACCESSED: NO")
lines.append("PRIMARY ENDPOINT REDEFINED: NO")
lines.append("FROZEN ANALYSIS PLAN MODIFIED: NO")
lines.append("1200 TRAJECTORIES USED AS INDEPENDENT INFERENCE: NO")

text = "\n".join(lines) + "\n"

OUT_TXT.write_text(text)

print(text)

print("saved:", OUT_DOSE)
print("saved:", OUT_CASE)
print("saved:", OUT_TXT)
