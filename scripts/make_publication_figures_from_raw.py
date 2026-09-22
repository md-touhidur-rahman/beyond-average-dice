from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/home/hpc/rlvl/rlvl178v/prl_medclipsam")
OUT = ROOT / "beyond-average-dice/paper_results/publication_from_raw"
OUT.mkdir(parents=True, exist_ok=True)

B600 = ROOT / "beyond-average-dice/results/brain600_language_ladder/case_metrics.csv"
B400 = ROOT / "beyond-average-dice/results/brain_p20/failure_success_pair_decomposition.csv"

SEED = 42
N_BOOT = 10_000

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def percentile_ci(x, statistic=np.mean, n_boot=N_BOOT, seed=SEED):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)

    for b in range(n_boot):
        sample = x[rng.integers(0, len(x), len(x))]
        vals[b] = statistic(sample)

    return np.percentile(vals, [2.5, 97.5])


def cluster_bootstrap(df, cluster_col, value_col,
                      statistic=np.mean,
                      n_boot=N_BOOT, seed=SEED):
    """
    Patient-cluster bootstrap preserving all pair observations
    belonging to sampled patients.
    """
    rng = np.random.default_rng(seed)
    clusters = df[cluster_col].dropna().unique()

    vals = np.empty(n_boot)

    groups = {
        c: df.loc[df[cluster_col] == c, value_col].to_numpy()
        for c in clusters
    }

    for b in range(n_boot):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        x = np.concatenate([groups[c] for c in sampled])
        vals[b] = statistic(x)

    return np.percentile(vals, [2.5, 97.5])


def savefig(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 1. BRAIN600 — LOAD ACTUAL CASE-LEVEL DATA
# ============================================================

b600 = pd.read_csv(B600)

required = (
    ["case"] +
    [f"dice_{x}" for x in ["H0","H1","H2","L3","L4","L5"]] +
    [f"box_iou_{x}" for x in ["H1","H2","L3","L4","L5"]]
)

missing = sorted(set(required) - set(b600.columns))
assert not missing, f"Brain600 missing columns: {missing}"
assert len(b600) == 600
assert b600["case"].nunique() == 600

conditions = ["H1", "H2", "L3", "L4", "L5"]

rows = []

for cond in conditions:
    signed = b600[f"dice_{cond}"] - b600["dice_H0"]
    absolute = signed.abs()

    lo, hi = percentile_ci(absolute)

    rows.append({
        "condition": cond,
        "n": len(b600),
        "mean_abs_delta": absolute.mean(),
        "median_abs_delta": absolute.median(),
        "ci_low": lo,
        "ci_high": hi,
        "p_abs_gt_010": (absolute > .10).mean(),
        "p_abs_gt_020": (absolute > .20).mean(),
        "p_abs_gt_050": (absolute > .50).mean(),
        "mean_signed_delta": signed.mean(),
        "mean_box_iou": b600[f"box_iou_{cond}"].mean()
    })

b600_summary = pd.DataFrame(rows)
b600_summary.to_csv(OUT / "figure1_brain600_recomputed_data.csv", index=False)

print("\nBRAIN600 RECOMPUTED")
print(b600_summary.to_string(index=False))


# Sanity checks derived from raw cases.
expected = {
    "H1": 0.020296,
    "H2": 0.029997,
    "L3": 0.033596,
    "L4": 0.080111,
    "L5": 0.072436,
}

for cond, target in expected.items():
    got = b600_summary.loc[
        b600_summary.condition == cond, "mean_abs_delta"
    ].iloc[0]

    assert np.isclose(got, target, atol=1e-6), (
        f"{cond}: recomputed {got:.9f}, expected {target:.9f}"
    )


# ------------------------------------------------------------
# Figure 1: controlled reformulation / information ablation
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7.2, 4.8))

x = np.arange(len(conditions))

means = b600_summary["mean_abs_delta"].to_numpy()
low = b600_summary["ci_low"].to_numpy()
high = b600_summary["ci_high"].to_numpy()

ax.errorbar(
    x, means,
    yerr=np.vstack([means-low, high-means]),
    fmt="o",
    capsize=4,
    linewidth=1.5
)

ax.set_xticks(x)
ax.set_xticklabels(["H1", "H2", "L3", "L4", "L5"])
ax.set_ylabel(r"Mean $|\Delta Dice|$ from H0")
ax.set_xlabel("Language condition")
ax.set_title("Controlled language reformulation and information ablation")
ax.grid(axis="y", alpha=.25)

fig.tight_layout()
savefig(fig, "01_brain600_controlled_language")


# ============================================================
# 2. BRAIN600 — SIX-CONDITION CASE RANGE
# ============================================================

dice_cols = [f"dice_{x}" for x in ["H0","H1","H2","L3","L4","L5"]]

D = b600[dice_cols].to_numpy()

case_range = D.max(axis=1) - D.min(axis=1)
oracle = D.max(axis=1)

condition_means = b600[dice_cols].mean()
best_fixed_col = condition_means.idxmax()
best_fixed = b600[best_fixed_col].to_numpy()

oracle_advantage = oracle - best_fixed

range_lo, range_hi = percentile_ci(case_range)
oracle_lo, oracle_hi = percentile_ci(oracle)
adv_lo, adv_hi = percentile_ci(oracle_advantage)

range_summary = pd.DataFrame([{
    "n": len(b600),
    "mean_range": case_range.mean(),
    "median_range": np.median(case_range),
    "max_range": case_range.max(),
    "range_gt_010": (case_range > .10).mean(),
    "range_gt_020": (case_range > .20).mean(),
    "range_gt_030": (case_range > .30).mean(),
    "range_gt_050": (case_range > .50).mean(),
    "best_fixed_condition": best_fixed_col.replace("dice_", ""),
    "best_fixed_mean": best_fixed.mean(),
    "oracle_mean": oracle.mean(),
    "oracle_advantage": oracle_advantage.mean(),
    "mean_range_ci_low": range_lo,
    "mean_range_ci_high": range_hi,
    "oracle_ci_low": oracle_lo,
    "oracle_ci_high": oracle_hi,
    "oracle_advantage_ci_low": adv_lo,
    "oracle_advantage_ci_high": adv_hi
}])

range_summary.to_csv(
    OUT / "figure2_brain600_case_range_recomputed_data.csv",
    index=False
)

assert np.isclose(case_range.mean(), 0.132872, atol=1e-6)
assert np.isclose(np.median(case_range), 0.036751, atol=1e-6)
assert np.isclose((case_range > .50).mean(), 0.08, atol=1e-12)

print("\nBRAIN600 SIX-CONDITION RANGE")
print(range_summary.to_string(index=False))


# empirical survival curve
xs = np.sort(case_range)
survival = 1.0 - np.arange(1, len(xs)+1) / len(xs)

fig, ax = plt.subplots(figsize=(7.2, 4.8))
ax.step(xs, survival, where="post")

for threshold in [.10, .20, .50]:
    ax.axvline(threshold, linestyle="--", linewidth=1, alpha=.45)

ax.set_xlabel("Within-case Dice range across H0–L5")
ax.set_ylabel("Fraction of cases exceeding range")
ax.set_title("Heavy-tailed case-level sensitivity in Brain600")
ax.set_xlim(left=0)
ax.set_ylim(0, 1)

fig.tight_layout()
savefig(fig, "02_brain600_case_range_survival")


# ============================================================
# 3. BRAIN400 — LOAD ACTUAL 5,398 PAIR RECORDS
# ============================================================

b400 = pd.read_csv(B400)

required = [
    "filename",
    "box_iou",
    "selected_diff",
    "oracle_diff",
    "different_selected_idx",
    "failure_has_oracle_50",
    "failure_selection_loss_20",
    "patient_id",
    "stratum",
]

missing = sorted(set(required) - set(b400.columns))
assert not missing, f"Brain400 missing columns: {missing}"
assert len(b400) == 5398

# Confirm these really are failure-success pairs.
fail_success = (
    ((b400.dice_a < .10) & (b400.dice_b >= .50)) |
    ((b400.dice_b < .10) & (b400.dice_a >= .50))
)
assert fail_success.all()


# ============================================================
# 4. LOCALIZATION DISTRIBUTION — RECOMPUTE FROM PAIRS
# ============================================================

loc = pd.DataFrame([
    {"criterion": "IoU < 0.10", "N": (b400.box_iou < .10).sum(),
     "rate": (b400.box_iou < .10).mean()},

    {"criterion": "IoU < 0.25", "N": (b400.box_iou < .25).sum(),
     "rate": (b400.box_iou < .25).mean()},

    {"criterion": "IoU < 0.50", "N": (b400.box_iou < .50).sum(),
     "rate": (b400.box_iou < .50).mean()},

    {"criterion": "IoU >= 0.75", "N": (b400.box_iou >= .75).sum(),
     "rate": (b400.box_iou >= .75).mean()},

    {"criterion": "IoU >= 0.90", "N": (b400.box_iou >= .90).sum(),
     "rate": (b400.box_iou >= .90).mean()},
])

loc.to_csv(
    OUT / "figure3_failure_success_localization_recomputed_data.csv",
    index=False
)

assert loc.iloc[0]["N"] == 4520
assert loc.iloc[1]["N"] == 4773
assert loc.iloc[2]["N"] == 5088
assert loc.iloc[4]["N"] == 25

print("\nBRAIN400 FAILURE-SUCCESS LOCALIZATION")
print(loc.to_string(index=False))


fig, ax = plt.subplots(figsize=(7.2, 4.8))

plot_loc = loc.iloc[:3]

ax.bar(
    plot_loc["criterion"],
    plot_loc["rate"] * 100
)

ax.set_ylabel("Failure-success pairs (%)")
ax.set_title("Most catastrophic transitions coincide with localization switching")
ax.set_ylim(0, 100)

for i, row in plot_loc.reset_index(drop=True).iterrows():
    ax.text(
        i,
        row["rate"] * 100 + 2,
        f'{row["rate"]*100:.1f}%',
        ha="center"
    )

fig.tight_layout()
savefig(fig, "03_failure_success_localization")


# ============================================================
# 5. COMPONENT DECOMPOSITION BY BOX-IoU STRATUM
# ============================================================

order = [
    "<0.10",
    "0.10–0.25",
    "0.25–0.50",
    "0.50–0.75",
    "0.75–0.90",
    "≥0.90",
]

rows = []

for stratum in order:
    d = b400[b400["stratum"] == stratum].copy()

    assert len(d) > 0, f"No rows for {stratum}"

    sel_ci = cluster_bootstrap(
        d, "patient_id", "selected_diff"
    )

    ora_ci = cluster_bootstrap(
        d, "patient_id", "oracle_diff"
    )

    rows.append({
        "stratum": stratum,
        "N_pairs": len(d),
        "N_images": d.filename.nunique(),
        "N_patients": d.patient_id.nunique(),
        "mean_box_iou": d.box_iou.mean(),

        "selected_diff_mean": d.selected_diff.mean(),
        "selected_diff_ci_low": sel_ci[0],
        "selected_diff_ci_high": sel_ci[1],

        "oracle_diff_mean": d.oracle_diff.mean(),
        "oracle_diff_ci_low": ora_ci[0],
        "oracle_diff_ci_high": ora_ci[1],

        "different_selected_idx_rate":
            d.different_selected_idx.astype(float).mean(),

        "fail_oracle_ge50_rate":
            d.failure_has_oracle_50.astype(float).mean(),

        "fail_selection_gap_gt20_rate":
            d.failure_selection_loss_20.astype(float).mean(),
    })

decomp = pd.DataFrame(rows)

decomp.to_csv(
    OUT / "figure4_component_decomposition_recomputed_data.csv",
    index=False
)

print("\nBRAIN400 COMPONENT DECOMPOSITION")
print(decomp.to_string(index=False))


# Strong sanity checks from the actual pair records
r0 = decomp.loc[decomp.stratum == "<0.10"].iloc[0]
r9 = decomp.loc[decomp.stratum == "≥0.90"].iloc[0]

assert int(r0.N_pairs) == 4520
assert int(r9.N_pairs) == 25
assert int(r9.N_images) == 6
assert int(r9.N_patients) == 6

assert np.isclose(
    r0.fail_oracle_ge50_rate,
    0.00022123893805309734,
    atol=1e-12
)

assert np.isclose(
    r9.fail_oracle_ge50_rate,
    0.88,
    atol=1e-12
)

assert np.isclose(
    r9.different_selected_idx_rate,
    0.96,
    atol=1e-12
)

assert np.isclose(
    r9.fail_selection_gap_gt20_rate,
    1.0,
    atol=1e-12
)


# ------------------------------------------------------------
# Figure 4: selected vs candidate-oracle Dice divergence
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8.2, 5.0))

x = np.arange(len(decomp))
width = .36

ax.bar(
    x - width/2,
    decomp.selected_diff_mean,
    width,
    label="Selected SAM mask"
)

ax.bar(
    x + width/2,
    decomp.oracle_diff_mean,
    width,
    label="Retrospective candidate oracle"
)

ax.set_xticks(x)
ax.set_xticklabels(order, rotation=20)
ax.set_xlabel("Bounding-box IoU between failure and success prompts")
ax.set_ylabel(r"Mean $|\Delta Dice|$")
ax.set_title("Failure mechanism changes with localization agreement")
ax.legend(frameon=False)

fig.tight_layout()
savefig(fig, "04_failure_component_decomposition")


# ============================================================
# 6. HIGH-OVERLAP RESIDUAL — COMPUTED, NOT ENTERED MANUALLY
# ============================================================

high = b400[b400.box_iou >= .90].copy()

high_summary = pd.DataFrame([{
    "N_pairs": len(high),
    "N_images": high.filename.nunique(),
    "N_patients": high.patient_id.nunique(),
    "mean_selected_diff": high.selected_diff.mean(),
    "mean_oracle_diff": high.oracle_diff.mean(),
    "oracle_ge50_rate": high.failure_has_oracle_50.mean(),
    "different_selected_idx_rate": high.different_selected_idx.mean(),
    "selection_gap_gt20_rate": high.failure_selection_loss_20.mean()
}])

high_summary.to_csv(
    OUT / "high_overlap_residual_recomputed_data.csv",
    index=False
)

print("\nHIGH-OVERLAP RESIDUAL")
print(high_summary.to_string(index=False))


# ============================================================
# FINAL MANIFEST
# ============================================================

manifest = pd.DataFrame([
    {
        "figure": "01",
        "source": str(B600),
        "unit": "image",
        "N": 600,
        "description":
            "H0-relative controlled language sensitivity"
    },
    {
        "figure": "02",
        "source": str(B600),
        "unit": "image",
        "N": 600,
        "description":
            "Six-condition within-case range"
    },
    {
        "figure": "03",
        "source": str(B400),
        "unit": "failure-success prompt pair",
        "N": 5398,
        "description":
            "Localization distribution of catastrophic transitions"
    },
    {
        "figure": "04",
        "source": str(B400),
        "unit": "failure-success prompt pair; patient-cluster bootstrap",
        "N": 5398,
        "description":
            "Localization/candidate component decomposition"
    }
])

manifest.to_csv(OUT / "FIGURE_PROVENANCE.csv", index=False)

print("\n" + "="*70)
print("ALL ASSERTIONS PASSED")
print("Publication outputs:", OUT)
print("="*70)
