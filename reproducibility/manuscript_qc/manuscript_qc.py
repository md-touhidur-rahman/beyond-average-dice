from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path("/home/hpc/rlvl/rlvl178v/prl_medclipsam")
B = 10000
SEED = 42
TOL = 5e-10

rows = []

def check(section, metric, got, expected, tol=TOL):
    if isinstance(got, str):
        ok = got == expected
        diff = ""
    else:
        diff = abs(float(got) - float(expected))
        ok = diff <= tol
    rows.append([section, metric, got, expected, diff, "PASS" if ok else "FAIL"])
    print(f"{'PASS' if ok else 'FAIL':4s} | {section:18s} | {metric:38s} | {got}")

def pid(x):
    m = re.search(r"pid-([^_\.]+)", str(x))
    if not m:
        raise ValueError(f"Cannot extract patient ID: {x}")
    return m.group(1)

def ci(x):
    return np.percentile(np.asarray(x, float), [2.5, 97.5])

# ============================================================
# BRAIN400: independently reconstruct from 400 x 20 Dice matrix
# ============================================================

P = ROOT / "beyond-average-dice/results/brain_p20"
df = pd.read_csv(P / "brain400_p20.csv")

prompts = [f"P{i:02d}" for i in range(1, 21)]
D = df[[f"{p}_dice" for p in prompts]].to_numpy(float)

n, k = D.shape
tri = np.triu_indices(k, 1)

case_pair = np.abs(
    D[:, :, None] - D[:, None, :]
)[:, tri[0], tri[1]].mean(axis=1)

case_range = D.max(axis=1) - D.min(axis=1)
oracle = D.max(axis=1)
means = D.mean(axis=0)

best_idx = int(np.argmax(means))
best = means[best_idx]
adv = oracle.mean() - best

s = pd.read_csv(P / "brain400_p20_summary.csv").set_index("metric")

check("Brain400", "N cases", n, 400, 0)
check("Brain400", "mean pairwise |dDice|",
      case_pair.mean(),
      float(s.loc["mean_pairwise_abs_dice", "estimate"]))

check("Brain400", "mean within-case range",
      case_range.mean(),
      float(s.loc["mean_within_case_range", "estimate"]))

check("Brain400", "median within-case range",
      np.median(case_range),
      float(s.loc["median_within_case_range", "estimate"]))

check("Brain400", "max within-case range",
      case_range.max(),
      float(s.loc["max_within_case_range", "estimate"]))

check("Brain400", "best fixed prompt",
      prompts[best_idx], "P14")

check("Brain400", "best fixed Dice",
      best,
      float(s.loc["best_fixed_dice", "estimate"]))

check("Brain400", "oracle Dice",
      oracle.mean(),
      float(s.loc["prompt_oracle_dice", "estimate"]))

check("Brain400", "oracle advantage",
      adv,
      float(s.loc["oracle_minus_best_fixed", "estimate"]))

# ============================================================
# BRAIN400 bootstrap: patient cluster, 10k, seed 42
# ============================================================

pids = np.array([pid(x) for x in df.filename])
patients = pd.unique(pids)
patient_rows = {p: np.flatnonzero(pids == p) for p in patients}

check("Brain400", "unique patients", len(patients), 174, 0)

rng = np.random.default_rng(SEED)

bp = np.empty(B)
br = np.empty(B)
bb = np.empty(B)
bo = np.empty(B)
ba = np.empty(B)

for b in range(B):
    sampled = rng.choice(patients, size=len(patients), replace=True)
    idx = np.concatenate([patient_rows[p] for p in sampled])
    Z = D[idx]

    bp[b] = case_pair[idx].mean()
    br[b] = case_range[idx].mean()
    bb[b] = Z.mean(axis=0).max()
    bo[b] = Z.max(axis=1).mean()
    ba[b] = bo[b] - bb[b]

for name, arr, metric in [
    ("pairwise CI", bp, "mean_pairwise_abs_dice"),
    ("range CI", br, "mean_within_case_range"),
    ("best fixed CI", bb, "best_fixed_dice"),
    ("oracle CI", bo, "prompt_oracle_dice"),
    ("oracle advantage CI", ba, "oracle_minus_best_fixed"),
]:
    lo, hi = ci(arr)
    check("Brain400", name + " low",
          lo, float(s.loc[metric, "ci_low"]), 2e-9)
    check("Brain400", name + " high",
          hi, float(s.loc[metric, "ci_high"]), 2e-9)

# ============================================================
# BRAIN400: full pair universe + two DIFFERENT residual analyses
# ============================================================

pw = pd.read_csv(P / "brain400_p20_pairwise_localization.csv")

check("B400 funnel", "all pairs", len(pw), 76000, 0)
check("B400 funnel", "all images", pw.filename.nunique(), 400, 0)
check("B400 funnel", "all patients",
      pw.filename.map(pid).nunique(), 174, 0)

general = pw[
    (pw.box_iou >= 0.90) &
    (pw.abs_dice_diff > 0.20)
].copy()

check("B400 funnel", "general residual pairs",
      len(general), 224, 0)
check("B400 funnel", "general residual images",
      general.filename.nunique(), 31, 0)
check("B400 funnel", "general residual patients",
      general.filename.map(pid).nunique(), 27, 0)

cat = pw[
    ((pw.dice1 < .10) & (pw.dice2 >= .50)) |
    ((pw.dice2 < .10) & (pw.dice1 >= .50))
].copy()

check("B400 funnel", "catastrophic pairs from 76k",
      len(cat), 5398, 0)

fs = pd.read_csv(P / "failure_success_pair_decomposition.csv")

check("B400 funnel", "stored catastrophic pairs",
      len(fs), 5398, 0)
check("B400 funnel", "catastrophic images",
      fs.filename.nunique(), 132, 0)
check("B400 funnel", "catastrophic patients",
      fs.patient_id.nunique(), 88, 0)

check("B400 funnel", "fraction box IoU < .10",
      np.mean(fs.box_iou < .10), 4520/5398)
check("B400 funnel", "fraction box IoU < .25",
      np.mean(fs.box_iou < .25), 4773/5398)
check("B400 funnel", "fraction box IoU < .50",
      np.mean(fs.box_iou < .50), 5088/5398)

high = fs[fs.box_iou >= .90]

check("B400 funnel", "catastrophic high-IoU pairs",
      len(high), 25, 0)
check("B400 funnel", "catastrophic high-IoU images",
      high.filename.nunique(), 6, 0)
check("B400 funnel", "catastrophic high-IoU patients",
      high.patient_id.nunique(), 6, 0)

# ============================================================
# GENERAL 224 residual candidate decomposition
# ============================================================

rd = pd.read_csv(P / "residual_pair_decomposition.csv")

check("Residual224", "N pairs", len(rd), 224, 0)
check("Residual224", "mean selected |dDice|",
      rd.selected_diff.mean(), 0.3609, 5e-5)
check("Residual224", "mean oracle |dDice|",
      rd.oracle_diff.mean(), 0.0732, 5e-5)
check("Residual224", "different selected candidate",
      rd.different_selected_idx.mean(), 0.875)
check("Residual224", "any selection loss > .20",
      rd.any_selection_loss_20.mean(),
      0.9241071428571429)
check("Residual224", "selection dominant",
      rd.selection_dominant.mean(),
      0.9196428571428571)

# ============================================================
# CATASTROPHIC 25 high-overlap decomposition
# ============================================================

check("CatHigh25", "mean selected |dDice|",
      high.selected_diff.mean(), 0.679701, 5e-7)
check("CatHigh25", "mean oracle |dDice|",
      high.oracle_diff.mean(), 0.130282, 5e-7)
check("CatHigh25", "different selected candidate",
      high.different_selected_idx.mean(), .96)
check("CatHigh25", "failure oracle candidate >= .50",
      high.failure_has_oracle_50.mean(), .88)
check("CatHigh25", "failure selection gap > .20",
      high.failure_selection_loss_20.mean(), 1.0)

# ============================================================
# BREAST97
# ============================================================

P = ROOT / "beyond-average-dice/results/breast_p20"
df = pd.read_csv(P / "breast97_p20.csv")

X = df[[f"{p}_dice" for p in prompts]].to_numpy(float)

pair = np.abs(
    X[:, :, None] - X[:, None, :]
)[:, tri[0], tri[1]].mean(axis=1)

ran = X.max(axis=1) - X.min(axis=1)
ora = X.max(axis=1)
means = X.mean(axis=0)
bi = int(np.argmax(means))
bf = means[bi]
og = ora.mean() - bf

sb = pd.read_csv(P / "breast97_p20_summary.csv").set_index("metric")

check("Breast97", "N", len(df), 97, 0)
check("Breast97", "mean pairwise |dDice|",
      pair.mean(),
      float(sb.loc["mean_pairwise_abs", "point"]))
check("Breast97", "mean range",
      ran.mean(),
      float(sb.loc["mean_range", "point"]))
check("Breast97", "best fixed prompt",
      prompts[bi], "P13")
check("Breast97", "best fixed Dice",
      bf,
      float(sb.loc["best_fixed_mean", "point"]))
check("Breast97", "oracle Dice",
      ora.mean(),
      float(sb.loc["oracle_mean", "point"]))
check("Breast97", "oracle advantage",
      og,
      float(sb.loc["oracle_over_best_fixed", "point"]))

# Breast bootstrap
rng = np.random.default_rng(SEED)

bpair = np.empty(B)
brange = np.empty(B)
bog = np.empty(B)

for b in range(B):
    idx = rng.integers(0, len(X), len(X))
    Z = X[idx]

    bpair[b] = pair[idx].mean()
    brange[b] = ran[idx].mean()
    bog[b] = Z.max(axis=1).mean() - Z.mean(axis=0).max()

for name, arr, metric in [
    ("pairwise CI", bpair, "mean_pairwise_abs"),
    ("range CI", brange, "mean_range"),
    ("oracle advantage CI", bog, "oracle_over_best_fixed"),
]:
    lo, hi = ci(arr)
    check("Breast97", name + " low",
          lo, float(sb.loc[metric, "ci_low"]), 2e-9)
    check("Breast97", name + " high",
          hi, float(sb.loc[metric, "ci_high"]), 2e-9)

# ============================================================
# BRAIN600
# ============================================================

P = ROOT / "beyond-average-dice/results/brain600_language_ladder"

df = pd.read_csv(P / "case_metrics.csv")

conds = ["H0", "H1", "H2", "L3", "L4", "L5"]

M = df[[f"dice_{c}" for c in conds]].to_numpy(float)

ran = M.max(axis=1) - M.min(axis=1)
ora = M.max(axis=1)
means = M.mean(axis=0)

bi = int(np.argmax(means))
bf = means[bi]
adv = ora.mean() - bf

ss = pd.read_csv(P / "six_condition_summary.csv").iloc[0]

check("Brain600", "N", len(df), 600, 0)
check("Brain600", "mean range",
      ran.mean(), float(ss.mean_range))
check("Brain600", "median range",
      np.median(ran), float(ss.median_range))
check("Brain600", "max range",
      ran.max(), float(ss.max_range))
check("Brain600", "best fixed condition",
      conds[bi], str(ss.best_fixed_condition))
check("Brain600", "best fixed Dice",
      bf, float(ss.best_fixed_mean_dice))
check("Brain600", "oracle Dice",
      ora.mean(), float(ss.oracle_mean_dice))
check("Brain600", "oracle advantage",
      adv, float(ss.oracle_advantage))

for t, col in [
    (.01, "range_gt_001"),
    (.05, "range_gt_005"),
    (.10, "range_gt_010"),
    (.20, "range_gt_020"),
    (.30, "range_gt_030"),
    (.50, "range_gt_050"),
]:
    check("Brain600", f"fraction range > {t}",
          np.mean(ran > t), float(ss[col]))

hr = pd.read_csv(P / "h0_relative_summary.csv").set_index("condition")

for j, c in enumerate(conds[1:], 1):
    delta = M[:, j] - M[:, 0]
    a = np.abs(delta)

    check("Brain600", f"{c} mean abs dDice",
          a.mean(), float(hr.loc[c, "mean_abs_delta"]))
    check("Brain600", f"{c} mean signed dDice",
          delta.mean(), float(hr.loc[c, "mean_signed_delta"]))

    for t, col in [
        (.10, "abs_gt_010"),
        (.20, "abs_gt_020"),
        (.50, "abs_gt_050"),
    ]:
        check("Brain600", f"{c} |dDice| > {t}",
              np.mean(a > t), float(hr.loc[c, col]))

# Brain600 bootstrap
bc = pd.read_csv(P / "bootstrap_95ci.csv")

def stored_ci(scope, metric):
    z = bc[
        (bc.scope == scope) &
        (bc.metric == metric)
    ].iloc[0]
    return float(z.ci_low), float(z.ci_high)

rng = np.random.default_rng(SEED)

b_range = np.empty(B)
b_oracle = np.empty(B)
b_best = np.empty(B)
b_adv = np.empty(B)

b_abs = {c: np.empty(B) for c in conds[1:]}

for b in range(B):
    idx = rng.integers(0, len(M), len(M))
    Z = M[idx]

    b_range[b] = (Z.max(axis=1) - Z.min(axis=1)).mean()
    b_oracle[b] = Z.max(axis=1).mean()
    b_best[b] = Z.mean(axis=0).max()
    b_adv[b] = b_oracle[b] - b_best[b]

    for j, c in enumerate(conds[1:], 1):
        b_abs[c][b] = np.abs(Z[:, j] - Z[:, 0]).mean()

for metric, arr in [
    ("mean_range_H0_L5", b_range),
    ("oracle_mean_H0_L5", b_oracle),
    ("best_fixed_mean_H0_L5", b_best),
    ("oracle_advantage_H0_L5", b_adv),
]:
    lo, hi = ci(arr)
    elo, ehi = stored_ci("H0_L5", metric)

    check("Brain600", metric + " CI low",
          lo, elo, 2e-9)
    check("Brain600", metric + " CI high",
          hi, ehi, 2e-9)

for c in conds[1:]:
    lo, hi = ci(b_abs[c])
    elo, ehi = stored_ci(c, "mean_abs_delta")

    check("Brain600", f"{c} mean abs CI low",
          lo, elo, 2e-9)
    check("Brain600", f"{c} mean abs CI high",
          hi, ehi, 2e-9)

# ============================================================
# TEXT3DSAM
# ============================================================

P = ROOT / "beyond-average-dice/results/text3dsam"

df = pd.read_csv(P / "text3dsam_amos30_liver_language.csv")

T = df[[f"dice_P{i}" for i in range(4)]].to_numpy(float)

ran = T.max(axis=1) - T.min(axis=1)
ora = T.max(axis=1)
means = T.mean(axis=0)
bi = int(np.argmax(means))
adv = ora.mean() - means[bi]

ts = pd.read_csv(P / "text3dsam_amos30_bootstrap.csv").set_index("metric")

check("Text3DSAM", "N", len(T), 30, 0)

for i in range(4):
    check("Text3DSAM", f"P{i} mean",
          means[i],
          float(ts.loc[f"P{i}_mean", "point"]))

check("Text3DSAM", "mean range",
      ran.mean(),
      float(ts.loc["range_mean", "point"]))

check("Text3DSAM", "max range",
      ran.max(),
      float(df.range_dice.max()))

check("Text3DSAM", "fraction range > .01",
      np.mean(ran > .01),
      float(ts.loc["range_gt_001", "point"]))

check("Text3DSAM", "oracle advantage",
      adv,
      float(ts.loc["oracle_over_best_fixed", "point"]))

# ============================================================
# WRITE FINAL QC
# ============================================================

out = pd.DataFrame(
    rows,
    columns=[
        "section", "metric", "computed",
        "expected", "abs_diff", "status"
    ]
)

out.to_csv("MANUSCRIPT_QC.csv", index=False)

funnel = pd.DataFrame([
    [
        "Complete Brain400 pairwise universe",
        len(pw), pw.filename.nunique(),
        pw.filename.map(pid).nunique(),
        "Localization-segmentation association"
    ],
    [
        "General high-overlap residual: IoU>=0.90 and |dDice|>0.20",
        len(general), general.filename.nunique(),
        general.filename.map(pid).nunique(),
        "Material disagreement despite preserved localization"
    ],
    [
        "Catastrophic failure-success: Dice<0.10 vs >=0.50",
        len(fs), fs.filename.nunique(),
        fs.patient_id.nunique(),
        "94.26% have box IoU<0.50"
    ],
    [
        "Catastrophic high-overlap: IoU>=0.90",
        len(high), high.filename.nunique(),
        high.patient_id.nunique(),
        "Rare residual with candidate-selection signatures"
    ],
], columns=[
    "population", "n_pairs", "n_images",
    "n_patients", "key_finding"
])

funnel.to_csv("BRAIN400_FUNNEL_TABLE.csv", index=False)

fails = out[out.status == "FAIL"]

print("\n" + "="*72)
print("FINAL MANUSCRIPT QC")
print("="*72)
print("TOTAL:", len(out))
print("PASS :", (out.status == "PASS").sum())
print("FAIL :", len(fails))

print("\nBRAIN400 FUNNEL")
print(funnel.to_string(index=False))

if len(fails):
    print("\n*** QC FAILURES ***")
    print(fails.to_string(index=False))
    raise SystemExit(1)
else:
    print("\nALL CHECKS PASSED.")
