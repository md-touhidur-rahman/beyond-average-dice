from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("/home/hpc/rlvl/rlvl178v/prl_medclipsam")
BASE = ROOT / "beyond-average-dice/remedy/results/embedding_trajectory"

INP = BASE / "trajectory_mechanism.csv"

OUT_TRAJ = BASE / "trajectory_analysis_1200.csv"
OUT_PAIR = BASE / "trajectory_analysis_240.csv"
OUT_CASE = BASE / "trajectory_analysis_12.csv"
OUT_TXT  = BASE / "trajectory_analysis.txt"

df = pd.read_csv(INP)

# ------------------------------------------------------------
# Integrity gates
# ------------------------------------------------------------
assert len(df) == 6000
assert df["filename"].nunique() == 12
assert df["pair_id"].nunique() == 20
assert df["seed"].nunique() == 5
assert sorted(df["alpha"].unique().tolist()) == [
    0.0, 0.25, 0.5, 0.75, 1.0
]

keys = ["filename", "pair_id", "seed", "alpha"]
assert df.duplicated(keys).sum() == 0
assert df.isna().sum().sum() == 0

# ------------------------------------------------------------
# Box instability
# ------------------------------------------------------------
def box_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih

    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    union = aa + ab - inter

    if union <= 0:
        return 1.0 if np.allclose(a, b) else 0.0

    return inter / union


traj_keys = ["filename", "pair_id", "seed"]

df["box_instability_from_A"] = np.nan
df["box_instability_from_B"] = np.nan

for _, idx in df.groupby(traj_keys, sort=False).groups.items():
    g = df.loc[idx].sort_values("alpha")

    assert len(g) == 5

    a = g.iloc[0][["x1","y1","x2","y2"]].to_numpy(float)
    b = g.iloc[-1][["x1","y1","x2","y2"]].to_numpy(float)

    for ridx, r in g.iterrows():
        q = r[["x1","y1","x2","y2"]].to_numpy(float)

        df.loc[ridx, "box_instability_from_A"] = 1.0 - box_iou(a, q)
        df.loc[ridx, "box_instability_from_B"] = 1.0 - box_iou(b, q)

assert df[
    ["box_instability_from_A", "box_instability_from_B"]
].notna().all().all()

# ------------------------------------------------------------
# Endpoint progress
# ------------------------------------------------------------
den = (
    df["corr_distance_from_A"] +
    df["corr_distance_from_B"]
)

df["endpoint_progress"] = np.where(
    den > 0,
    df["corr_distance_from_A"] / den,
    np.nan
)

# Endpoint rows should have defined progress unless A and B maps
# themselves are identical. Keep any undefined values explicit.
# No result-based exclusion occurs below.

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def rho(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y)

    if ok.sum() < 2:
        return np.nan

    if np.unique(x[ok]).size < 2:
        return np.nan

    if np.unique(y[ok]).size < 2:
        return 0.0

    return float(spearmanr(x[ok], y[ok])[0])


def nondecreasing(y, tol=1e-12):
    y = np.asarray(y, dtype=float)
    if not np.isfinite(y).all():
        return False
    return bool(np.all(np.diff(y) >= -tol))


def nonincreasing(y, tol=1e-12):
    y = np.asarray(y, dtype=float)
    if not np.isfinite(y).all():
        return False
    return bool(np.all(np.diff(y) <= tol))


# ------------------------------------------------------------
# Level 1: 1200 trajectories
# ------------------------------------------------------------
rows = []

for k, g in df.groupby(traj_keys, sort=True):
    g = g.sort_values("alpha")
    assert len(g) == 5

    t = g["alpha"].to_numpy(float)

    da = g["corr_distance_from_A"].to_numpy(float)
    db = g["corr_distance_from_B"].to_numpy(float)
    ma = g["heatmap_mae_from_A"].to_numpy(float)
    ba = g["box_instability_from_A"].to_numpy(float)
    ep = g["endpoint_progress"].to_numpy(float)

    rows.append({
        "filename": k[0],
        "pair_id": int(k[1]),
        "seed": int(k[2]),
        "risk_region": g["risk_region"].iloc[0],
        "frozen_risk": float(g["frozen_risk"].iloc[0]),
        "prompt1": g["prompt1"].iloc[0],
        "prompt2": g["prompt2"].iloc[0],
        "distance_quintile": int(g["distance_quintile"].iloc[0]),
        "text_cosine_distance":
            float(g["text_cosine_distance"].iloc[0]),

        "corr_A_spearman": rho(t, da),
        "corr_A_nondecreasing": nondecreasing(da),

        "corr_B_spearman": rho(t, db),
        "corr_B_nonincreasing": nonincreasing(db),

        "mae_A_spearman": rho(t, ma),
        "mae_A_nondecreasing": nondecreasing(ma),

        "box_A_spearman": rho(t, ba),
        "box_A_nondecreasing": nondecreasing(ba),

        "progress_spearman": rho(t, ep),
        "progress_nondecreasing":
            nondecreasing(ep) if np.isfinite(ep).all() else False,

        "progress_finite": bool(np.isfinite(ep).all()),
    })

traj = pd.DataFrame(rows)

assert len(traj) == 1200
assert traj["filename"].nunique() == 12

# ------------------------------------------------------------
# Level 2: 240 case x prompt-pair summaries
# Collapse five matched seeds.
# ------------------------------------------------------------
pair_rows = []

for (fn, pid), g in traj.groupby(["filename", "pair_id"], sort=True):
    pair_rows.append({
        "filename": fn,
        "pair_id": int(pid),
        "risk_region": g["risk_region"].iloc[0],
        "frozen_risk": float(g["frozen_risk"].iloc[0]),
        "prompt1": g["prompt1"].iloc[0],
        "prompt2": g["prompt2"].iloc[0],
        "distance_quintile": int(g["distance_quintile"].iloc[0]),
        "text_cosine_distance":
            float(g["text_cosine_distance"].iloc[0]),

        "corr_A_spearman_mean":
            g["corr_A_spearman"].mean(),
        "corr_A_spearman_median":
            g["corr_A_spearman"].median(),

        "corr_B_spearman_mean":
            g["corr_B_spearman"].mean(),

        "mae_A_spearman_mean":
            g["mae_A_spearman"].mean(),

        "box_A_spearman_mean":
            g["box_A_spearman"].mean(),

        "progress_spearman_mean":
            g["progress_spearman"].mean(),

        "corr_A_positive_seeds":
            int((g["corr_A_spearman"] > 0).sum()),

        "corr_A_nondecreasing_seeds":
            int(g["corr_A_nondecreasing"].sum()),
    })

pair = pd.DataFrame(pair_rows)
assert len(pair) == 240

# ------------------------------------------------------------
# Level 3: 12 case summaries
# ------------------------------------------------------------
case_rows = []

for fn, g in traj.groupby("filename", sort=True):
    case_rows.append({
        "filename": fn,
        "risk_region": g["risk_region"].iloc[0],
        "frozen_risk": float(g["frozen_risk"].iloc[0]),
        "n_trajectories": len(g),

        "corr_A_spearman_mean":
            g["corr_A_spearman"].mean(),
        "corr_A_spearman_median":
            g["corr_A_spearman"].median(),
        "corr_A_positive_fraction":
            (g["corr_A_spearman"] > 0).mean(),
        "corr_A_nondecreasing_fraction":
            g["corr_A_nondecreasing"].mean(),

        "corr_B_spearman_mean":
            g["corr_B_spearman"].mean(),

        "mae_A_spearman_mean":
            g["mae_A_spearman"].mean(),

        "box_A_spearman_mean":
            g["box_A_spearman"].mean(),

        "progress_spearman_mean":
            g["progress_spearman"].mean(),
    })

case = pd.DataFrame(case_rows)
assert len(case) == 12

# ------------------------------------------------------------
# Frozen-plan reporting
# ------------------------------------------------------------
def describe_rho(series):
    x = pd.Series(series).dropna()

    return {
        "n": len(x),
        "mean": x.mean(),
        "median": x.median(),
        "q1": x.quantile(.25),
        "q3": x.quantile(.75),
        "min": x.min(),
        "max": x.max(),
        "positive": int((x > 0).sum()),
        "zero": int((x == 0).sum()),
        "negative": int((x < 0).sum()),
    }


lines = []

def out(s=""):
    print(s)
    lines.append(str(s))


out("=== EMBEDDING INTERPOLATION -> LOCALIZATION ===")
out("PRE-SPECIFIED ANALYSIS OF FROZEN MECHANISM RUN")
out()
out(f"rows: {len(df)}")
out(f"trajectories: {len(traj)}")
out(f"case-prompt-pair summaries: {len(pair)}")
out(f"cases: {len(case)}")
out()

# Primary
d = describe_rho(traj["corr_A_spearman"])

out("=== PRIMARY: CORRELATION DISTANCE FROM A ===")
out(f"successfully evaluated: {d['n']} / 1200")
out(f"mean Spearman:   {d['mean']:.6f}")
out(f"median Spearman: {d['median']:.6f}")
out(f"IQR:             [{d['q1']:.6f}, {d['q3']:.6f}]")
out(f"range:           [{d['min']:.6f}, {d['max']:.6f}]")
out(f"positive:        {d['positive']} / {d['n']} "
    f"({d['positive']/d['n']:.3%})")
out(f"zero:            {d['zero']} / {d['n']} "
    f"({d['zero']/d['n']:.3%})")
out(f"negative:        {d['negative']} / {d['n']} "
    f"({d['negative']/d['n']:.3%})")
out(
    "fully non-decreasing: "
    f"{int(traj['corr_A_nondecreasing'].sum())} / {len(traj)} "
    f"({traj['corr_A_nondecreasing'].mean():.3%})"
)
out()

# Secondary MAE
d = describe_rho(traj["mae_A_spearman"])
out("=== SECONDARY: HEATMAP MAE FROM A ===")
out(f"mean Spearman:   {d['mean']:.6f}")
out(f"median Spearman: {d['median']:.6f}")
out(f"positive:        {d['positive']} / {d['n']} "
    f"({d['positive']/d['n']:.3%})")
out(
    "fully non-decreasing: "
    f"{int(traj['mae_A_nondecreasing'].sum())} / {len(traj)} "
    f"({traj['mae_A_nondecreasing'].mean():.3%})"
)
out()

# Secondary box
d = describe_rho(traj["box_A_spearman"])
out("=== SECONDARY: BOX INSTABILITY FROM A ===")
out(f"mean Spearman:   {d['mean']:.6f}")
out(f"median Spearman: {d['median']:.6f}")
out(f"positive:        {d['positive']} / {d['n']} "
    f"({d['positive']/d['n']:.3%})")
out(
    "fully non-decreasing: "
    f"{int(traj['box_A_nondecreasing'].sum())} / {len(traj)} "
    f"({traj['box_A_nondecreasing'].mean():.3%})"
)
out()

# B
d = describe_rho(traj["corr_B_spearman"])
out("=== SYMMETRIC CHECK: CORRELATION DISTANCE FROM B ===")
out(f"mean Spearman:   {d['mean']:.6f}")
out(f"median Spearman: {d['median']:.6f}")
out(f"negative:        {d['negative']} / {d['n']} "
    f"({d['negative']/d['n']:.3%})")
out(
    "fully non-increasing: "
    f"{int(traj['corr_B_nonincreasing'].sum())} / {len(traj)} "
    f"({traj['corr_B_nonincreasing'].mean():.3%})"
)
out()

# Progress
d = describe_rho(traj["progress_spearman"])
out("=== SECONDARY: ENDPOINT PROGRESS ===")
out(
    "fully finite trajectories: "
    f"{int(traj['progress_finite'].sum())} / {len(traj)}"
)
out(f"mean Spearman:   {d['mean']:.6f}")
out(f"median Spearman: {d['median']:.6f}")
out(f"positive:        {d['positive']} / {d['n']} "
    f"({d['positive']/d['n']:.3%})")
out(
    "fully non-decreasing: "
    f"{int(traj['progress_nondecreasing'].sum())} / {len(traj)} "
    f"({traj['progress_nondecreasing'].mean():.3%})"
)
out()

out("=== CASE-LEVEL PRIMARY REPLICATION ===")
out(case.to_string(index=False))
out()

out("=== CASE-LEVEL DIRECTION ===")
out(
    "cases with positive mean primary Spearman: "
    f"{int((case['corr_A_spearman_mean'] > 0).sum())} / 12"
)
out(
    "cases with negative mean B Spearman: "
    f"{int((case['corr_B_spearman_mean'] < 0).sum())} / 12"
)
out(
    "cases with positive mean progress Spearman: "
    f"{int((case['progress_spearman_mean'] > 0).sum())} / 12"
)
out()

out("DEVELOPMENT DIAGNOSTIC ONLY: YES")
out("GROUND TRUTH USED: NO")
out("DICE OUTCOMES USED: NO")
out("HELDOUT OUTCOMES ACCESSED: NO")
out("METHOD/THRESHOLD TUNING FROM TRAJECTORY OUTCOMES: NO")
out("RESULT-BASED TRAJECTORY EXCLUSION: NO")
out("1200 TRAJECTORIES TREATED AS INDEPENDENT FORMAL INFERENCE: NO")

traj.to_csv(OUT_TRAJ, index=False)
pair.to_csv(OUT_PAIR, index=False)
case.to_csv(OUT_CASE, index=False)

OUT_TXT.write_text("\n".join(lines) + "\n")

print()
print("saved:", OUT_TRAJ)
print("saved:", OUT_PAIR)
print("saved:", OUT_CASE)
print("saved:", OUT_TXT)
