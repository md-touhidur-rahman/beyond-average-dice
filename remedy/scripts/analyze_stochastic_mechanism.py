from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

ROOT = Path("/home/hpc/rlvl/rlvl178v/prl_medclipsam")
BASE = ROOT / "beyond-average-dice/remedy/results/stochastic_mechanism"

NPZ = BASE / "raw_heatmaps_float16.npz"
BOX  = BASE / "mechanism_boxes.csv"

PAIR_OUT = BASE / "heatmap_pairwise_mechanism.csv"
CASE_OUT = BASE / "case_mechanism_summary.csv"
REG_OUT  = BASE / "region_mechanism_summary.csv"
TXT_OUT  = BASE / "mechanism_analysis.txt"

z = np.load(NPZ)
H = z["heatmaps"]
keys = z["keys"].astype(str)

box = pd.read_csv(BOX)

assert H.shape == (1200, 512, 512)
assert len(keys) == 1200
assert len(set(keys.tolist())) == 1200
assert len(box) == 1200

# Parse keys: filename|Pxx|seed
meta = []
for i, key in enumerate(keys):
    fn, prompt, seed = key.rsplit("|", 2)
    meta.append({
        "idx": i,
        "filename": fn,
        "prompt": prompt,
        "seed": int(seed),
    })

meta = pd.DataFrame(meta)

# Attach frozen risk region/risk only for stratified description.
need = box[
    ["filename", "risk_region", "frozen_risk"]
].drop_duplicates()

assert need["filename"].nunique() == 12

meta = meta.merge(
    need,
    on="filename",
    how="left",
    validate="many_to_one"
)

assert meta["risk_region"].notna().all()

lookup = {
    (r.filename, r.prompt, int(r.seed)): int(r.idx)
    for r in meta.itertuples(index=False)
}

# Box lookup from the byte-identical reproduced factorial.
box_lookup = {
    (str(r.filename), str(r.prompt_id), int(r.seed)):
    np.array([r.x1, r.y1, r.x2, r.y2], dtype=np.float64)
    for r in box.itertuples(index=False)
}

def box_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    iw = max(0.0, x2 - x1 + 1.0)
    ih = max(0.0, y2 - y1 + 1.0)
    inter = iw * ih

    aa = max(0.0, a[2] - a[0] + 1.0) * max(0.0, a[3] - a[1] + 1.0)
    bb = max(0.0, b[2] - b[0] + 1.0) * max(0.0, b[3] - b[1] + 1.0)

    den = aa + bb - inter
    return 1.0 if den <= 0 else inter / den

def distances(i, j):
    # Cast float16 storage back to float32 before calculations.
    a = H[i].astype(np.float32, copy=False).ravel()
    b = H[j].astype(np.float32, copy=False).ravel()

    mae = float(np.mean(np.abs(a - b)))

    ac = a - a.mean()
    bc = b - b.mean()

    den = float(
        np.sqrt(np.dot(ac, ac)) *
        np.sqrt(np.dot(bc, bc))
    )

    if den == 0.0:
        corr = 1.0 if np.array_equal(a, b) else 0.0
    else:
        corr = float(np.dot(ac, bc) / den)
        corr = max(-1.0, min(1.0, corr))

    corr_dist = 1.0 - corr

    return mae, corr_dist

rows = []

files = sorted(meta["filename"].unique())
prompts = sorted(meta["prompt"].unique())
seeds = sorted(meta["seed"].unique())

assert len(files) == 12
assert len(prompts) == 20
assert len(seeds) == 5

# ------------------------------------------------------------
# Seed comparisons: same case + same prompt, different seed.
# 12 * 20 * C(5,2) = 2400
# ------------------------------------------------------------
for fn in files:
    rr = meta.loc[meta["filename"].eq(fn), "risk_region"].iloc[0]
    fr = float(meta.loc[meta["filename"].eq(fn), "frozen_risk"].iloc[0])

    for prompt in prompts:
        for s1, s2 in combinations(seeds, 2):
            i = lookup[(fn, prompt, s1)]
            j = lookup[(fn, prompt, s2)]

            mae, cd = distances(i, j)

            b1 = box_lookup[(fn, prompt, s1)]
            b2 = box_lookup[(fn, prompt, s2)]

            rows.append({
                "filename": fn,
                "risk_region": rr,
                "frozen_risk": fr,
                "comparison": "seed",
                "prompt1": prompt,
                "prompt2": prompt,
                "seed1": s1,
                "seed2": s2,
                "heatmap_mae": mae,
                "heatmap_corr_distance": cd,
                "box_instability": 1.0 - box_iou(b1, b2),
            })

# ------------------------------------------------------------
# Language comparisons: same case + same seed, different prompt.
# 12 * 5 * C(20,2) = 11400
# ------------------------------------------------------------
for fn in files:
    rr = meta.loc[meta["filename"].eq(fn), "risk_region"].iloc[0]
    fr = float(meta.loc[meta["filename"].eq(fn), "frozen_risk"].iloc[0])

    for seed in seeds:
        for p1, p2 in combinations(prompts, 2):
            i = lookup[(fn, p1, seed)]
            j = lookup[(fn, p2, seed)]

            mae, cd = distances(i, j)

            b1 = box_lookup[(fn, p1, seed)]
            b2 = box_lookup[(fn, p2, seed)]

            rows.append({
                "filename": fn,
                "risk_region": rr,
                "frozen_risk": fr,
                "comparison": "language",
                "prompt1": p1,
                "prompt2": p2,
                "seed1": seed,
                "seed2": seed,
                "heatmap_mae": mae,
                "heatmap_corr_distance": cd,
                "box_instability": 1.0 - box_iou(b1, b2),
            })

pairs = pd.DataFrame(rows)

assert (pairs["comparison"] == "seed").sum() == 2400
assert (pairs["comparison"] == "language").sum() == 11400

pairs.to_csv(PAIR_OUT, index=False)

# Case is the experimental unit for summary.
case = (
    pairs
    .groupby(["filename", "risk_region", "frozen_risk", "comparison"], as_index=False)
    [["heatmap_mae", "heatmap_corr_distance", "box_instability"]]
    .mean()
)

wide = case.pivot(
    index=["filename", "risk_region", "frozen_risk"],
    columns="comparison",
    values=["heatmap_mae", "heatmap_corr_distance", "box_instability"]
)

wide.columns = [
    f"{metric}_{comparison}"
    for metric, comparison in wide.columns
]

wide = wide.reset_index()

for metric in ["heatmap_mae", "heatmap_corr_distance", "box_instability"]:
    wide[f"{metric}_language_minus_seed"] = (
        wide[f"{metric}_language"] -
        wide[f"{metric}_seed"]
    )
    wide[f"{metric}_language_to_seed"] = (
        wide[f"{metric}_language"] /
        wide[f"{metric}_seed"].replace(0, np.nan)
    )

wide.to_csv(CASE_OUT, index=False)

region = (
    wide
    .groupby("risk_region", as_index=False)
    .agg(
        n_cases=("filename", "size"),
        frozen_risk_mean=("frozen_risk", "mean"),

        heatmap_mae_seed=("heatmap_mae_seed", "mean"),
        heatmap_mae_language=("heatmap_mae_language", "mean"),

        heatmap_corr_distance_seed=("heatmap_corr_distance_seed", "mean"),
        heatmap_corr_distance_language=("heatmap_corr_distance_language", "mean"),

        box_instability_seed=("box_instability_seed", "mean"),
        box_instability_language=("box_instability_language", "mean"),
    )
)

region.to_csv(REG_OUT, index=False)

lines = []

def emit(x=""):
    print(x)
    lines.append(str(x))

emit("=== STOCHASTIC MECHANISM: RAW HEATMAP -> BOX ===")
emit(f"cases: {len(wide)}")
emit(f"pairwise rows: {len(pairs)}")
emit(f"seed comparisons: {(pairs.comparison == 'seed').sum()}")
emit(f"language comparisons: {(pairs.comparison == 'language').sum()}")
emit()

for metric, label in [
    ("heatmap_mae", "RAW HEATMAP MAE"),
    ("heatmap_corr_distance", "RAW HEATMAP CORRELATION DISTANCE"),
    ("box_instability", "BOX INSTABILITY (1-IoU)"),
]:
    s = float(wide[f"{metric}_seed"].mean())
    l = float(wide[f"{metric}_language"].mean())
    d = l - s
    ratio = l / s if s != 0 else np.nan
    n = int((wide[f"{metric}_language_minus_seed"] > 0).sum())

    emit(f"=== {label} ===")
    emit(f"mean seed:             {s:.6f}")
    emit(f"mean language:         {l:.6f}")
    emit(f"language - seed:       {d:+.6f}")
    emit(f"language / seed:       {ratio:.6f}")
    emit(f"language > seed cases: {n} / {len(wide)}")
    emit()

emit("=== BY RISK REGION ===")
emit(region.to_string(index=False))
emit()

emit("=== CASE TABLE ===")
cols = [
    "risk_region",
    "filename",
    "frozen_risk",

    "heatmap_mae_seed",
    "heatmap_mae_language",
    "heatmap_mae_language_minus_seed",

    "heatmap_corr_distance_seed",
    "heatmap_corr_distance_language",
    "heatmap_corr_distance_language_minus_seed",

    "box_instability_seed",
    "box_instability_language",
    "box_instability_language_minus_seed",
]
emit(wide[cols].sort_values(["risk_region", "filename"]).to_string(index=False))
emit()

emit("DEVELOPMENT DIAGNOSTIC ONLY: YES")
emit("GROUND TRUTH USED: NO")
emit("DICE OUTCOMES USED: NO")
emit("HELDOUT OUTCOMES ACCESSED: NO")
emit("METHOD/THRESHOLD TUNING: NO")
emit("PAIRWISE ROWS TREATED AS INDEPENDENT INFERENCE: NO")

TXT_OUT.write_text("\n".join(lines) + "\n")

print("\nsaved:", PAIR_OUT)
print("saved:", CASE_OUT)
print("saved:", REG_OUT)
print("saved:", TXT_OUT)
