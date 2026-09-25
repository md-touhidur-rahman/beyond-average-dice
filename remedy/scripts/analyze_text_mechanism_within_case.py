from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path("/home/hpc/rlvl/rlvl178v/prl_medclipsam")
BASE = ROOT / "beyond-average-dice/remedy/results/stochastic_mechanism"

PAIR = BASE / "heatmap_pairwise_mechanism.csv"
TEXT = BASE / "text_embedding_pairwise.csv"

OUT_CASE = BASE / "text_mechanism_within_case.csv"
OUT_ALL  = BASE / "text_mechanism_within_case_analysis.txt"

lang = pd.read_csv(PAIR)
txt  = pd.read_csv(TEXT)

print("=== LANGUAGE COLUMNS ===")
print(lang.columns.tolist())
print("=== TEXT COLUMNS ===")
print(txt.columns.tolist())

# Keep language comparisons only.
lang = lang.loc[
    lang["comparison"].astype(str).str.lower().eq("language")
].copy()

assert len(lang) == 11400, len(lang)

# Determine text-distance column.
distance_candidates = [
    "text_distance",
    "embedding_distance",
    "cosine_distance",
    "text_cosine_distance",
]

dist_col = None
for c in distance_candidates:
    if c in txt.columns:
        dist_col = c
        break

if dist_col is None:
    raise RuntimeError(
        "Cannot identify text-distance column. "
        f"Columns are: {txt.columns.tolist()}"
    )

# Canonicalize prompt pair orientation.
def canon(a, b):
    a, b = str(a), str(b)
    return (a, b) if a <= b else (b, a)

lang[["cp1", "cp2"]] = pd.DataFrame(
    [canon(a, b) for a, b in zip(lang["prompt1"], lang["prompt2"])],
    index=lang.index
)

txt[["cp1", "cp2"]] = pd.DataFrame(
    [canon(a, b) for a, b in zip(txt["prompt1"], txt["prompt2"])],
    index=txt.index
)

td = (
    txt[["cp1", "cp2", dist_col]]
    .drop_duplicates()
    .copy()
)

assert len(td) == 190, len(td)
assert not td.duplicated(["cp1", "cp2"]).any()

x = lang.merge(
    td,
    on=["cp1", "cp2"],
    how="left",
    validate="many_to_one"
)

assert len(x) == 11400
assert x[dist_col].notna().all()

metrics = [
    "heatmap_mae",
    "heatmap_corr_distance",
    "box_instability",
]

rows = []

for fn, g in x.groupby("filename", sort=True):

    base = {
        "filename": fn,
        "risk_region": g["risk_region"].iloc[0],
        "frozen_risk": float(g["frozen_risk"].iloc[0]),
        "n_comparisons": len(g),
    }

    assert len(g) == 950, (fn, len(g))

    for metric in metrics:
        pr = pearsonr(g[dist_col], g[metric])
        sr = spearmanr(g[dist_col], g[metric])

        # Compatible with older and newer SciPy versions.
        base[f"{metric}_pearson"] = float(pr[0])
        base[f"{metric}_spearman"] = float(sr[0])

    rows.append(base)

case = pd.DataFrame(rows)
assert len(case) == 12

case.to_csv(OUT_CASE, index=False)

lines = []

def emit(s=""):
    print(s)
    lines.append(str(s))

emit("=== TEXT DISTANCE -> LOCALIZATION WITHIN IMAGE ===")
emit(f"cases: {len(case)}")
emit(f"language comparisons: {len(x)}")
emit("comparisons per case: 950")
emit("")

for metric in metrics:

    p = case[f"{metric}_pearson"]
    s = case[f"{metric}_spearman"]

    emit(f"=== {metric} ===")
    emit(f"mean within-case Pearson:  {p.mean():.6f}")
    emit(f"median within-case Pearson:{p.median():.6f}")
    emit(f"positive Pearson cases:    {(p > 0).sum()} / 12")
    emit(f"mean within-case Spearman: {s.mean():.6f}")
    emit(f"median within-case Spearman:{s.median():.6f}")
    emit(f"positive Spearman cases:   {(s > 0).sum()} / 12")
    emit("")

emit("=== CASE TABLE ===")
emit(case.to_string(index=False))
emit("")
emit("DEVELOPMENT DIAGNOSTIC ONLY: YES")
emit("GROUND TRUTH USED: NO")
emit("DICE OUTCOMES USED: NO")
emit("HELDOUT OUTCOMES ACCESSED: NO")
emit("METHOD/THRESHOLD TUNING: NO")
emit("PAIRWISE ROWS TREATED AS INDEPENDENT INFERENCE: NO")
emit("")
emit(
    "INTERPRETATION TARGET: determine whether the association between "
    "text-representation distance and localization change recurs within "
    "individual images rather than arising only after aggregation."
)

OUT_ALL.write_text("\n".join(lines) + "\n")

print("\nsaved:", OUT_CASE)
print("saved:", OUT_ALL)
