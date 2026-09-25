from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path("/home/hpc/rlvl/rlvl178v/prl_medclipsam")
BASE = ROOT / "beyond-average-dice/remedy/results/stochastic_mechanism"

PAIR = BASE / "heatmap_pairwise_mechanism.csv"
TEXT = BASE / "text_embedding_pairwise.csv"

OUT = BASE / "text_mechanism_casecollapsed.csv"
TXT = BASE / "text_mechanism_casecollapsed_analysis.txt"

lang = pd.read_csv(PAIR)
text = pd.read_csv(TEXT)

lang = lang.loc[
    lang["comparison"].astype(str).str.lower().eq("language")
].copy()

assert len(lang) == 11400

def canon(a, b):
    a, b = str(a), str(b)
    return (a, b) if a <= b else (b, a)

lang[["cp1","cp2"]] = pd.DataFrame(
    [canon(a,b) for a,b in zip(lang.prompt1, lang.prompt2)],
    index=lang.index
)

text[["cp1","cp2"]] = pd.DataFrame(
    [canon(a,b) for a,b in zip(text.prompt1, text.prompt2)],
    index=text.index
)

td = (
    text[["cp1","cp2","text_cosine_distance"]]
    .drop_duplicates()
)

assert len(td) == 190
assert not td.duplicated(["cp1","cp2"]).any()

# Collapse the five seed-matched realizations BEFORE correlation.
collapsed = (
    lang.groupby(
        ["filename","risk_region","frozen_risk","cp1","cp2"],
        as_index=False
    )
    .agg(
        heatmap_mae=("heatmap_mae","mean"),
        heatmap_corr_distance=("heatmap_corr_distance","mean"),
        box_instability=("box_instability","mean"),
        n_seed_realizations=("seed1","size"),
    )
)

assert len(collapsed) == 12 * 190
assert (collapsed["n_seed_realizations"] == 5).all()

x = collapsed.merge(
    td,
    on=["cp1","cp2"],
    how="left",
    validate="many_to_one"
)

assert len(x) == 2280
assert x["text_cosine_distance"].notna().all()

metrics = [
    "heatmap_mae",
    "heatmap_corr_distance",
    "box_instability",
]

rows = []

for fn, g in x.groupby("filename", sort=True):

    assert len(g) == 190

    row = {
        "filename": fn,
        "risk_region": g["risk_region"].iloc[0],
        "frozen_risk": float(g["frozen_risk"].iloc[0]),
        "n_unique_prompt_pairs": len(g),
    }

    for metric in metrics:
        row[f"{metric}_pearson"] = float(
            pearsonr(g["text_cosine_distance"], g[metric])[0]
        )
        row[f"{metric}_spearman"] = float(
            spearmanr(g["text_cosine_distance"], g[metric])[0]
        )

    rows.append(row)

case = pd.DataFrame(rows)
assert len(case) == 12
case.to_csv(OUT, index=False)

lines = []

def emit(x=""):
    print(x)
    lines.append(str(x))

emit("=== TEXT DISTANCE -> LOCALIZATION: SEED-COLLAPSED WITHIN IMAGE ===")
emit("cases: 12")
emit("unique prompt pairs per case: 190")
emit("case-prompt-pair observations: 2280")
emit("five matched seed realizations averaged before correlation")
emit("")

for metric in metrics:
    p = case[f"{metric}_pearson"]
    s = case[f"{metric}_spearman"]

    emit(f"=== {metric} ===")
    emit(f"mean Pearson:    {p.mean():.6f}")
    emit(f"median Pearson:  {p.median():.6f}")
    emit(f"positive cases:  {(p > 0).sum()} / 12")
    emit(f"mean Spearman:   {s.mean():.6f}")
    emit(f"median Spearman: {s.median():.6f}")
    emit(f"positive cases:  {(s > 0).sum()} / 12")
    emit("")

emit("=== CASE TABLE ===")
emit(case.to_string(index=False))
emit("")
emit("DEVELOPMENT DIAGNOSTIC ONLY: YES")
emit("GROUND TRUTH USED: NO")
emit("DICE OUTCOMES USED: NO")
emit("HELDOUT OUTCOMES ACCESSED: NO")
emit("METHOD/THRESHOLD TUNING: NO")
emit("SEED REALIZATIONS COLLAPSED BEFORE ASSOCIATION: YES")
emit("UNIQUE PROMPT PAIRS TREATED AS INDEPENDENT FORMAL INFERENCE: NO")

TXT.write_text("\n".join(lines) + "\n")

print("\nsaved:", OUT)
print("saved:", TXT)
