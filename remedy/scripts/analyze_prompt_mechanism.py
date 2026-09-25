from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/hpc/rlvl/rlvl178v/prl_medclipsam")
BASE = ROOT / "beyond-average-dice/remedy/results/stochastic_mechanism"

PAIR = BASE / "heatmap_pairwise_mechanism.csv"
OUT  = BASE / "prompt_mechanism_summary.csv"
TXT  = BASE / "prompt_mechanism_analysis.txt"

df = pd.read_csv(PAIR)

print("COLUMNS:")
print(df.columns.tolist())

# Detect naming used by the existing analysis.
def pick(candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(f"None found: {candidates}")

type_col = pick(["comparison_type", "type", "comparison"])
p1_col   = pick(["prompt1", "prompt_1", "p1"])
p2_col   = pick(["prompt2", "prompt_2", "p2"])
mae_col  = pick(["heatmap_mae", "mae"])
corr_col = pick(["heatmap_corr_distance", "corr_distance"])
box_col  = pick(["box_instability", "box_1miou", "one_minus_iou"])

lang = df[df[type_col].astype(str).str.lower().str.contains("language")].copy()

assert len(lang) == 11400, len(lang)

prompts = [f"P{i:02d}" for i in range(1,21)]

rows = []

for p in prompts:
    z = lang[
        (lang[p1_col].astype(str) == p) |
        (lang[p2_col].astype(str) == p)
    ]

    rows.append({
        "prompt_id": p,
        "n_pairwise": len(z),
        "heatmap_mae_mean": z[mae_col].mean(),
        "heatmap_corr_distance_mean": z[corr_col].mean(),
        "box_instability_mean": z[box_col].mean(),
        "heatmap_mae_median": z[mae_col].median(),
        "heatmap_corr_distance_median": z[corr_col].median(),
        "box_instability_median": z[box_col].median(),
    })

out = pd.DataFrame(rows)

# Rank: larger = more destabilizing relative to the other paraphrases.
for c in [
    "heatmap_mae_mean",
    "heatmap_corr_distance_mean",
    "box_instability_mean"
]:
    out[c.replace("_mean","_rank")] = out[c].rank(
        ascending=False,
        method="average"
    )

out["mean_rank"] = out[
    [
        "heatmap_mae_rank",
        "heatmap_corr_distance_rank",
        "box_instability_rank"
    ]
].mean(axis=1)

out = out.sort_values("mean_rank")

lines = []
lines.append("=== PROMPT-LEVEL LANGUAGE MECHANISM ===")
lines.append(f"language pairwise rows: {len(lang)}")
lines.append(f"prompts: {len(out)}")
lines.append("")
lines.append("=== MOST DESTABILIZING PROMPTS ===")
lines.append(out.head(20).to_string(index=False))
lines.append("")
lines.append("DEVELOPMENT DIAGNOSTIC ONLY: YES")
lines.append("GROUND TRUTH USED: NO")
lines.append("DICE OUTCOMES USED: NO")
lines.append("HELDOUT OUTCOMES ACCESSED: NO")
lines.append("PROMPT TAXONOMY INVENTED POST-HOC: NO")
lines.append("PAIRWISE ROWS TREATED AS INDEPENDENT INFERENCE: NO")

text = "\n".join(lines)

print(text)
out.to_csv(OUT, index=False)
TXT.write_text(text + "\n")

print("\nsaved:", OUT)
print("saved:", TXT)
