from pathlib import Path
from itertools import combinations
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path("/home/hpc/rlvl/rlvl178v/prl_medclipsam")
V2 = ROOT / "MedCLIP-SAMv2"

BASE = ROOT / "beyond-average-dice/remedy/results/stochastic_mechanism"
PAIR = BASE / "heatmap_pairwise_mechanism.csv"

OUT_PAIR = BASE / "text_embedding_pairwise.csv"
OUT_TXT  = BASE / "text_embedding_mechanism.txt"

sys.path.insert(0, str(V2 / "saliency_maps"))
sys.path.insert(0, str(V2))

from transformers import AutoModel, AutoTokenizer

PROMPTS = [
    "A medical brain MRI scan revealing a suspicious, irregularly shaped mass suggestive of a brain tumor.",
    "A brain MRI displaying an abnormal mass that could indicate the presence of a tumor.",
    "A brain imaging study showing an irregular mass potentially indicative of a brain tumor.",
    "A medical brain MRI scan identifying an unusual mass suggestive of a tumor in the brain.",
    "A brain MRI revealing a prominent mass within the brain tissue suggestive of a tumor.",
    "A brain imaging scan showing an irregularly shaped mass that may be indicative of a brain tumor.",
    "A brain MRI detecting a concerning mass suggestive of a tumor in the brain.",
    "A medical brain MRI scan showing a significant mass that could indicate a brain tumor.",
    "A brain imaging study identifying an abnormal, irregular mass suggestive of a brain tumor.",
    "A brain MRI displaying a notable mass within the brain potentially indicative of a tumor.",
    "A brain MRI revealing an irregular mass that could suggest a brain tumor.",
    "A brain imaging scan showing an abnormal mass that may be indicative of a brain tumor.",
    "A medical brain MRI scan identifying a distinct mass suggestive of a tumor in the brain.",
    "A brain MRI displaying an irregular mass that could be indicative of a brain tumor.",
    "A brain imaging study showing an unusual mass that may indicate a brain tumor.",
    "A brain MRI revealing an abnormal mass within the brain suggestive of a tumor.",
    "A medical brain MRI scan detecting a suspicious mass that could indicate a brain tumor.",
    "A brain imaging scan showing a prominent, irregular mass suggestive of a brain tumor.",
    "A brain MRI revealing an unusual mass potentially indicative of a brain tumor.",
    "A medical brain MRI scan identifying an abnormal, irregularly shaped mass suggestive of a brain tumor.",
]

assert len(PROMPTS) == 20
assert len(set(PROMPTS)) == 20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(12)
np.random.seed(12)

print("=== LOADING FROZEN MODEL ===", flush=True)
print("device:", DEVICE, flush=True)

model = AutoModel.from_pretrained(
    str(V2 / "saliency_maps/model"),
    trust_remote_code=True
).to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained(
    str(ROOT / "chuhac_BiomedCLIP-vit-bert-hf"),
    trust_remote_code=True
)

model.eval()

# Match the tokenization convention used by the M2IB experiment.
ids_list = [
    tokenizer.encode(p, add_special_tokens=True)
    for p in PROMPTS
]

lengths = [len(x) for x in ids_list]

# IMPORTANT:
# Preserve the exact per-prompt tokenization convention used by the
# original M2IB experiment. Do not pad prompts together.
#
# Original M2IB:
#   ids = torch.tensor(
#       [tokenizer.encode(text, add_special_tokens=True)],
#       device=DEVICE
#   )
#
# Therefore extract each text representation independently.

emb_list = []

with torch.no_grad():
    for prompt_id, token_ids in enumerate(ids_list, start=1):
        ids = torch.tensor(
            [token_ids],
            device=DEVICE
        )

        e = model.get_text_features(ids)

        assert e.shape[0] == 1

        emb_list.append(
            e.detach().float().cpu().numpy()[0]
        )

        print(
            f"P{prompt_id:02d}: "
            f"tokens={len(token_ids)} "
            f"embedding_shape={tuple(e.shape)}",
            flush=True
        )

emb = np.stack(emb_list, axis=0)

# CLIP-style normalized representation used for similarity.
norm = np.linalg.norm(emb, axis=1, keepdims=True)
assert np.all(norm > 0)
embn = emb / norm

print("embedding shape:", emb.shape, flush=True)
print("token lengths:", lengths, flush=True)

# ------------------------------------------------------------
# 190 unique prompt-pair embedding distances.
# ------------------------------------------------------------

rows = []

for i, j in combinations(range(20), 2):
    cos = float(np.dot(embn[i], embn[j]))
    cos = float(np.clip(cos, -1.0, 1.0))

    rows.append({
        "prompt1": f"P{i+1:02d}",
        "prompt2": f"P{j+1:02d}",
        "text_cosine_similarity": cos,
        "text_cosine_distance": 1.0 - cos,
        "text_euclidean_normalized": float(
            np.linalg.norm(embn[i] - embn[j])
        ),
    })

textpair = pd.DataFrame(rows)

assert len(textpair) == 190
assert not textpair.isna().any().any()

# ------------------------------------------------------------
# Collapse existing LANGUAGE comparisons to the same 190 pairs.
# Do NOT treat 11,400 rows as independent observations.
# ------------------------------------------------------------

df = pd.read_csv(PAIR)

lang = df.loc[
    df["comparison"].astype(str).str.lower().eq("language")
].copy()

assert len(lang) == 11400

# Canonicalize pair ordering.
lang["pa"] = lang[["prompt1", "prompt2"]].min(axis=1)
lang["pb"] = lang[["prompt1", "prompt2"]].max(axis=1)

agg = (
    lang.groupby(["pa", "pb"], as_index=False)
    .agg(
        heatmap_mae_mean=("heatmap_mae", "mean"),
        heatmap_corr_distance_mean=("heatmap_corr_distance", "mean"),
        box_instability_mean=("box_instability", "mean"),
        n_observations=("heatmap_mae", "size"),
    )
    .rename(columns={"pa": "prompt1", "pb": "prompt2"})
)

assert len(agg) == 190
assert agg["n_observations"].eq(60).all(), agg["n_observations"].value_counts()

x = textpair.merge(
    agg,
    on=["prompt1", "prompt2"],
    how="inner",
    validate="one_to_one"
)

assert len(x) == 190

# ------------------------------------------------------------
# Pair-level association.
# Descriptive mechanism analysis; prompt pairs are not claimed
# to be statistically independent.
# ------------------------------------------------------------

metrics = [
    "heatmap_mae_mean",
    "heatmap_corr_distance_mean",
    "box_instability_mean",
]

lines = []
lines.append("=== TEXT REPRESENTATION -> LOCALIZATION MECHANISM ===")
lines.append(f"unique prompt pairs: {len(x)}")
lines.append("underlying language comparisons: 11400")
lines.append("")

for metric in metrics:
    pearson = x["text_cosine_distance"].corr(
        x[metric], method="pearson"
    )
    spearman = x["text_cosine_distance"].corr(
        x[metric], method="spearman"
    )

    lines.append(f"=== {metric} ===")
    lines.append(f"Pearson r:  {pearson:.6f}")
    lines.append(f"Spearman r: {spearman:.6f}")
    lines.append("")

lines.append("=== TEXT DISTANCE DISTRIBUTION ===")
lines.append(
    x["text_cosine_distance"]
    .describe()
    .to_string()
)
lines.append("")

# Quartile contrast gives an intuitive effect-size view.
x["text_distance_quartile"] = pd.qcut(
    x["text_cosine_distance"],
    4,
    labels=["Q1_nearest", "Q2", "Q3", "Q4_farthest"]
)

q = (
    x.groupby("text_distance_quartile", observed=True)
    [metrics]
    .mean()
)

lines.append("=== OUTCOMES BY TEXT-DISTANCE QUARTILE ===")
lines.append(q.to_string())
lines.append("")

lines.append("DEVELOPMENT DIAGNOSTIC ONLY: YES")
lines.append("GROUND TRUTH USED: NO")
lines.append("DICE OUTCOMES USED: NO")
lines.append("HELDOUT OUTCOMES ACCESSED: NO")
lines.append("METHOD/THRESHOLD TUNING: NO")
lines.append("PROMPT TAXONOMY INVENTED POST-HOC: NO")
lines.append("190 PROMPT PAIRS TREATED AS INDEPENDENT INFERENCE: NO")

text = "\n".join(lines)

print()
print(text, flush=True)

x.to_csv(OUT_PAIR, index=False)
OUT_TXT.write_text(text + "\n")

print("\nsaved:", OUT_PAIR, flush=True)
print("saved:", OUT_TXT, flush=True)
