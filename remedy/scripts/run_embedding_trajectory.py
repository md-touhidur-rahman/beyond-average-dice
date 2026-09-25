from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path("/home/hpc/rlvl/rlvl178v/prl_medclipsam")
V2   = ROOT / "MedCLIP-SAMv2"
DATA = ROOT / "MedCLIP-SAM/part2/action400_dataset"

PILOT = ROOT / "beyond-average-dice/remedy/results/stochastic_pilot/pilot_cases.csv"
TEXT  = ROOT / "beyond-average-dice/remedy/results/stochastic_mechanism/text_embedding_pairwise.csv"

OUTDIR = ROOT / "beyond-average-dice/remedy/results/embedding_trajectory"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT = OUTDIR / "trajectory_mechanism.csv"
PAIR_OUT = OUTDIR / "selected_prompt_pairs.csv"

sys.path.insert(0, str(V2 / "saliency_maps"))
sys.path.insert(0, str(V2))
sys.path.insert(
    0,
    str(ROOT / "beyond-average-dice/remedy/scripts")
)

from transformers import AutoModel, AutoProcessor, AutoTokenizer
from scripts.methods import (
    extract_feature_map,
    extract_bert_layer,
    get_compression_estimator,
)
from iba_embedding_intervention import IBAInterpreter

DEVICE = "cuda"

VLAYER = 9
VBETA  = 2.0
VVAR   = 0.3

SEEDS = [12, 23, 34, 45, 56]
ALPHAS = [0.0, 0.25, 0.50, 0.75, 1.0]

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
assert torch.cuda.is_available()

# ------------------------------------------------------------
# Development cases only.
# ------------------------------------------------------------

pilot = pd.read_csv(PILOT)

assert len(pilot) == 12
assert pilot["filename"].nunique() == 12

print("=== SAFETY ===", flush=True)
print("CASES: 12 DEVELOPMENT PILOT CASES", flush=True)
print("GROUND TRUTH LOADED: NO", flush=True)
print("DICE OUTCOMES USED: NO", flush=True)
print("HELDOUT OUTCOMES ACCESSED: NO", flush=True)
print("METHOD/THRESHOLD TUNING: NO", flush=True)

# ------------------------------------------------------------
# Frozen model.
# ------------------------------------------------------------

torch.manual_seed(12)
np.random.seed(12)

model = AutoModel.from_pretrained(
    str(V2 / "saliency_maps/model"),
    trust_remote_code=True
).to(DEVICE)

processor = AutoProcessor.from_pretrained(
    str(ROOT / "chuhac_BiomedCLIP-vit-bert-hf"),
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    str(ROOT / "chuhac_BiomedCLIP-vit-bert-hf"),
    trust_remote_code=True
)

model.eval()

# ------------------------------------------------------------
# Exact projected text representations.
# ------------------------------------------------------------

emb = {}

with torch.no_grad():
    for i, text in enumerate(PROMPTS, start=1):
        ids = torch.tensor(
            [tokenizer.encode(
                text,
                add_special_tokens=True
            )],
            device=DEVICE
        )

        e = model.get_text_features(ids)
        assert e.shape == (1, 512)

        emb[f"P{i:02d}"] = e.detach()

# ------------------------------------------------------------
# Deterministically select 20 prompt pairs from text-distance
# quintiles. Selection uses TEXT REPRESENTATION ONLY.
# No localization outcome enters pair selection.
# ------------------------------------------------------------

td = pd.read_csv(TEXT)

needed = {
    "prompt1",
    "prompt2",
    "text_cosine_distance",
}
assert needed.issubset(td.columns)
assert len(td) == 190

td = td[
    ["prompt1", "prompt2", "text_cosine_distance"]
].copy()

td = td.sort_values(
    ["text_cosine_distance", "prompt1", "prompt2"]
).reset_index(drop=True)

# Five equal-sized strata: 38 pairs each.
td["distance_quintile"] = (
    np.arange(len(td)) * 5 // len(td)
) + 1

selected = []

for q in range(1, 6):
    g = td[td["distance_quintile"] == q].reset_index(drop=True)
    assert len(g) == 38

    # Four deterministic positions spanning each stratum.
    idx = np.linspace(
        0,
        len(g) - 1,
        4
    ).round().astype(int)

    selected.append(g.iloc[idx])

pairs = pd.concat(
    selected,
    ignore_index=True
)

assert len(pairs) == 20
assert not pairs.duplicated(
    ["prompt1", "prompt2"]
).any()

pairs.to_csv(PAIR_OUT, index=False)

print("\n=== SELECTED PAIRS ===", flush=True)
print(
    pairs.to_string(index=False),
    flush=True
)

# ------------------------------------------------------------
# Raw linear interpolation preserving exact raw endpoints.
#
# This is intentionally NOT unit normalization:
# alpha=0 exactly equals e1
# alpha=1 exactly equals e2
#
# Since the IBA fitting term is cosine similarity, direction is
# mechanistically central, while preserving endpoints gives the
# cleanest equivalence test.
# ------------------------------------------------------------

def interpolate(e1, e2, alpha):
    if alpha == 0.0:
        return e1.clone()

    if alpha == 1.0:
        return e2.clone()

    return (1.0 - alpha) * e1 + alpha * e2


def embedding_heatmap(image_feat, target):
    features = extract_feature_map(
        model.vision_model,
        VLAYER,
        image_feat
    )

    layer = extract_bert_layer(
        model.vision_model,
        VLAYER
    )

    estimator = get_compression_estimator(
        VVAR,
        layer,
        features
    )

    reader = IBAInterpreter(
        model,
        estimator,
        beta=VBETA,
        lr=1,
        steps=10,
        progbar=False,
        ensemble=False
    )

    return reader.vision_heatmap_embedding_target(
        target,
        image_feat
    )


def kmeans_mask(vmap):
    from sklearn.cluster import KMeans

    h, w = vmap.shape

    small = cv2.resize(
        vmap,
        (256, 256),
        interpolation=cv2.INTER_NEAREST
    )

    km = KMeans(
        n_clusters=2,
        random_state=10,
        n_init=10
    )

    lab = km.fit_predict(
        small.reshape(-1, 1)
    ).reshape(256, 256)

    bg = np.argmin(
        km.cluster_centers_.flatten()
    )

    m = (lab != bg).astype(np.uint8)

    m = cv2.resize(
        m,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    n, labels, stats, _ = \
        cv2.connectedComponentsWithStats(m)

    if n <= 1:
        return m

    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))

    return (labels == idx).astype(np.uint8)


def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        h, w = mask.shape
        return np.array(
            [0, 0, w - 1, h - 1],
            dtype=np.float32
        )

    return np.array(
        [xs.min(), ys.min(), xs.max(), ys.max()],
        dtype=np.float32
    )


rows = []

expected = (
    len(pilot)
    * len(pairs)
    * len(SEEDS)
    * len(ALPHAS)
)

print("\n=== DESIGN ===", flush=True)
print("cases:", len(pilot), flush=True)
print("prompt pairs:", len(pairs), flush=True)
print("alphas:", ALPHAS, flush=True)
print("seeds:", SEEDS, flush=True)
print("planned runs:", expected, flush=True)
print("GPU:", torch.cuda.get_device_name(0), flush=True)

run_i = 0

for case_i, r in enumerate(
    pilot.itertuples(index=False),
    start=1
):
    fn = str(r.filename)

    image = Image.open(
        DATA / "test_images" / fn
    ).convert("RGB")

    image_feat = processor(
        images=image,
        return_tensors="pt"
    )["pixel_values"].to(DEVICE)

    risk_region = getattr(r, "risk_region", "")
    frozen_risk = getattr(r, "frozen_risk", np.nan)

    print(
        f"\n=== CASE {case_i}/12: {fn} ===",
        flush=True
    )

    for pair_i, pr in enumerate(
        pairs.itertuples(index=False),
        start=1
    ):
        p1 = str(pr.prompt1)
        p2 = str(pr.prompt2)

        e1 = emb[p1]
        e2 = emb[p2]

        for seed in SEEDS:
            # Generate the complete five-dose trajectory first so that
            # frozen raw-heatmap endpoint distances can be measured
            # directly. Only these five maps are retained temporarily.
            trajectory = []

            for alpha in ALPHAS:

                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                np.random.seed(seed)

                target = interpolate(
                    e1,
                    e2,
                    alpha
                )

                vm = embedding_heatmap(
                    image_feat,
                    target
                )

                vm = np.asarray(
                    vm,
                    dtype=np.float32
                )

                vm = cv2.resize(
                    vm,
                    image.size,
                    interpolation=cv2.INTER_NEAREST
                )

                vm = np.clip(vm, 0, 1)

                cm = kmeans_mask(vm)
                box = bbox_from_mask(cm)

                trajectory.append({
                    "alpha": alpha,
                    "vm": vm,
                    "box": box,
                    "heatmap_mean": float(vm.mean()),
                    "heatmap_max": float(vm.max()),
                    "target_norm":
                        float(target.norm().item()),
                })

            assert len(trajectory) == len(ALPHAS)
            assert trajectory[0]["alpha"] == 0.0
            assert trajectory[-1]["alpha"] == 1.0

            h_a = trajectory[0]["vm"]
            h_b = trajectory[-1]["vm"]

            def corr_distance(a, b):
                a = np.asarray(a, dtype=np.float64).ravel()
                b = np.asarray(b, dtype=np.float64).ravel()

                ac = a - a.mean()
                bc = b - b.mean()

                den = np.sqrt(
                    np.dot(ac, ac) * np.dot(bc, bc)
                )

                if den == 0.0:
                    # Degenerate constant maps are not silently assigned
                    # a correlation distance.
                    return np.nan

                corr = np.dot(ac, bc) / den
                corr = np.clip(corr, -1.0, 1.0)
                return float(1.0 - corr)

            for item in trajectory:
                alpha = item["alpha"]
                vm = item["vm"]
                box = item["box"]

                # Endpoint identities are mathematically fixed rather
                # than estimated numerically.
                if alpha == 0.0:
                    corr_a = 0.0
                    mae_a = 0.0
                else:
                    corr_a = corr_distance(vm, h_a)
                    mae_a = float(
                        np.mean(
                            np.abs(
                                vm.astype(np.float64) -
                                h_a.astype(np.float64)
                            )
                        )
                    )

                if alpha == 1.0:
                    corr_b = 0.0
                    mae_b = 0.0
                else:
                    corr_b = corr_distance(vm, h_b)
                    mae_b = float(
                        np.mean(
                            np.abs(
                                vm.astype(np.float64) -
                                h_b.astype(np.float64)
                            )
                        )
                    )

                rows.append({
                    "filename": fn,
                    "risk_region": risk_region,
                    "frozen_risk": frozen_risk,
                    "pair_id": pair_i,
                    "prompt1": p1,
                    "prompt2": p2,
                    "distance_quintile":
                        int(pr.distance_quintile),
                    "text_cosine_distance":
                        float(pr.text_cosine_distance),
                    "seed": seed,
                    "alpha": alpha,
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                    "heatmap_mean":
                        item["heatmap_mean"],
                    "heatmap_max":
                        item["heatmap_max"],
                    "target_norm":
                        item["target_norm"],
                    "corr_distance_from_A":
                        corr_a,
                    "corr_distance_from_B":
                        corr_b,
                    "heatmap_mae_from_A":
                        mae_a,
                    "heatmap_mae_from_B":
                        mae_b,
                })

                run_i += 1

                if run_i % 100 == 0:
                    print(
                        f"completed {run_i}/{expected}",
                        flush=True
                    )

            # Make the temporary nature explicit.
            del trajectory, h_a, h_b

out = pd.DataFrame(rows)

assert len(out) == expected

out.to_csv(
    OUT,
    index=False
)

print("\n=== COMPLETE ===", flush=True)
print("rows:", len(out), flush=True)
print("expected:", expected, flush=True)
print("saved:", OUT, flush=True)
print("pair file:", PAIR_OUT, flush=True)

print("\nDEVELOPMENT DIAGNOSTIC ONLY: YES", flush=True)
print("GROUND TRUTH USED: NO", flush=True)
print("DICE OUTCOMES USED: NO", flush=True)
print("HELDOUT OUTCOMES ACCESSED: NO", flush=True)
print("METHOD/THRESHOLD TUNING: NO", flush=True)
print(
    "PAIR SELECTION USED LOCALIZATION OUTCOMES: NO",
    flush=True
)
