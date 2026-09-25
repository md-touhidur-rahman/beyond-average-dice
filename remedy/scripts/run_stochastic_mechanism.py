import os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image

ROOT = Path("/home/hpc/rlvl/rlvl178v/prl_medclipsam")
V2   = ROOT / "MedCLIP-SAMv2"
DATA = ROOT / "MedCLIP-SAM/part2/action400_dataset"

PILOT = ROOT / "beyond-average-dice/remedy/results/stochastic_pilot/pilot_cases.csv"
P20   = ROOT / "beyond-average-dice/results/brain_p20/brain400_p20.csv"
SPLIT = ROOT / "beyond-average-dice/remedy/results/brain400_slice_partition.csv"

OUTDIR = ROOT / "beyond-average-dice/remedy/results/stochastic_mechanism"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "mechanism_boxes.csv"
HEATOUT = OUTDIR / "raw_heatmaps_float16.npz"

sys.path.insert(0, str(V2 / "saliency_maps"))
sys.path.insert(0, str(V2))

from transformers import AutoModel, AutoProcessor, AutoTokenizer
from scripts.methods import vision_heatmap_iba

DEVICE = "cuda"

VLAYER = 9
VBETA  = 2.0
VVAR   = 0.3

SEEDS = [12, 23, 34, 45, 56]

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
assert torch.cuda.is_available()

# ------------------------------------------------------------
# Inputs / leakage gates
# ------------------------------------------------------------

pilot = pd.read_csv(PILOT)
frozen = pd.read_csv(P20)
split = pd.read_csv(SPLIT)

assert len(pilot) == 12
assert pilot["filename"].nunique() == 12

dev = set(
    split.loc[
        split["split"].astype(str).str.lower().eq("development"),
        "filename"
    ].astype(str)
)

assert set(pilot["filename"].astype(str)).issubset(dev)

# We deliberately use ONLY frozen localization coordinates.
needed = ["filename"]
for p in range(1, 21):
    tag = f"P{p:02d}"
    needed += [
        f"{tag}_x1", f"{tag}_y1",
        f"{tag}_x2", f"{tag}_y2"
    ]

assert all(c in frozen.columns for c in needed)

frozen_boxes = frozen[needed].copy().set_index("filename")

print("=== DESIGN ===", flush=True)
print("cases:", len(pilot), flush=True)
print("prompts:", len(PROMPTS), flush=True)
print("seeds:", SEEDS, flush=True)
print("planned runs:", len(pilot) * len(PROMPTS) * len(SEEDS), flush=True)
print("GPU:", torch.cuda.get_device_name(0), flush=True)

print("\n=== SAFETY / LEAKAGE ===", flush=True)
print("ALL CASES DEVELOPMENT: YES", flush=True)
print("GROUND TRUTH LOADED: NO", flush=True)
print("DICE OUTCOMES USED: NO", flush=True)
print("HELDOUT OUTCOMES ACCESSED: NO", flush=True)
print("METHOD/THRESHOLD TUNING: NO", flush=True)

# ------------------------------------------------------------
# Frozen model
# ------------------------------------------------------------

torch.manual_seed(12)
np.random.seed(12)

print("\n=== LOADING MODEL ===", flush=True)

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

def m2ib_single(image, text):
    image_feat = processor(
        images=image,
        return_tensors="pt"
    )["pixel_values"].to(DEVICE)

    ids = torch.tensor(
        [tokenizer.encode(text, add_special_tokens=True)],
        device=DEVICE
    )

    vmap = vision_heatmap_iba(
        ids,
        image_feat,
        model,
        VLAYER,
        VBETA,
        VVAR,
        ensemble=False,
        progbar=False
    )

    arr = np.asarray(vmap, dtype=np.float32)

    arr = cv2.resize(
        arr,
        image.size,
        interpolation=cv2.INTER_NEAREST
    )

    return np.clip(arr, 0, 1)

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

    bg = np.argmin(km.cluster_centers_.flatten())
    m = (lab != bg).astype(np.uint8)

    m = cv2.resize(
        m,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m)

    if n <= 1:
        return m

    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))

    return (labels == idx).astype(np.uint8)

def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        h, w = mask.shape
        return np.array([0, 0, w-1, h-1], dtype=np.float32)

    return np.array(
        [xs.min(), ys.min(), xs.max(), ys.max()],
        dtype=np.float32
    )

# ------------------------------------------------------------
# Factorial run
# ------------------------------------------------------------

rows = []
heatmaps = []
heat_keys = []
gate_total = 0
gate_exact = 0
gate_failures = []

for ci, prow in pilot.reset_index(drop=True).iterrows():

    fn = str(prow["filename"])
    region = str(prow["risk_region"])
    frozen_risk = float(prow["frozen_risk"])

    print(
        f"\n=== CASE {ci+1}/12: {fn} "
        f"region={region} frozen_risk={frozen_risk:.6f} ===",
        flush=True
    )

    image = Image.open(
        DATA / "test_images" / fn
    ).convert("RGB")

    for pidx, prompt in enumerate(PROMPTS, start=1):

        tag = f"P{pidx:02d}"

        stored = np.array([
            frozen_boxes.loc[fn, f"{tag}_x1"],
            frozen_boxes.loc[fn, f"{tag}_y1"],
            frozen_boxes.loc[fn, f"{tag}_x2"],
            frozen_boxes.loc[fn, f"{tag}_y2"],
        ], dtype=np.float32)

        for seed in SEEDS:

            # Exact original RNG control, generalized over seeds.
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

            vm = m2ib_single(image, prompt)

            # Preserve raw M2IB output BEFORE KMeans/component selection.
            heatmaps.append(vm.astype(np.float16))
            heat_keys.append(f"{fn}|{tag}|{seed}")

            cm = kmeans_mask(vm)
            box = bbox_from_mask(cm)

            exact_seed12 = np.nan

            if seed == 12:
                gate_total += 1
                exact_seed12 = bool(np.array_equal(box, stored))

                if exact_seed12:
                    gate_exact += 1
                else:
                    gate_failures.append({
                        "filename": fn,
                        "prompt": tag,
                        "stored": stored.tolist(),
                        "regenerated": box.tolist()
                    })

            rows.append({
                "filename": fn,
                "risk_region": region,
                "frozen_risk": frozen_risk,
                "prompt_id": tag,
                "seed": seed,
                "x1": float(box[0]),
                "y1": float(box[1]),
                "x2": float(box[2]),
                "y2": float(box[3]),
                "seed12_exact_frozen": exact_seed12,
            })

            print(
                f"{tag} seed={seed:02d} "
                f"box={box.tolist()}",
                flush=True
            )

        # Incremental checkpoint after every prompt.
        pd.DataFrame(rows).to_csv(OUT, index=False)

print("\n=== SAVING RAW HEATMAPS ===", flush=True)

heatmaps_arr = np.stack(heatmaps, axis=0)

assert heatmaps_arr.shape[0] == 1200
assert len(heat_keys) == 1200
assert len(set(heat_keys)) == 1200

np.savez_compressed(
    HEATOUT,
    heatmaps=heatmaps_arr,
    keys=np.asarray(heat_keys)
)

print("heatmap array shape:", heatmaps_arr.shape, flush=True)
print("heatmap dtype:", heatmaps_arr.dtype, flush=True)
print("heatmap file:", HEATOUT, flush=True)

print("\n=== FACTORIAL COMPLETE ===", flush=True)
print("rows:", len(rows), flush=True)
print("expected:", 12 * 20 * 5, flush=True)

assert len(rows) == 1200

print("\n=== SEED-12 FULL REPRODUCTION GATE ===", flush=True)
print("exact:", gate_exact, "/", gate_total, flush=True)
print("failures:", len(gate_failures), flush=True)

if gate_failures:
    for x in gate_failures:
        print("FAIL:", x, flush=True)

print("\nALL CASES DEVELOPMENT: YES", flush=True)
print("GROUND TRUTH LOADED: NO", flush=True)
print("DICE OUTCOMES USED: NO", flush=True)
print("HELDOUT OUTCOMES ACCESSED: NO", flush=True)
print("METHOD/THRESHOLD TUNING: NO", flush=True)

if gate_exact != 240:
    raise RuntimeError(
        f"FULL REPRODUCTION GATE FAILED: {gate_exact}/240 exact"
    )

print("\nFULL REPRODUCTION GATE: PASS", flush=True)
print("saved:", OUT, flush=True)
