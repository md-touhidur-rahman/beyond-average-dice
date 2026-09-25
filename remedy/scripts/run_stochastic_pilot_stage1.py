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

OUTDIR = ROOT / "beyond-average-dice/remedy/results/stochastic_pilot"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT = OUTDIR / "stage1_p01_multiseed.csv"

sys.path.insert(0, str(V2 / "saliency_maps"))
sys.path.insert(0, str(V2))

from transformers import AutoModel, AutoProcessor, AutoTokenizer
from scripts.methods import vision_heatmap_iba

DEVICE = "cuda"

VLAYER = 9
VBETA = 2.0
VVAR = 0.3

# Frozen seed plus four prespecified alternative stochastic replicates.
SEEDS = [12, 23, 34, 45, 56]

PROMPT = (
    "A medical brain MRI scan revealing a suspicious, irregularly "
    "shaped mass suggestive of a brain tumor."
)

assert torch.cuda.is_available(), "CUDA unavailable"

# ------------------------------------------------------------
# Guards
# ------------------------------------------------------------

pilot = pd.read_csv(PILOT)
p20 = pd.read_csv(P20)
split = pd.read_csv(SPLIT)

assert len(pilot) == 12
assert pilot["filename"].nunique() == 12

dev_names = set(
    split.loc[
        split["split"].astype(str).str.lower().eq("development"),
        "filename"
    ].astype(str)
)

assert len(dev_names) == 268
assert set(pilot["filename"].astype(str)).issubset(dev_names)

p20 = p20.set_index("filename")

print("=== STOCHASTIC PILOT STAGE 1 ===", flush=True)
print("cases:", len(pilot), flush=True)
print("prompt: P01", flush=True)
print("seeds:", SEEDS, flush=True)
print("evaluations:", len(pilot) * len(SEEDS), flush=True)
print("GPU:", torch.cuda.get_device_name(0), flush=True)

# ------------------------------------------------------------
# Exact frozen model loading
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
        return np.array(
            [0, 0, w - 1, h - 1],
            dtype=np.float32
        )

    return np.array(
        [xs.min(), ys.min(), xs.max(), ys.max()],
        dtype=np.float32
    )


def box_iou(a, b):
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))

    iw = max(0.0, x2 - x1 + 1.0)
    ih = max(0.0, y2 - y1 + 1.0)
    inter = iw * ih

    aa = max(0.0, float(a[2]-a[0]+1.0)) * \
         max(0.0, float(a[3]-a[1]+1.0))
    bb = max(0.0, float(b[2]-b[0]+1.0)) * \
         max(0.0, float(b[3]-b[1]+1.0))

    union = aa + bb - inter

    return inter / union if union > 0 else 1.0


rows = []

for case_idx, r in pilot.iterrows():

    fn = str(r["filename"])

    print(
        f"\n[{case_idx+1}/{len(pilot)}] {fn}",
        flush=True
    )

    image = Image.open(
        DATA / "test_images" / fn
    ).convert("RGB")

    stored = np.array([
        p20.loc[fn, "P01_x1"],
        p20.loc[fn, "P01_y1"],
        p20.loc[fn, "P01_x2"],
        p20.loc[fn, "P01_y2"],
    ], dtype=np.float32)

    for seed in SEEDS:

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        vm = m2ib_single(image, PROMPT)
        cm = kmeans_mask(vm)
        box = bbox_from_mask(cm)

        rec = {
            "filename": fn,
            "risk_region": r["risk_region"],
            "frozen_risk": float(r["frozen_risk"]),
            "prompt_id": "P01",
            "seed": seed,
            "x1": float(box[0]),
            "y1": float(box[1]),
            "x2": float(box[2]),
            "y2": float(box[3]),
            "foreground_pixels": int(cm.sum()),
            "heatmap_mean": float(vm.mean()),
            "iou_vs_frozen_P01": box_iou(box, stored),
            "exact_frozen_P01": bool(np.array_equal(box, stored)),
        }

        rows.append(rec)

        print(
            f" seed={seed:2d}"
            f" box={box.tolist()}"
            f" IoU_frozen={rec['iou_vs_frozen_P01']:.6f}"
            f" exact={rec['exact_frozen_P01']}",
            flush=True
        )

        # Incremental checkpoint.
        pd.DataFrame(rows).to_csv(OUT, index=False)

res = pd.DataFrame(rows)

assert len(res) == 60
assert res.groupby(["filename", "seed"]).size().eq(1).all()

# Critical provenance check:
# seed 12 must reproduce the frozen P01 box for every pilot case.
seed12 = res.loc[res["seed"].eq(12)]

print("\n=== SEED-12 REPRODUCTION ===", flush=True)
print(
    "exact:",
    int(seed12["exact_frozen_P01"].sum()),
    "/",
    len(seed12),
    flush=True
)
print(
    "mean IoU vs frozen:",
    f"{seed12['iou_vs_frozen_P01'].mean():.9f}",
    flush=True
)

if not seed12["exact_frozen_P01"].all():
    bad = seed12.loc[
        ~seed12["exact_frozen_P01"],
        ["filename", "iou_vs_frozen_P01", "x1", "y1", "x2", "y2"]
    ]
    print("\nFAILED SEED-12 CASES:", flush=True)
    print(bad.to_string(index=False), flush=True)
    raise RuntimeError(
        "STAGE-1 PROVENANCE FAILURE: seed 12 did not reproduce "
        "all frozen P01 localizations."
    )

# Pairwise stochastic box IoU within each case.
pairs = []

for fn, g in res.groupby("filename"):

    g = g.sort_values("seed").reset_index(drop=True)

    for i in range(len(g)):
        for j in range(i + 1, len(g)):

            a = g.loc[i, ["x1","y1","x2","y2"]].to_numpy(float)
            b = g.loc[j, ["x1","y1","x2","y2"]].to_numpy(float)

            pairs.append({
                "filename": fn,
                "risk_region": g.loc[i, "risk_region"],
                "frozen_risk": g.loc[i, "frozen_risk"],
                "seed_a": int(g.loc[i, "seed"]),
                "seed_b": int(g.loc[j, "seed"]),
                "box_iou": box_iou(a, b),
            })

pairs = pd.DataFrame(pairs)

assert len(pairs) == 12 * 10

pairs.to_csv(
    OUTDIR / "stage1_p01_seed_pairwise_iou.csv",
    index=False
)

case_summary = (
    pairs.groupby(
        ["filename", "risk_region", "frozen_risk"],
        as_index=False
    )
    .agg(
        mean_within_prompt_iou=("box_iou", "mean"),
        min_within_prompt_iou=("box_iou", "min"),
        max_within_prompt_iou=("box_iou", "max"),
    )
)

case_summary["stochastic_risk"] = (
    1.0 - case_summary["mean_within_prompt_iou"]
)

case_summary.to_csv(
    OUTDIR / "stage1_case_summary.csv",
    index=False
)

print("\n=== WITHIN-P01 STOCHASTIC STABILITY ===", flush=True)
print(case_summary.to_string(index=False), flush=True)

print("\n=== OVERALL ===", flush=True)
print(
    "mean pairwise seed IoU:",
    f"{pairs['box_iou'].mean():.6f}",
    flush=True
)
print(
    "median pairwise seed IoU:",
    f"{pairs['box_iou'].median():.6f}",
    flush=True
)
print(
    "minimum pairwise seed IoU:",
    f"{pairs['box_iou'].min():.6f}",
    flush=True
)

print("\nDEVELOPMENT ONLY: YES", flush=True)
print("GROUND TRUTH USED: NO", flush=True)
print("HELDOUT OUTCOMES ACCESSED: NO", flush=True)
print("METHOD/THRESHOLD TUNING: NO", flush=True)
