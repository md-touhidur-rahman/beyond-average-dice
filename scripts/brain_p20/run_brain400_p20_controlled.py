import os, sys
import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image

ROOT = "/home/hpc/rlvl/rlvl178v/prl_medclipsam"
V2   = f"{ROOT}/MedCLIP-SAMv2"
DATA = f"{ROOT}/MedCLIP-SAM/part2/action400_dataset"
OUT  = f"{ROOT}/MedCLIP-SAM/part2/brain400_p20_controlled"
CSVOUT = f"{OUT}/brain400_p20.csv"

os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, f"{V2}/saliency_maps")
sys.path.insert(0, V2)

from transformers import AutoModel, AutoProcessor, AutoTokenizer
from scripts.methods import vision_heatmap_iba

sys.path.insert(0, f"{V2}/segment-anything")
from segment_anything import sam_model_registry, SamPredictor

DEVICE = "cuda"

# Exact frozen official Brain P2 prompts.
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

# Frozen published Brain M2IB parameters.
VLAYER = 9
VBETA  = 2.0
VVAR   = 0.3
SEED   = 12

torch.manual_seed(SEED)
np.random.seed(SEED)

print("Loading BiomedCLIP/M2IB...", flush=True)

model = AutoModel.from_pretrained(
    f"{V2}/saliency_maps/model",
    trust_remote_code=True
).to(DEVICE)

processor = AutoProcessor.from_pretrained(
    f"{ROOT}/chuhac_BiomedCLIP-vit-bert-hf",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    f"{ROOT}/chuhac_BiomedCLIP-vit-bert-hf",
    trust_remote_code=True
)

model.eval()

print("Loading SAM-B...", flush=True)

sam_b = sam_model_registry["vit_b"](
    checkpoint=f"{ROOT}/MedCLIP-SAM/checkpoints/sam_vit_b_01ec64.pth"
).to(DEVICE)

pred_b = SamPredictor(sam_b)


def dice(mask, gt):
    a = mask.astype(bool)
    b = gt.astype(bool)
    den = a.sum() + b.sum()
    if den == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / den)


def iou(mask, gt):
    a = mask.astype(bool)
    b = gt.astype(bool)
    u = np.logical_or(a, b).sum()
    if u == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / u)


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
            [0, 0, w-1, h-1],
            dtype=np.float32
        )

    return np.array(
        [xs.min(), ys.min(), xs.max(), ys.max()],
        dtype=np.float32
    )


def sam_from_current_image(predictor, box):
    masks, scores, _ = predictor.predict(
        box=box,
        multimask_output=True
    )

    k = int(np.argmax(scores))

    return masks[k], float(scores[k])


files = sorted(
    x for x in os.listdir(f"{DATA}/test_images")
    if x.lower().endswith((".png", ".jpg", ".jpeg"))
)

assert len(files) == 400, len(files)

print("N =", len(files), flush=True)

rows = []

for n, fn in enumerate(files, 1):

    print(
        f"\n[{n}/{len(files)}] {fn}",
        flush=True
    )

    image = Image.open(
        f"{DATA}/test_images/{fn}"
    ).convert("RGB")

    rgb = np.asarray(image)

    gt = cv2.imread(
        f"{DATA}/test_masks/{fn}",
        cv2.IMREAD_GRAYSCALE
    ) > 0

    # SAM image embedding is identical for all 20 prompts.
    pred_b.set_image(rgb)

    row = {"filename": fn}

    for pidx, prompt in enumerate(
        PROMPTS,
        start=1
    ):
        tag = f"P{pidx:02d}"

        # CRITICAL CONTROL:
        # Every language condition receives exactly the same
        # initial RNG state.
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        np.random.seed(SEED)

        vm = m2ib_single(
            image,
            prompt
        )

        cm = kmeans_mask(vm)
        box = bbox_from_mask(cm)

        mask, conf = sam_from_current_image(
            pred_b,
            box
        )

        d = dice(mask, gt)
        j = iou(mask, gt)

        row[f"{tag}_dice"] = d
        row[f"{tag}_iou"] = j
        row[f"{tag}_sam_conf"] = conf

        row[f"{tag}_x1"] = float(box[0])
        row[f"{tag}_y1"] = float(box[1])
        row[f"{tag}_x2"] = float(box[2])
        row[f"{tag}_y2"] = float(box[3])

        print(
            f"  {tag}: "
            f"Dice={d:.4f} "
            f"IoU={j:.4f} "
            f"conf={conf:.3f}",
            flush=True
        )

    rows.append(row)

    # Incremental checkpoint.
    pd.DataFrame(rows).to_csv(
        CSVOUT,
        index=False
    )

    print(
        "checkpoint:",
        CSVOUT,
        flush=True
    )

print("\nBRAIN P20 COMPLETE", flush=True)
print("Saved:", CSVOUT, flush=True)
