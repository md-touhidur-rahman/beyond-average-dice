import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path("/home/hpc/rlvl/rlvl178v/prl_medclipsam")
V2   = ROOT / "MedCLIP-SAMv2"
DATA = ROOT / "MedCLIP-SAM/part2/action400_dataset"

P20 = ROOT / "beyond-average-dice/results/brain_p20/brain400_p20.csv"
SPLIT = ROOT / "beyond-average-dice/remedy/results/brain400_slice_partition.csv"

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
    vision_heatmap_iba,
)
from iba_embedding_intervention import IBAInterpreter

DEVICE = "cuda"

VLAYER = 9
VBETA  = 2.0
VVAR   = 0.3
SEED   = 12

PROMPT = (
    "A medical brain MRI scan revealing a suspicious, irregularly "
    "shaped mass suggestive of a brain tumor."
)

assert torch.cuda.is_available()

# ------------------------------------------------------------
# Same development case used by the original reproduction gate.
# ------------------------------------------------------------

sp = pd.read_csv(SPLIT)
df = pd.read_csv(P20)

dev = sp.loc[
    sp["split"].astype(str).str.lower().eq("development"),
    "filename"
].astype(str).tolist()

assert len(dev) == 268

fn = dev[0]

src = df.set_index("filename").loc[fn]

stored_box = np.array(
    [
        src["P01_x1"],
        src["P01_y1"],
        src["P01_x2"],
        src["P01_y2"],
    ],
    dtype=np.float32
)

print("=== DESIGN ===", flush=True)
print("filename:", fn, flush=True)
print("prompt: P01", flush=True)
print("seed:", SEED, flush=True)
print("stored box:", stored_box.tolist(), flush=True)

# ------------------------------------------------------------
# Frozen model.
# ------------------------------------------------------------

torch.manual_seed(SEED)
np.random.seed(SEED)

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

image = Image.open(
    DATA / "test_images" / fn
).convert("RGB")

image_feat = processor(
    images=image,
    return_tensors="pt"
)["pixel_values"].to(DEVICE)

ids = torch.tensor(
    [
        tokenizer.encode(
            PROMPT,
            add_special_tokens=True
        )
    ],
    device=DEVICE
)

# Projected text representation that the ordinary IBA path uses.
with torch.no_grad():
    text_target = model.get_text_features(ids).detach()

print("text target shape:", tuple(text_target.shape), flush=True)
print(
    "text target norm:",
    float(text_target.norm().item()),
    flush=True
)

# ------------------------------------------------------------
# Helpers: exact frozen post-processing.
# ------------------------------------------------------------

def resize_map(vmap):
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
            [0, 0, w - 1, h - 1],
            dtype=np.float32
        )

    return np.array(
        [xs.min(), ys.min(), xs.max(), ys.max()],
        dtype=np.float32
    )


# ------------------------------------------------------------
# A. Original IBA.
# ------------------------------------------------------------

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

print("\n=== ORIGINAL TOKEN PATH ===", flush=True)

orig = vision_heatmap_iba(
    ids,
    image_feat,
    model,
    VLAYER,
    VBETA,
    VVAR,
    ensemble=False,
    progbar=False
)

orig = resize_map(orig)
orig_mask = kmeans_mask(orig)
orig_box = bbox_from_mask(orig_mask)

print("box:", orig_box.tolist(), flush=True)
print(
    "map min/mean/max:",
    float(orig.min()),
    float(orig.mean()),
    float(orig.max()),
    flush=True
)

# ------------------------------------------------------------
# B. Embedding-target intervention path.
#
# Reconstruct a fresh compression estimator exactly as
# vision_heatmap_iba() does.
# ------------------------------------------------------------

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

print("\n=== EMBEDDING TARGET PATH ===", flush=True)

features = extract_feature_map(
    model.vision_model,
    VLAYER,
    image_feat
)

layer = extract_bert_layer(
    model.vision_model,
    VLAYER
)

compression_estimator = get_compression_estimator(
    VVAR,
    layer,
    features
)

reader = IBAInterpreter(
    model,
    compression_estimator,
    beta=VBETA,
    lr=1,
    steps=10,
    ensemble=False,
    progbar=False
)

inter = reader.vision_heatmap_embedding_target(
    text_target,
    image_feat
)

inter = resize_map(inter)
inter_mask = kmeans_mask(inter)
inter_box = bbox_from_mask(inter_mask)

print("box:", inter_box.tolist(), flush=True)
print(
    "map min/mean/max:",
    float(inter.min()),
    float(inter.mean()),
    float(inter.max()),
    flush=True
)

# ------------------------------------------------------------
# Gate.
# ------------------------------------------------------------

absdiff = np.abs(orig - inter)

print("\n=== ENDPOINT EQUIVALENCE ===", flush=True)

print(
    "heatmap MAE:",
    float(absdiff.mean()),
    flush=True
)

print(
    "heatmap max abs difference:",
    float(absdiff.max()),
    flush=True
)

corr = np.corrcoef(
    orig.reshape(-1),
    inter.reshape(-1)
)[0, 1]

print(
    "heatmap correlation:",
    float(corr),
    flush=True
)

print(
    "original vs intervention box exact:",
    bool(np.array_equal(orig_box, inter_box)),
    flush=True
)

print(
    "original vs frozen box exact:",
    bool(np.array_equal(orig_box, stored_box)),
    flush=True
)

print(
    "intervention vs frozen box exact:",
    bool(np.array_equal(inter_box, stored_box)),
    flush=True
)

assert np.array_equal(
    orig_box,
    stored_box
), "Original reproduction failed."

assert np.array_equal(
    inter_box,
    stored_box
), "Embedding intervention endpoint failed."

assert np.array_equal(
    orig_box,
    inter_box
), "Endpoint boxes disagree."

print("\nENDPOINT INTERVENTION GATE: PASS", flush=True)

print("\nDEVELOPMENT CASE: YES", flush=True)
print("GROUND TRUTH USED: NO", flush=True)
print("DICE OUTCOMES USED: NO", flush=True)
print("HELDOUT OUTCOMES ACCESSED: NO", flush=True)
print("METHOD/THRESHOLD TUNING: NO", flush=True)
