import os, sys
import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image

ROOT = "/home/hpc/rlvl/rlvl178v/prl_medclipsam"
V2   = f"{ROOT}/MedCLIP-SAMv2"
DATA = f"{ROOT}/MedCLIP-SAM/part2/action400_dataset"

P20 = f"{ROOT}/beyond-average-dice/results/brain_p20/brain400_p20.csv"
SPLIT = f"{ROOT}/beyond-average-dice/remedy/results/brain400_slice_partition.csv"

sys.path.insert(0, f"{V2}/saliency_maps")
sys.path.insert(0, V2)

from transformers import AutoModel, AutoProcessor, AutoTokenizer
from scripts.methods import vision_heatmap_iba

DEVICE = "cuda"

VLAYER = 9
VBETA  = 2.0
VVAR   = 0.3
SEED   = 12

PROMPT = (
    "A medical brain MRI scan revealing a suspicious, irregularly "
    "shaped mass suggestive of a brain tumor."
)

assert torch.cuda.is_available(), "CUDA unavailable on allocated node"

# ------------------------------------------------------------
# Development-only case selection.
# ------------------------------------------------------------

df = pd.read_csv(P20)
sp = pd.read_csv(SPLIT)

assert "filename" in df.columns
assert "filename" in sp.columns
assert "split" in sp.columns

dev = sp.loc[
    sp["split"].astype(str).str.lower().eq("development"),
    "filename"
].astype(str).tolist()

assert len(dev) == 268, len(dev)

fn = dev[0]

src = df.set_index("filename").loc[fn]

# Access ONLY stored localization coordinates.
stored_box = np.array([
    src["P01_x1"],
    src["P01_y1"],
    src["P01_x2"],
    src["P01_y2"],
], dtype=np.float32)

print("=== PROVENANCE ===", flush=True)
print("filename:", fn, flush=True)
print("prompt: P01", flush=True)
print("stored box:", stored_box.tolist(), flush=True)
print("M2IB layer:", VLAYER, flush=True)
print("beta:", VBETA, flush=True)
print("variance:", VVAR, flush=True)
print("seed:", SEED, flush=True)
print("GPU:", torch.cuda.get_device_name(0), flush=True)

# ------------------------------------------------------------
# Exact frozen model loading.
# ------------------------------------------------------------

torch.manual_seed(SEED)
np.random.seed(SEED)

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

# ------------------------------------------------------------
# Exact frozen image loading.
# ------------------------------------------------------------

image = Image.open(
    f"{DATA}/test_images/{fn}"
).convert("RGB")

# ------------------------------------------------------------
# Exact original functions.
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# Exact original per-prompt RNG reset.
# ------------------------------------------------------------

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

print("\n=== REGENERATING ===", flush=True)

vm = m2ib_single(image, PROMPT)
cm = kmeans_mask(vm)
regen_box = bbox_from_mask(cm)

print("heatmap shape:", vm.shape, flush=True)
print(
    "heatmap min/mean/max:",
    float(vm.min()),
    float(vm.mean()),
    float(vm.max()),
    flush=True
)
print("foreground pixels:", int(cm.sum()), flush=True)
print("regenerated box:", regen_box.tolist(), flush=True)

diff = regen_box - stored_box
exact = bool(np.array_equal(regen_box, stored_box))

print("\n=== REPRODUCIBILITY GATE ===", flush=True)
print("coordinate difference:", diff.tolist(), flush=True)
print("max abs difference:", float(np.abs(diff).max()), flush=True)
print("exact match:", exact, flush=True)

print("\nDEVELOPMENT CASE: YES", flush=True)
print("GROUND TRUTH USED: NO", flush=True)
print("HELDOUT OUTCOMES ACCESSED: NO", flush=True)

if not exact:
    raise RuntimeError(
        "REPRODUCTION GATE FAILED: "
        f"stored={stored_box.tolist()} "
        f"regenerated={regen_box.tolist()}"
    )

print("\nREPRODUCTION GATE: PASS", flush=True)
