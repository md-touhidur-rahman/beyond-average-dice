import os, sys, csv
import numpy as np
import cv2
import torch
from PIL import Image

ROOT = "/home/hpc/rlvl/rlvl178v/prl_medclipsam"
V2 = f"{ROOT}/MedCLIP-SAMv2"
DATA = f"{ROOT}/MedCLIP-SAM/part2/breast98_dataset"
OUT = f"{ROOT}/MedCLIP-SAM/part2/breast97_p20_controlled"
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, f"{V2}/saliency_maps")
sys.path.insert(0, V2)

from transformers import AutoModel, AutoProcessor, AutoTokenizer
from scripts.methods import vision_heatmap_iba

sys.path.insert(0, f"{V2}/segment-anything")
from segment_anything import sam_model_registry, SamPredictor

DEVICE = "cuda"

# ------------------------------------------------------------
# FROZEN BEFORE OUTCOMES:
# exact 20 official Breast P2 descriptions already used by
# the existing A2 ensemble experiment.
# Each description is now evaluated INDIVIDUALLY.
# ------------------------------------------------------------
PROMPTS = [
    'A medical breast mammogram revealing an area of concern suggestive of a breast tumor.',
    'A mammogram displaying a mass in the breast that may indicate a tumor.',
    'A breast mammogram showing an abnormal mass potentially indicative of a tumor.',
    'A breast imaging study showing a concerning mass suggestive of a tumor.',
    'A medical mammogram identifying a distinct mass that could suggest a breast tumor.',
    'A breast mammogram with findings suggestive of a mass potentially indicating a tumor.',
    'A mammogram showing a prominent mass within the breast tissue suggestive of a tumor.',
    'A breast imaging scan revealing a mass that may be indicative of a tumor.',
    'A mammogram showing a significant area of density that could indicate a breast tumor.',
    'A medical breast scan revealing an unusual mass suggestive of a tumor.',
    'A breast mammogram detecting a mass that raises suspicion of a tumor.',
    'A breast imaging study showing a suspicious area that could represent a tumor.',
    'A medical mammogram highlighting a mass that may be indicative of a breast tumor.',
    'A breast imaging scan identifying a notable mass suggestive of a tumor.',
    'A mammogram showing an irregular mass in the breast that could indicate a tumor.',
    'A breast mammogram displaying a dense mass that may be suggestive of a tumor.',
    'A breast imaging study revealing an area of concern suggestive of a potential tumor.',
    'A medical mammogram detecting a suspicious mass that could indicate a breast tumor.',
    'A mammogram showing an area of increased density that may be indicative of a tumor.',
    'A breast scan revealing a concerning mass suggestive of a breast tumor.',
]
assert len(PROMPTS) == 20
assert len(set(PROMPTS)) == 20

# Same frozen M2IB parameters as existing Breast experiment.
VLAYER = 9
VBETA = 2.0
VVAR = 0.3

torch.manual_seed(12)
np.random.seed(12)

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
    return float(
        2.0 * np.logical_and(a, b).sum() / den
    )


def iou(mask, gt):
    a = mask.astype(bool)
    b = gt.astype(bool)
    u = np.logical_or(a, b).sum()
    if u == 0:
        return 1.0
    return float(
        np.logical_and(a, b).sum() / u
    )


# IMPORTANT:
# single-prompt path only.
# ensemble=False for EVERY P01...P20.
def m2ib_single(image, text):
    image_feat = processor(
        images=image,
        return_tensors="pt"
    )["pixel_values"].to(DEVICE)

    ids = torch.tensor(
        [tokenizer.encode(
            text,
            add_special_tokens=True
        )],
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
        [
            xs.min(),
            ys.min(),
            xs.max(),
            ys.max()
        ],
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
    x for x in os.listdir(
        f"{DATA}/test_images"
    )
    if x.lower().endswith(
        (".png", ".jpg", ".jpeg")
    )
)

EXCLUDED = {"benign_000062.png"}
files = [
    x for x in files
    if x not in EXCLUDED
]

assert len(files) == 97
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

    # Compute SAM image embedding ONCE.
    pred_b.set_image(rgb)

    row = {"filename": fn}

    for pidx, prompt in enumerate(
        PROMPTS,
        start=1
    ):
        tag = f"P{pidx:02d}"

        # Controlled prompt experiment:
        # identical RNG state before every M2IB evaluation,
        # so prompt text is the intended changing variable.
        torch.manual_seed(12)
        torch.cuda.manual_seed_all(12)
        np.random.seed(12)

        vm = m2ib_single(
            image,
            prompt
        )

        cm = kmeans_mask(vm)
        box = bbox_from_mask(cm)

        mask, conf = \
            sam_from_current_image(
                pred_b,
                box
            )

        d = dice(mask, gt)
        j = iou(mask, gt)

        row[f"{tag}_dice"] = d
        row[f"{tag}_iou"] = j
        row[f"{tag}_sam_conf"] = conf

        # Box coordinates are useful for mechanism analysis.
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

        del vm, cm, mask

    rows.append(row)

    # Incremental checkpoint: don't lose completed cases.
    csvout = f"{OUT}/breast97_p20.csv"

    with open(
        csvout,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        "checkpoint:",
        csvout,
        flush=True
    )

print("\nP20 COMPLETE")
print(
    f"Saved: {OUT}/breast97_p20.csv"
)
