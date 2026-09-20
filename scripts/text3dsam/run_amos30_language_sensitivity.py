from pathlib import Path
import csv
import time

import monai.transforms as mtf
import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertTokenizer

from src.model.modeling import Med3DSeg

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/ct_amos30"
INPUT_DIR = DATA / "3D_val_npz"
GT_DIR = DATA / "3D_val_gt/3D_val_gt_text"

CKPT = ROOT / "checkpoints/Text3DSAM_offline"
TOKENIZER = ROOT / "checkpoints/Text3DSAM"

OUTDIR = ROOT / "part2"
OUTDIR.mkdir(exist_ok=True)
OUTCSV = OUTDIR / "text3dsam_amos30_liver_language.csv"

TARGET = 6

# FROZEN BEFORE FULL-COHORT INFERENCE.
PROMPTS = {
    "P0": "Visualization of the liver in abdominal CT imaging",
    "P1": "Liver in abdominal CT",
    "P2": "Abdominal CT showing liver structures",
    "P3": "CT imaging of the liver in the abdomen",
}

device = torch.device("cuda")
print("GPU:", torch.cuda.get_device_name(0), flush=True)

model = Med3DSeg.from_pretrained(
    str(CKPT),
    local_files_only=True,
).to(device).eval()

tokenizer = BertTokenizer.from_pretrained(
    str(TOKENIZER),
    local_files_only=True,
)

cases = sorted(INPUT_DIR.glob("CT_AMOS_*.npz"))
assert len(cases) == 30, len(cases)

pre = mtf.Compose([
    mtf.EnsureChannelFirst(channel_dim="no_channel"),
    mtf.Resize(
        spatial_size=model.config.image_size,
        mode="trilinear",
    ),
    mtf.ToTensor(dtype=torch.float32),
])


def get_prob(image, prompt, origin_shape):
    tokens = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model.generate(
            image=image,
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
        )

        logits = F.interpolate(
            logits,
            size=origin_shape,
            mode="trilinear",
        )

        return torch.sigmoid(logits)[0, 0]


def dice(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    den = pred.sum() + gt.sum()
    if den == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, gt).sum() / den)


rows = []

for ci, input_path in enumerate(cases, 1):
    t0 = time.time()

    name = input_path.name
    gt_path = GT_DIR / name
    assert gt_path.exists(), gt_path

    npz = np.load(input_path, allow_pickle=True)
    image_np = npz["imgs"].astype(np.float32)
    official = npz["text_prompts"].item()

    assert int(official["instance_label"]) == 0

    gt_labels = np.load(
        gt_path,
        allow_pickle=True,
    )["gts"]

    gt_target = gt_labels == TARGET
    assert gt_target.sum() > 0, (
        name,
        "liver absent from GT",
    )

    origin_shape = image_np.shape
    image = pre(image_np).unsqueeze(0).to(device)

    # -------------------------------------------------------
    # Compute the 14 UNCHANGED organ probabilities once.
    # This is identical across P0-P3.
    # -------------------------------------------------------
    max_other = None

    for k, prompt in official.items():
        if k == "instance_label":
            continue

        class_id = int(k)

        if class_id == TARGET:
            continue

        prob = get_prob(
            image,
            prompt,
            origin_shape,
        )

        if max_other is None:
            max_other = prob
        else:
            max_other = torch.maximum(max_other, prob)

        del prob

    assert max_other is not None

    result = {
        "case": name,
        "gt_voxels": int(gt_target.sum()),
    }

    # -------------------------------------------------------
    # Only liver language changes.
    #
    # Official multiclass rule:
    #   background = 1 - max(all foreground)
    #   argmax(background, organ probabilities)
    #
    # For deciding whether label 6 wins, liver probability
    # must beat every other organ AND background.
    # -------------------------------------------------------
    for pid, prompt in PROMPTS.items():
        liver_prob = get_prob(
            image,
            prompt,
            origin_shape,
        )

        max_fg = torch.maximum(
            max_other,
            liver_prob,
        )

        background = torch.clamp(
            1.0 - max_fg,
            min=0.0,
            max=1.0,
        )

        # Liver wins the same argmax used by official pred.py.
        pred_target = (
            (liver_prob > max_other)
            & (liver_prob > background)
        ).cpu().numpy()

        result[f"dice_{pid}"] = dice(
            pred_target,
            gt_target,
        )
        result[f"voxels_{pid}"] = int(
            pred_target.sum()
        )

    # Pairwise sensitivity.
    vals = [result[f"dice_P{i}"] for i in range(4)]

    result["range_dice"] = max(vals) - min(vals)
    result["mean_abs_vs_P0"] = float(
        np.mean([
            abs(vals[1] - vals[0]),
            abs(vals[2] - vals[0]),
            abs(vals[3] - vals[0]),
        ])
    )
    result["oracle_dice"] = max(vals)
    result["oracle_gain_over_P0"] = max(vals) - vals[0]
    result["best_prompt"] = f"P{int(np.argmax(vals))}"

    rows.append(result)

    del image, max_other
    torch.cuda.empty_cache()

    print(
        f"[{ci:02d}/30] {name} "
        f"P0={vals[0]:.4f} "
        f"P1={vals[1]:.4f} "
        f"P2={vals[2]:.4f} "
        f"P3={vals[3]:.4f} "
        f"range={result['range_dice']:.4f} "
        f"time={time.time()-t0:.1f}s",
        flush=True,
    )

    # Write incrementally so partial results survive interruption.
    with open(OUTCSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


print("\n=== FINAL SUMMARY ===")

for pid in PROMPTS:
    x = np.array(
        [r[f"dice_{pid}"] for r in rows]
    )
    print(
        pid,
        "mean=", f"{x.mean():.6f}",
        "median=", f"{np.median(x):.6f}",
    )

p0 = np.array([r["dice_P0"] for r in rows])
ranges = np.array([r["range_dice"] for r in rows])
sens = np.array([r["mean_abs_vs_P0"] for r in rows])
oracle = np.array([r["oracle_dice"] for r in rows])

print("mean range:", f"{ranges.mean():.6f}")
print("median range:", f"{np.median(ranges):.6f}")
print("mean abs vs P0:", f"{sens.mean():.6f}")
print("median abs vs P0:", f"{np.median(sens):.6f}")

for t in [0.01, 0.05, 0.10, 0.20]:
    print(
        f"range > {t:.2f}:",
        int((ranges > t).sum()),
        "/",
        len(rows),
    )

print(
    "P0 mean:",
    f"{p0.mean():.6f}",
)
print(
    "oracle mean:",
    f"{oracle.mean():.6f}",
)
print(
    "oracle gain:",
    f"{(oracle.mean()-p0.mean()):.6f}",
)

print("CSV:", OUTCSV)
