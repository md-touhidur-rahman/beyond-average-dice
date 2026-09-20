import os
import sys
import numpy as np
import pandas as pd
import cv2
import torch

ROOT = "/home/hpc/rlvl/rlvl178v/prl_medclipsam"
MED  = f"{ROOT}/MedCLIP-SAM"
V2   = f"{ROOT}/MedCLIP-SAMv2"

DATA = f"{MED}/part2/action400_dataset"
P20  = f"{MED}/part2/brain400_p20_controlled/brain400_p20.csv"
PAIRS = f"{MED}/part2/brain400_p20_controlled/high_box_overlap_residual_pairs.csv"

OUT = f"{MED}/part2/brain400_p20_controlled/residual_sam_candidates.csv"

sys.path.insert(0, f"{V2}/segment-anything")
from segment_anything import sam_model_registry, SamPredictor

DEVICE = "cuda"


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


print("Loading frozen SAM-B...", flush=True)

sam = sam_model_registry["vit_b"](
    checkpoint=f"{MED}/checkpoints/sam_vit_b_01ec64.pth"
).to(DEVICE)

predictor = SamPredictor(sam)

p20 = pd.read_csv(P20)
pairs = pd.read_csv(PAIRS)

p20 = p20.set_index("filename")

# Unique case-prompt evaluations required by the residual pairs.
needed = set()

for _, r in pairs.iterrows():
    needed.add((r["filename"], r["prompt_a"]))
    needed.add((r["filename"], r["prompt_b"]))

print("Residual pairs:", len(pairs), flush=True)
print("Unique cases:", pairs["filename"].nunique(), flush=True)
print("Unique case-prompt evaluations:", len(needed), flush=True)

# Group requested prompts by image so SAM image embedding
# is computed only once per affected case.
by_case = {}

for fn, tag in sorted(needed):
    by_case.setdefault(fn, []).append(tag)

rows = []

for ci, (fn, tags) in enumerate(sorted(by_case.items()), 1):

    print(
        f"[{ci}/{len(by_case)}] {fn} "
        f"({len(tags)} prompts)",
        flush=True
    )

    rgb = cv2.imread(
        f"{DATA}/test_images/{fn}",
        cv2.IMREAD_COLOR
    )

    if rgb is None:
        raise RuntimeError(f"Could not read image: {fn}")

    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    gt = cv2.imread(
        f"{DATA}/test_masks/{fn}",
        cv2.IMREAD_GRAYSCALE
    )

    if gt is None:
        raise RuntimeError(f"Could not read mask: {fn}")

    gt = gt > 0

    predictor.set_image(rgb)

    src = p20.loc[fn]

    for tag in sorted(tags):

        box = np.array([
            src[f"{tag}_x1"],
            src[f"{tag}_y1"],
            src[f"{tag}_x2"],
            src[f"{tag}_y2"],
        ], dtype=np.float32)

        masks, scores, _ = predictor.predict(
            box=box,
            multimask_output=True
        )

        cand_dice = [
            dice(masks[k], gt)
            for k in range(len(masks))
        ]

        cand_iou = [
            iou(masks[k], gt)
            for k in range(len(masks))
        ]

        selected_idx = int(np.argmax(scores))
        oracle_idx = int(np.argmax(cand_dice))

        selected_dice = float(cand_dice[selected_idx])
        oracle_dice = float(cand_dice[oracle_idx])

        old_dice = float(src[f"{tag}_dice"])
        old_conf = float(src[f"{tag}_sam_conf"])

        row = {
            "filename": fn,
            "prompt": tag,

            "x1": float(box[0]),
            "y1": float(box[1]),
            "x2": float(box[2]),
            "y2": float(box[3]),

            "old_selected_dice": old_dice,
            "rerun_selected_dice": selected_dice,
            "rerun_selected_iou":
                float(cand_iou[selected_idx]),

            "old_sam_conf": old_conf,

            "selected_idx": selected_idx,
            "oracle_idx": oracle_idx,

            "oracle_candidate_dice": oracle_dice,
            "selection_gap":
                oracle_dice - selected_dice,
        }

        for k in range(len(masks)):
            row[f"cand{k}_score"] = float(scores[k])
            row[f"cand{k}_dice"] = float(cand_dice[k])
            row[f"cand{k}_iou"] = float(cand_iou[k])

        rows.append(row)

        print(
            f"  {tag}: "
            f"selected={selected_idx} "
            f"D={selected_dice:.4f} "
            f"oracle={oracle_dice:.4f} "
            f"gap={oracle_dice-selected_dice:.4f}",
            flush=True
        )

pd.DataFrame(rows).to_csv(
    OUT,
    index=False
)

print("\nRESIDUAL SAM COMPLETE", flush=True)
print("Saved:", OUT, flush=True)
