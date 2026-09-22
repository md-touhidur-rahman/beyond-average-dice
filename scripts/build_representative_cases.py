#!/usr/bin/env python3

from pathlib import Path
import ast
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path("/home/hpc/rlvl/rlvl178v/prl_medclipsam")
V2   = ROOT / "MedCLIP-SAMv2"
DATA = ROOT / "authors_segmentation_data"
BASE = V2 / "parent_repro_brain600"
RES  = ROOT / "beyond-average-dice/results/brain600_language_ladder"
OUT  = ROOT / "beyond-average-dice/paper_results/FINAL_FIGURES"
OUT.mkdir(parents=True, exist_ok=True)

AUDIT = RES / "representative_case_audit.csv"

# ------------------------------------------------------------
# Locked scientific examples
# ------------------------------------------------------------

CASES = [
    {
        "case": "1401",
        "panel": "a",
        "phenomenon": "Localization switch",
        "conditions": ["H1", "H2"],
    },
    {
        "case": "2982",
        "panel": "b",
        "phenomenon": "High-overlap residual",
        "conditions": ["H0", "H1"],
    },
    {
        "case": "762",
        "panel": "c",
        "phenomenon": "Failure and recovery",
        "conditions": ["H0", "L4", "L5"],
    },
    {
        "case": "2816",
        "panel": "d",
        "phenomenon": "Broad-description rescue",
        "conditions": ["H0", "L5"],
    },
]

COND_LABEL = {
    "H0": "H0 original",
    "H1": "H1 verb",
    "H2": "H2 intro",
    "L3": "L3 reformulation",
    "L4": "L4 compressed",
    "L5": "L5 broad",
}

# Exact experiment directories.
COND_DIR = {
    "H0": BASE,
    "H1": BASE / "highop/H1_verb",
    "H2": BASE / "highop/H2_intro",
    "L3": BASE / "language_ladder/L3_semantic",
    "L4": BASE / "language_ladder/L4_concise",
    "L5": BASE / "language_ladder/L5_broad",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
})

# ------------------------------------------------------------
# IO
# ------------------------------------------------------------

def read_image(path):
    x = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if x is None:
        raise FileNotFoundError(path)

    if x.ndim == 3:
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

    return x

def read_mask(path):
    x = read_image(path)
    return x > 0

def image_path(case):
    return DATA / f"data/brain_tumors/test_images/{case}.png"

def gt_path(case):
    return DATA / f"data/brain_tumors/test_masks/{case}.png"

def sam_path(case, cond):
    return COND_DIR[cond] / "sam" / f"{case}.png"

def coarse_path(case, cond):
    return COND_DIR[cond] / "coarse" / f"{case}.png"

# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def dice_score(a, b):
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)

    denom = a.sum() + b.sum()

    if denom == 0:
        return 1.0

    return 2.0 * np.logical_and(a, b).sum() / denom

def bbox(mask):
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()),
        int(ys.max()),
    )

def box_iou(a, b):
    if a is None and b is None:
        return 1.0

    if a is None or b is None:
        return 0.0

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1 + 1)
    ih = max(0, iy2 - iy1 + 1)

    inter = iw * ih
    aa = (ax2-ax1+1) * (ay2-ay1+1)
    bb = (bx2-bx1+1) * (by2-by1+1)

    union = aa + bb - inter

    return inter / union if union else 0.0

# ------------------------------------------------------------
# Display helpers
# ------------------------------------------------------------

def normalize_image(x):
    x = x.astype(float)

    lo, hi = np.percentile(x, [1, 99])

    if hi <= lo:
        lo, hi = x.min(), x.max()

    if hi <= lo:
        return np.zeros_like(x, dtype=float)

    return np.clip((x-lo)/(hi-lo), 0, 1)

def overlay_prediction(image, gt, pred):
    """
    Grayscale MRI with contour overlays only.
    GT = solid contour
    prediction = dashed contour

    No filled masks, so anatomy remains visible.
    """
    return normalize_image(image)

def draw_contour(ax, mask, linestyle="-", linewidth=1.1):
    if mask.any():
        ax.contour(
            mask.astype(float),
            levels=[0.5],
            linewidths=linewidth,
            linestyles=linestyle,
        )

def draw_box(ax, b):
    if b is None:
        return

    x1, y1, x2, y2 = b

    ax.add_patch(
        Rectangle(
            (x1, y1),
            x2-x1+1,
            y2-y1+1,
            fill=False,
            linewidth=1.0,
        )
    )

# ------------------------------------------------------------
# Load frozen audit
# ------------------------------------------------------------

audit = pd.read_csv(AUDIT)
audit["case"] = audit["case"].astype(str)

needed = {
    (cfg["case"], c)
    for cfg in CASES
    for c in cfg["conditions"]
}

available = set(zip(audit["case"], audit["condition"]))

missing = needed - available

if missing:
    raise RuntimeError(
        f"Missing audit rows: {sorted(missing)}"
    )

# ------------------------------------------------------------
# Validate every source file BEFORE plotting
# ------------------------------------------------------------

for cfg in CASES:
    case = cfg["case"]

    required = [
        image_path(case),
        gt_path(case),
    ]

    for cond in cfg["conditions"]:
        required.extend([
            sam_path(case, cond),
            coarse_path(case, cond),
        ])

    for p in required:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing authentic source file: {p}"
            )

print("ALL AUTHENTIC SOURCE FILES PRESENT")

# ------------------------------------------------------------
# Recompute scientific quantities
# ------------------------------------------------------------

rows = []

for cfg in CASES:
    case = cfg["case"]

    image = read_image(image_path(case))
    gt = read_mask(gt_path(case))

    h0_box = bbox(read_mask(coarse_path(case, "H0")))

    for cond in cfg["conditions"]:

        pred = read_mask(sam_path(case, cond))
        coarse = read_mask(coarse_path(case, cond))

        d = float(dice_score(gt, pred))
        b = bbox(coarse)
        biou = float(box_iou(h0_box, b))

        frozen = audit[
            (audit["case"] == case) &
            (audit["condition"] == cond)
        ]

        if len(frozen) != 1:
            raise RuntimeError(
                f"Expected one audit row: {case} {cond}"
            )

        frozen = frozen.iloc[0]

        frozen_dice = float(frozen["dice"])
        frozen_iou = float(frozen["box_iou_vs_H0"])

        # Strong check: plotted SAM mask must reproduce
        # the frozen case-level Dice.
        if not np.isclose(d, frozen_dice, atol=2e-3):
            raise RuntimeError(
                f"DICE MISMATCH {case} {cond}: "
                f"mask={d:.8f}, frozen={frozen_dice:.8f}"
            )

        if not np.isclose(biou, frozen_iou, atol=2e-3):
            raise RuntimeError(
                f"BOX IOU MISMATCH {case} {cond}: "
                f"coarse={biou:.8f}, frozen={frozen_iou:.8f}"
            )

        rows.append({
            "case": case,
            "phenomenon": cfg["phenomenon"],
            "condition": cond,
            "dice_recomputed": d,
            "dice_frozen": frozen_dice,
            "box_iou_vs_H0_recomputed": biou,
            "box_iou_vs_H0_frozen": frozen_iou,
            "box": str(b),
            "image_path": str(image_path(case)),
            "gt_path": str(gt_path(case)),
            "sam_path": str(sam_path(case, cond)),
            "coarse_path": str(coarse_path(case, cond)),
            "prompt": frozen["prompt"],
        })

figdata = pd.DataFrame(rows)

figdata.to_csv(
    OUT / "Fig9_representative_cases_data.csv",
    index=False
)

print()
print("AUTHENTIC MASK VALIDATION PASSED")
print(
    figdata[
        [
            "case",
            "condition",
            "dice_recomputed",
            "box_iou_vs_H0_recomputed",
        ]
    ].to_string(index=False)
)

# ------------------------------------------------------------
# Publication figure
#
# Each row:
# MRI | Ground truth | selected condition(s)
#
# Variable number of condition panels is handled with blank cells.
# ------------------------------------------------------------

max_conditions = max(
    len(cfg["conditions"]) for cfg in CASES
)

ncols = 2 + max_conditions

fig, axes = plt.subplots(
    len(CASES),
    ncols,
    figsize=(7.15, 8.25),
    gridspec_kw={
        "wspace": .035,
        "hspace": .28,
    }
)

for r, cfg in enumerate(CASES):

    case = cfg["case"]
    conds = cfg["conditions"]

    image = read_image(image_path(case))
    gt = read_mask(gt_path(case))

    display = normalize_image(image)

    # --------------------------------------------------------
    # MRI
    # --------------------------------------------------------

    ax = axes[r, 0]

    ax.imshow(
        display,
        cmap="gray",
        interpolation="nearest"
    )

    ax.set_title("MRI")

    ax.text(
        -.10, 1.03,
        cfg["panel"],
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
    )

    ax.text(
        .5, -.055,
        f'Case {case} — {cfg["phenomenon"]}',
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        fontweight="bold",
    )

    ax.axis("off")

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    ax = axes[r, 1]

    ax.imshow(
        display,
        cmap="gray",
        interpolation="nearest"
    )

    draw_contour(
        ax,
        gt,
        linestyle="-",
        linewidth=1.25
    )

    ax.set_title("Ground truth")
    ax.axis("off")

    # --------------------------------------------------------
    # Conditions
    # --------------------------------------------------------

    for j, cond in enumerate(conds):

        ax = axes[r, j+2]

        pred = read_mask(sam_path(case, cond))
        coarse = read_mask(coarse_path(case, cond))
        b = bbox(coarse)

        rr = figdata[
            (figdata["case"] == case) &
            (figdata["condition"] == cond)
        ].iloc[0]

        ax.imshow(
            display,
            cmap="gray",
            interpolation="nearest"
        )

        # GT solid
        draw_contour(
            ax,
            gt,
            linestyle="-",
            linewidth=1.15
        )

        # prediction dashed
        draw_contour(
            ax,
            pred,
            linestyle="--",
            linewidth=1.15
        )

        # localization box
        draw_box(ax, b)

        if cond == "H0":
            subtitle = (
                f'{COND_LABEL[cond]}\n'
                f'Dice {rr.dice_recomputed:.3f}'
            )
        else:
            subtitle = (
                f'{COND_LABEL[cond]}\n'
                f'Dice {rr.dice_recomputed:.3f} | '
                f'box IoU {rr.box_iou_vs_H0_recomputed:.3f}'
            )

        ax.set_title(
            subtitle,
            fontsize=7.2,
            linespacing=1.15
        )

        ax.axis("off")

    # unused cells
    for j in range(len(conds), max_conditions):
        axes[r, j+2].axis("off")

# ------------------------------------------------------------
# Figure-level legend
# ------------------------------------------------------------

fig.text(
    .5, .006,
    "Solid contour: ground truth    "
    "Dashed contour: SAM prediction    "
    "Rectangle: text-conditioned localization box",
    ha="center",
    va="bottom",
    fontsize=7.5,
)

pdf = OUT / "Fig9_representative_cases.pdf"
png = OUT / "Fig9_representative_cases.png"

fig.savefig(
    pdf,
    bbox_inches="tight",
    pad_inches=.03
)

fig.savefig(
    png,
    dpi=600,
    bbox_inches="tight",
    pad_inches=.03
)

plt.close(fig)

print()
print("WROTE", pdf)
print("WROTE", png)
print("WROTE", OUT / "Fig9_representative_cases_data.csv")
print()
print("="*76)
print("REPRESENTATIVE CASE FIGURE COMPLETE")
print("="*76)
